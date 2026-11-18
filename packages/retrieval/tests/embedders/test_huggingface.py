"""`HuggingFaceDenseEmbedder`: its settings and loading, and a real model when one is set.

The tests with a model run only when KB_TEST_EMBEDDER_MODEL names one, as a hub id or a
local folder, e.g. `sentence-transformers/all-MiniLM-L6-v2`. They use English text, so
any model can run them.
"""

import math
import os
import re
import uuid
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from salemdex_retrieval.chunks import Chunk
from salemdex_retrieval.embedders.huggingface import (
    Device,
    HuggingFaceDenseEmbedder,
    ResolvedDevice,
    _resolve_device,
)
from salemdex_retrieval.embedders.protocol import (
    DenseBinding,
    DenseEmbedder,
    embed_chunks,
)
from salemdex_retrieval.errors import ModelLoadError
from salemdex_retrieval.filters import Everything
from salemdex_retrieval.index import InMemoryIndex
from salemdex_retrieval.schema import One, Schema
from salemdex_retrieval.vectors import Dense

MODEL = os.environ.get("KB_TEST_EMBEDDER_MODEL")

PASSAGES = [
    "Scope of application: this regulation applies to every employee of the company.",
    "Late payment of an invoice is charged interest of two percent per month.",
    "Annual leave is twelve working days for each full year of work.",
]


# Settings, checked before the model loads


@pytest.mark.parametrize("batch_size", [0, -1, True, 2.0])
def test_a_batch_size_that_is_not_a_positive_int_is_refused(
    batch_size: object, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        HuggingFaceDenseEmbedder(
            tmp_path / "missing",
            device="cpu",
            batch_size=batch_size,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_a_gpu_pytorch_cannot_find_is_refused_naming_it(
    device: Device, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_gpus(monkeypatch, cuda=False, mps=False)

    message = f"device {device!r} was asked for, but PyTorch found no"
    with pytest.raises(ValueError, match=re.escape(message)):
        HuggingFaceDenseEmbedder(tmp_path / "missing", device=device, batch_size=8)


@pytest.mark.parametrize(
    ("device", "cuda", "mps", "expected"),
    [
        ("auto", True, True, "cuda"),
        ("auto", False, True, "mps"),
        ("auto", False, False, "cpu"),
        ("cpu", True, True, "cpu"),
        ("cuda", True, False, "cuda"),
        ("mps", False, True, "mps"),
    ],
)
def test_the_device_is_the_one_asked_for_or_the_best_one_for_auto(
    device: Device,
    cuda: bool,
    mps: bool,
    expected: ResolvedDevice,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gpus(monkeypatch, cuda=cuda, mps=mps)

    assert _resolve_device(device) == expected


def test_a_model_that_cannot_be_found_raises_naming_it(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    message = f"can't load model {str(missing)!r} as a local folder or a hub model"
    with pytest.raises(ModelLoadError, match=re.escape(message)) as raised:
        HuggingFaceDenseEmbedder(missing, device="cpu", batch_size=8)

    assert isinstance(raised.value.__cause__, OSError)


def test_a_folder_that_is_not_a_model_raises_the_same_error(tmp_path: Path) -> None:
    not_a_model = tmp_path / "notes"
    not_a_model.mkdir()

    message = f"can't load model {str(not_a_model)!r} as a local folder or a hub model"
    with pytest.raises(ModelLoadError, match=re.escape(message)) as raised:
        HuggingFaceDenseEmbedder(not_a_model, device="cpu", batch_size=8)

    assert isinstance(raised.value.__cause__, ValueError)


def test_a_load_error_is_neither_an_os_error_nor_a_value_error() -> None:
    # Code catching a bad batch size (ValueError) or a file problem (OSError) mustn't
    # swallow a model that can't be loaded by accident.
    assert not issubclass(ModelLoadError, (OSError, ValueError))


# A real model


@pytest.fixture(scope="module")
def embedder() -> HuggingFaceDenseEmbedder:
    """Return the embedder of KB_TEST_EMBEDDER_MODEL, loaded once for the module."""
    if MODEL is None:
        pytest.skip("KB_TEST_EMBEDDER_MODEL is not set")
    # A batch smaller than PASSAGES, so a call spans batches.
    return HuggingFaceDenseEmbedder(MODEL, device="cpu", batch_size=2)


def test_it_is_a_dense_embedder(embedder: HuggingFaceDenseEmbedder) -> None:
    usable: DenseEmbedder = embedder

    assert usable is embedder


def test_each_passage_gets_one_row_of_plain_floats_of_one_length(
    embedder: HuggingFaceDenseEmbedder,
) -> None:
    rows = embedder.embed_passages(PASSAGES)

    assert len(rows) == len(PASSAGES)
    assert len({len(row) for row in rows}) == 1
    assert all(type(number) is float for row in rows for number in row)


def test_rows_come_back_in_the_order_of_the_texts(
    embedder: HuggingFaceDenseEmbedder,
) -> None:
    together = embedder.embed_passages(PASSAGES)
    one_by_one = [embedder.embed_passages([text])[0] for text in PASSAGES]

    # Padding in a batch changes the last digits, not the vector.
    for row, alone in zip(together, one_by_one, strict=True):
        assert _close(row, alone)


def test_a_query_row_is_as_long_as_a_passage_row(
    embedder: HuggingFaceDenseEmbedder,
) -> None:
    [query] = embedder.embed_queries(["Who does the regulation apply to?"])
    [passage] = embedder.embed_passages(PASSAGES[:1])

    assert len(query) == len(passage)


def test_no_texts_give_no_rows(embedder: HuggingFaceDenseEmbedder) -> None:
    assert list(embedder.embed_queries([])) == []
    assert list(embedder.embed_passages([])) == []


def test_a_search_finds_the_passage_that_answers_the_question(
    embedder: HuggingFaceDenseEmbedder,
) -> None:
    size = len(embedder.embed_passages(PASSAGES[:1])[0])

    class Chunks(Schema):
        position = One(int)
        dense = Dense(size=size, distance="cosine")

    chunks = [
        Chunk(
            id=uuid.UUID(int=i + 1), text=text, field_values=[Chunks.position.value(i)]
        )
        for i, text in enumerate(PASSAGES)
    ]
    index = InMemoryIndex(schema=Chunks)
    binding = DenseBinding(vector=Chunks.dense, embedder=embedder)
    index.put(embed_chunks(chunks, bindings=[binding]))

    [question] = embedder.embed_queries(["How many days of annual leave do I get?"])
    [best] = index.search(Chunks.dense.query(question), where=Everything(), limit=1)

    assert best.chunk.text == PASSAGES[2]


def _set_gpus(monkeypatch: pytest.MonkeyPatch, *, cuda: bool, mps: bool) -> None:
    """Make PyTorch report the given GPUs as available or not."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


def _close(row: Sequence[float], other: Sequence[float]) -> bool:
    """Return whether two rows are equal up to rounding."""
    return len(row) == len(other) and all(
        math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-5) for a, b in zip(row, other)
    )
