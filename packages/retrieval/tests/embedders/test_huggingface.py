"""`HuggingFaceDenseEmbedder` with a real model; its settings and loading are tested with
the reranker's, in tests/test_huggingface.py.

They run only when KB_TEST_EMBEDDER_MODEL names a model, as a hub id or a local folder,
e.g. `sentence-transformers/all-MiniLM-L6-v2`. They use English text, so any model can
run them.
"""

import math
import os
import uuid
from collections.abc import Sequence

import pytest

from salemdex_retrieval.chunks import Chunk
from salemdex_retrieval.embedders.huggingface import HuggingFaceDenseEmbedder
from salemdex_retrieval.embedders.protocol import (
    DenseBinding,
    DenseEmbedder,
    embed_chunks,
)
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


def _close(row: Sequence[float], other: Sequence[float]) -> bool:
    """Return whether two rows are equal up to rounding."""
    return len(row) == len(other) and all(
        math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-5) for a, b in zip(row, other)
    )
