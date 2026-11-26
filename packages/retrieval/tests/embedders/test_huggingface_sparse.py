"""`HuggingFaceSparseEmbedder`: tiny models built here, and a real model when one is set.
Its settings and the loading it shares with the other adapters are tested in
tests/test_huggingface.py.

The tiny models have random weights from a fixed seed, so their weights mean nothing
but are the same on every run. They show what comes back, and the refusal of models
that aren't sparse encoders, with no download. The search test runs only when
KB_TEST_SPARSE_EMBEDDER_MODEL names a model, as a hub id or a local folder, e.g.
`naver/splade-v3`. It uses English text, so any English model can run it.
"""

import os
import re
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
import torch
from sentence_transformers import SentenceTransformer, SparseEncoder
from sentence_transformers.base.modules import Transformer
from sentence_transformers.sentence_transformer.modules import Pooling
from sentence_transformers.sparse_encoder.modules import SpladePooling
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizerFast

from salemdex_retrieval.chunks import Chunk
from salemdex_retrieval.embedders import SparseBinding, SparseEmbedder, embed_chunks
from salemdex_retrieval.embedders.huggingface import HuggingFaceSparseEmbedder
from salemdex_retrieval.errors import ModelLoadError
from salemdex_retrieval.filters import Everything
from salemdex_retrieval.index import InMemoryIndex
from salemdex_retrieval.schema import One, Schema
from salemdex_retrieval.vectors import Sparse

MODEL = os.environ.get("KB_TEST_SPARSE_EMBEDDER_MODEL")

WORDS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "annual", "leave", "days"]

TEXTS = ["annual leave", "days", "leave days annual"]

# Prepended to questions only, so a query's weights differ from a passage's
QUERY_PROMPT = "days "


# A tiny model


@pytest.fixture(scope="module")
def splade_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return the folder of a tiny SPLADE model with a query prompt, built once."""
    folder = tmp_path_factory.mktemp("tiny-splade")
    _build_splade(folder)
    return folder


@pytest.fixture(scope="module")
def tiny_embedder(splade_folder: Path) -> HuggingFaceSparseEmbedder:
    """Return a sparse embedder of the tiny SPLADE model, loaded once."""
    # A batch smaller than TEXTS, so a call spans batches.
    return HuggingFaceSparseEmbedder(splade_folder, device="cpu", batch_size=2)


@pytest.fixture(scope="module")
def encoder(splade_folder: Path) -> SparseEncoder:
    """Return the tiny SPLADE model loaded by sentence-transformers itself."""
    return SparseEncoder(str(splade_folder), device="cpu")


def test_it_is_a_sparse_embedder(tiny_embedder: HuggingFaceSparseEmbedder) -> None:
    usable: SparseEmbedder = tiny_embedder

    assert usable is tiny_embedder


def test_each_text_gets_one_mapping_of_plain_int_ids_to_plain_floats(
    tiny_embedder: HuggingFaceSparseEmbedder,
) -> None:
    for mappings in (
        tiny_embedder.embed_passages(TEXTS),
        tiny_embedder.embed_queries(TEXTS),
    ):
        assert len(mappings) == len(TEXTS)
        for weights in mappings:
            assert weights
            assert all(type(sparse_id) is int for sparse_id in weights)
            assert all(type(weight) is float for weight in weights.values())
            assert set(weights) <= set(range(len(WORDS)))


def test_a_passage_gets_the_weights_the_model_gives_its_document(
    tiny_embedder: HuggingFaceSparseEmbedder, encoder: SparseEncoder
) -> None:
    expected = _nonzero_by_row(encoder.encode_document(TEXTS))

    _assert_close(tiny_embedder.embed_passages(TEXTS), expected)


def test_a_question_gets_the_weights_the_model_gives_its_query(
    tiny_embedder: HuggingFaceSparseEmbedder, encoder: SparseEncoder
) -> None:
    expected = _nonzero_by_row(encoder.encode_query(TEXTS))

    _assert_close(tiny_embedder.embed_queries(TEXTS), expected)


def test_a_question_and_a_passage_are_encoded_each_their_own_way(
    tiny_embedder: HuggingFaceSparseEmbedder,
) -> None:
    [question] = tiny_embedder.embed_queries(["leave"])
    [passage] = tiny_embedder.embed_passages(["leave"])

    assert question != pytest.approx(passage)


def test_mappings_come_back_in_the_order_of_the_texts(
    tiny_embedder: HuggingFaceSparseEmbedder,
) -> None:
    together = tiny_embedder.embed_passages(TEXTS)
    one_by_one = [tiny_embedder.embed_passages([text])[0] for text in TEXTS]

    # Close, not equal: padding a batch moves float32 sums in their 7th digit
    _assert_close(together, one_by_one)


def test_no_texts_give_no_mappings(tiny_embedder: HuggingFaceSparseEmbedder) -> None:
    assert tiny_embedder.embed_passages([]) == []
    assert tiny_embedder.embed_queries([]) == []


def test_a_text_the_model_weighs_nothing_gets_an_empty_mapping(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "silent"
    _build_splade(folder, silent=True)
    embedder = HuggingFaceSparseEmbedder(folder, device="cpu", batch_size=2)

    assert embedder.embed_passages(TEXTS) == [{}, {}, {}]


def test_its_mappings_pass_the_checks_where_they_enter(
    tiny_embedder: HuggingFaceSparseEmbedder,
) -> None:
    class Docs(Schema):
        position = One(int)
        lexical = Sparse(modifier="none")

    chunks = [
        Chunk(id=uuid.uuid4(), text=text, field_values=[Docs.position.value(position)])
        for position, text in enumerate(TEXTS)
    ]

    embedded = embed_chunks(
        chunks, bindings=[SparseBinding(vector=Docs.lexical, embedder=tiny_embedder)]
    )

    assert len(embedded) == len(TEXTS)


# Models that aren't sparse encoders, refused when they load


def test_a_dense_model_is_refused_rather_than_converted(tmp_path: Path) -> None:
    folder = tmp_path / "dense"
    _build_dense(folder)

    message = f"{str(folder)!r} is a SentenceTransformer model, not a SparseEncoder"
    with pytest.raises(ModelLoadError, match=re.escape(message)):
        HuggingFaceSparseEmbedder(folder, device="cpu", batch_size=2)


def test_a_bare_transformers_checkpoint_is_refused_naming_it(tmp_path: Path) -> None:
    folder = tmp_path / "bare"
    _build_masked_language_model(folder)

    # The path as a string, as every load error names it, never PosixPath('...')
    named = rf"(^|\s){re.escape(repr(str(folder)))}"
    with pytest.raises(ModelLoadError, match=named) as raised:
        HuggingFaceSparseEmbedder(folder, device="cpu", batch_size=2)

    assert str(raised.value).endswith("not a SparseEncoder")


# A real model


@pytest.fixture(scope="module")
def embedder() -> HuggingFaceSparseEmbedder:
    """Return the embedder of KB_TEST_SPARSE_EMBEDDER_MODEL, loaded once."""
    if MODEL is None:
        pytest.skip("KB_TEST_SPARSE_EMBEDDER_MODEL is not set")
    return HuggingFaceSparseEmbedder(MODEL, device="cpu", batch_size=2)


def test_a_search_finds_the_passage_that_answers_the_question(
    embedder: HuggingFaceSparseEmbedder,
) -> None:
    class Docs(Schema):
        position = One(int)
        lexical = Sparse(modifier="none")

    passages = [
        "Scope of application: this regulation applies to every employee.",
        "Late payment of an invoice is charged interest of two percent per month.",
        "Annual leave is twelve working days for each full year of work.",
    ]
    chunks = [
        Chunk(id=uuid.uuid4(), text=text, field_values=[Docs.position.value(position)])
        for position, text in enumerate(passages)
    ]
    binding = SparseBinding(vector=Docs.lexical, embedder=embedder)
    index = InMemoryIndex(schema=Docs)
    index.put(embed_chunks(chunks, bindings=[binding]))
    [asked] = embedder.embed_queries(["How many days off do I get each year?"])

    [best] = index.search(Docs.lexical.query(asked), where=Everything(), limit=1)

    assert best.chunk[Docs.position] == 2


# Builders


def _build_splade(folder: Path, *, silent: bool = False) -> None:
    """Save a tiny SPLADE model (masked LM + max pooling) with a query prompt.

    `silent` gives every token a large negative logit, so SPLADE weighs nothing.
    """
    checkpoint = folder.parent / f"{folder.name}-checkpoint"
    _build_masked_language_model(checkpoint, silent=silent)
    SparseEncoder(
        modules=[
            Transformer(str(checkpoint), transformer_task="fill-mask"),
            SpladePooling(pooling_strategy="max"),
        ],
        device="cpu",
        prompts={"query": QUERY_PROMPT},
    ).save(str(folder))


def _build_dense(folder: Path) -> None:
    """Save a tiny dense sentence-transformers model (BERT + mean pooling)."""
    checkpoint = folder.parent / f"{folder.name}-checkpoint"
    _build_masked_language_model(checkpoint)
    SentenceTransformer(
        modules=[Transformer(str(checkpoint)), Pooling(8)], device="cpu"
    ).save(str(folder))


def _build_masked_language_model(folder: Path, *, silent: bool = False) -> None:
    """Save a tiny BERT masked language model and its tokenizer, as transformers does."""
    tokenizer = Tokenizer(
        models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    # transformers leaves these constructors untyped.
    PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=32,
    ).save_pretrained(folder)
    config = BertConfig(
        vocab_size=len(WORDS),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=16,
        max_position_embeddings=32,
    )
    # A fixed seed, so every run builds the same model; forked, so the seed doesn't
    # leak into other tests
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = BertForMaskedLM(config)  # type: ignore[no-untyped-call]
    if silent:
        with torch.no_grad():
            model.cls.predictions.bias.fill_(-100.0)
    model.save_pretrained(folder)


def _assert_close(
    actual: Sequence[Mapping[int, float]], expected: Sequence[Mapping[int, float]]
) -> None:
    """Assert both hold the same ids per row, with weights equal to float32 noise."""
    # pytest.approx compares a mapping, not a list of them; rel=1e-5: float32 sums
    # differ in the 7th digit
    assert [set(weights) for weights in actual] == [set(row) for row in expected]
    for weights, row in zip(actual, expected, strict=True):
        assert weights == pytest.approx(row, rel=1e-5)


def _nonzero_by_row(embeddings: torch.Tensor) -> list[dict[int, float]]:
    """Return each row's non-zero entries as id to weight, read from the dense form."""
    rows: list[list[float]] = embeddings.to_dense().tolist()
    return [
        {sparse_id: weight for sparse_id, weight in enumerate(row) if weight}
        for row in rows
    ]
