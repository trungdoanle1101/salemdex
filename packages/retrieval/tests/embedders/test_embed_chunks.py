import uuid
from collections.abc import Sequence

import pytest

from salemdex_retrieval.chunks import Chunk, EmbeddedChunk
from salemdex_retrieval.embedders.protocol import (
    DenseBinding,
    SparseBinding,
    embed_chunks,
)
from salemdex_retrieval.schema import One, Schema
from salemdex_retrieval.vectors import Dense, DenseValue, Sparse, SparseValue


class Chunks(Schema):
    position = One(int)
    body = Dense(size=2, distance="cosine")
    title = Dense(size=2, distance="dot")


class Entries(Schema):
    position = One(int)
    body = Dense(size=2, distance="cosine")


class Hybrid(Schema):
    position = One(int)
    body = Dense(size=2, distance="cosine")
    terms = Sparse(modifier="idf")


class RecordingEmbedder:
    """A dense embedder of size 2 that records every batch it is given.

    A text's row is `[its length, marker]`, so each row can be traced to its text,
    and two embedders with different markers to their vector.
    """

    def __init__(self, *, marker: float = 1.0, extra_rows: int = 0) -> None:
        self.marker = marker
        self.extra_rows = extra_rows
        self.passage_batches: list[list[str]] = []

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(len(text)), self.marker] for text in texts]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        self.passage_batches.append(list(texts))
        rows = [[float(len(text)), self.marker] for text in texts]
        if self.extra_rows >= 0:
            return rows + [[0.0, self.marker]] * self.extra_rows
        return rows[: self.extra_rows]


class RecordingSparseEmbedder:
    """A sparse embedder that records every batch it is given.

    A text's weights are `{its length: marker}`, so each mapping can be traced to its
    text.
    """

    def __init__(self, *, marker: float = 1.0, extra_rows: int = 0) -> None:
        self.marker = marker
        self.extra_rows = extra_rows
        self.passage_batches: list[list[str]] = []

    def embed_queries(self, texts: Sequence[str]) -> list[dict[int, float]]:
        return [{len(text): self.marker} for text in texts]

    def embed_passages(self, texts: Sequence[str]) -> list[dict[int, float]]:
        self.passage_batches.append(list(texts))
        rows = [{len(text): self.marker} for text in texts]
        if self.extra_rows >= 0:
            return rows + [{}] * self.extra_rows
        return rows[: self.extra_rows]


def build_chunks(count: int) -> list[Chunk]:
    """Return `count` chunks whose embedded texts have different lengths."""
    return [
        Chunk(
            id=uuid.uuid4(),
            text=f"text {position}",
            embedded_text="x" * (position + 1),
            field_values=[Chunks.position.value(position)],
        )
        for position in range(count)
    ]


def test_each_chunk_gets_the_embedding_of_its_embedded_text() -> None:
    chunks = build_chunks(3)

    embedded = embed_chunks(
        chunks,
        bindings=[DenseBinding(vector=Chunks.body, embedder=RecordingEmbedder())],
    )

    assert [item.chunk for item in embedded] == chunks
    assert [numbers_of(item, Chunks.body) for item in embedded] == [
        (float(len(chunk.embedded_text)), 1.0) for chunk in chunks
    ]


def test_the_embedder_gets_every_chunk_in_one_batch() -> None:
    chunks = build_chunks(3)
    embedder = RecordingEmbedder()

    embed_chunks(chunks, bindings=[DenseBinding(vector=Chunks.body, embedder=embedder)])

    assert embedder.passage_batches == [[chunk.embedded_text for chunk in chunks]]


def test_each_binding_fills_its_own_vector() -> None:
    chunks = build_chunks(2)

    embedded = embed_chunks(
        chunks,
        bindings=[
            DenseBinding(vector=Chunks.body, embedder=RecordingEmbedder(marker=1.0)),
            DenseBinding(vector=Chunks.title, embedder=RecordingEmbedder(marker=2.0)),
        ],
    )

    for item in embedded:
        assert numbers_of(item, Chunks.body)[1] == 1.0
        assert numbers_of(item, Chunks.title)[1] == 2.0
        item.check_fits(Chunks)


def test_no_chunks_give_nothing_without_calling_the_embedder() -> None:
    embedder = RecordingEmbedder()

    embedded = embed_chunks(
        [], bindings=[DenseBinding(vector=Chunks.body, embedder=embedder)]
    )

    assert embedded == []
    assert embedder.passage_batches == []


def test_no_bindings_give_chunks_without_vectors_for_a_schema_without_vectors() -> None:
    class Lookup(Schema):
        position = One(int)

    chunk = Chunk(id=uuid.uuid4(), text="text", field_values=[Lookup.position.value(0)])

    [embedded] = embed_chunks([chunk], bindings=[])

    assert embedded.chunk == chunk
    assert embedded.vector_values == {}
    embedded.check_fits(Lookup)


@pytest.mark.parametrize(
    "second_vector", [Chunks.body, Entries.body], ids=["same vector", "same name"]
)
def test_two_bindings_for_one_vector_name_are_refused_before_embedding(
    second_vector: Dense,
) -> None:
    embedder = RecordingEmbedder()
    bindings = [
        DenseBinding(vector=Chunks.body, embedder=embedder),
        DenseBinding(vector=second_vector, embedder=embedder),
    ]

    with pytest.raises(ValueError, match="body"):
        embed_chunks(build_chunks(2), bindings=bindings)

    assert embedder.passage_batches == []


@pytest.mark.parametrize("extra_rows", [-1, 1], ids=["too few", "too many"])
def test_an_embedder_returning_the_wrong_number_of_rows_is_refused(
    extra_rows: int,
) -> None:
    embedder = RecordingEmbedder(extra_rows=extra_rows)

    # The message names the vector, so the faulty embedder can be found among several
    with pytest.raises(ValueError, match="body"):
        embed_chunks(
            build_chunks(3),
            bindings=[DenseBinding(vector=Chunks.body, embedder=embedder)],
        )


def test_a_row_of_the_wrong_size_is_refused() -> None:
    class WrongSize(RecordingEmbedder):
        def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
            return [[1.0, 2.0, 3.0] for _ in texts]

    with pytest.raises(ValueError, match="size"):
        embed_chunks(
            build_chunks(2),
            bindings=[DenseBinding(vector=Chunks.body, embedder=WrongSize())],
        )


def numbers_of(embedded: EmbeddedChunk, vector: Dense) -> tuple[float, ...]:
    """Return the numbers of `embedded`'s value for the dense `vector`."""
    value = embedded.vector_values[vector]
    assert isinstance(value, DenseValue)
    return value.numbers


# Sparse bindings


def test_a_sparse_binding_fills_its_vector_with_each_texts_weights() -> None:
    chunks = build_chunks(3)
    embedder = RecordingSparseEmbedder(marker=0.5)

    embedded = embed_chunks(
        chunks,
        bindings=[
            DenseBinding(vector=Hybrid.body, embedder=RecordingEmbedder()),
            SparseBinding(vector=Hybrid.terms, embedder=embedder),
        ],
    )

    assert [weights_of(item, Hybrid.terms) for item in embedded] == [
        {len(chunk.embedded_text): 0.5} for chunk in chunks
    ]
    assert embedder.passage_batches == [[chunk.embedded_text for chunk in chunks]]


@pytest.mark.parametrize("extra_rows", [-1, 1], ids=["too few", "too many"])
def test_a_sparse_embedder_returning_the_wrong_number_of_rows_is_refused(
    extra_rows: int,
) -> None:
    embedder = RecordingSparseEmbedder(extra_rows=extra_rows)

    with pytest.raises(ValueError, match="terms"):
        embed_chunks(
            build_chunks(3),
            bindings=[SparseBinding(vector=Hybrid.terms, embedder=embedder)],
        )


def test_a_sparse_row_with_an_id_that_is_not_an_int_is_refused() -> None:
    class StringIds(RecordingSparseEmbedder):
        def embed_passages(self, texts: Sequence[str]) -> list[dict[int, float]]:
            # Some libraries give ids as strings; the adapter must convert them
            return [{"4865": 1.0} for _ in texts]  # type: ignore[dict-item]

    with pytest.raises(TypeError, match="'4865'"):
        embed_chunks(
            build_chunks(2),
            bindings=[SparseBinding(vector=Hybrid.terms, embedder=StringIds())],
        )


def weights_of(embedded: EmbeddedChunk, vector: Sparse) -> dict[int, float]:
    """Return the weights of `embedded`'s value for the sparse `vector`."""
    value = embedded.vector_values[vector]
    assert isinstance(value, SparseValue)
    return dict(value.weights)
