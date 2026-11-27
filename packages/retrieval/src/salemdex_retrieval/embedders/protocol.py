"""The interface every embedder adapter implements, and how chunks are embedded with it.

An embedder turns texts into numbers. The app supplies it as an adapter, a client for a
hosted model or a wrapper around a local one, speaking plain data only, so it never
needs to know the schema. A binding pairs a schema vector with the embedder that fills
it, and the framework checks every row through that vector (`Dense.value`,
`Sparse.value`).

Two kinds exist: dense embedders (`DenseEmbedder`, `DenseBinding`) and sparse ones
(`SparseEmbedder`, `SparseBinding`). `bind` builds either, choosing by the vector's
kind. Others become further protocols and bindings when a scan needs them, each binding
added to `VectorBinding` and to `bind`.
"""

from collections.abc import Mapping, Sequence, Sized
from dataclasses import dataclass
from typing import Protocol, assert_never, cast, overload

from salemdex_retrieval.chunks import Chunk, EmbeddedChunk
from salemdex_retrieval.vectors import Dense, Sparse, Vector, VectorValue


class DenseEmbedder(Protocol):
    """A model turning texts into dense rows of numbers, one row per text.

    The app implements it, for a hosted model or a local one. It speaks plain data
    only: the framework pairs it with a vector through `DenseBinding` and checks every
    row there, so an adapter needn't check sizes or values itself.

    Questions and passages go through separate methods because many models encode them
    differently (e.g. a `query: ` or `passage: ` prefix). Both must use the same model,
    so questions and passages are compared in one space. Both take batches, because a
    round trip to a hosted model costs more than the work.

    Nothing here names the model. The app chooses it, records which one filled an
    index, and checks that the two agree.
    """

    def embed_queries(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one row of numbers per question in `texts`, in the same order.

        Args:
            texts: The questions, as asked.

        Returns:
            One row per text, each as long as the model's output.
        """
        ...

    def embed_passages(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one row of numbers per passage in `texts`, in the same order.

        Args:
            texts: The passages: each chunk's `embedded_text`.

        Returns:
            One row per text, each as long as the model's output.
        """
        ...


class SparseEmbedder(Protocol):
    """A model turning texts into weights for a few ids out of many, one mapping a text.

    The app implements it: a learned model (SPLADE, bge-m3's lexical weights) or a
    counter such as BM25. An id stands for a dimension of the embedder's own space, a
    token of its vocabulary or a word's hash; turning terms into ids is the embedder's
    job, so the framework never owns a vocabulary. It speaks plain data only: the
    framework pairs it with a vector through `SparseBinding` and checks every mapping
    there, so an adapter needn't check ids or weights itself.

    Questions and passages go through separate methods because the two sides often
    differ: BM25 weighs a passage by its term counts and length but a question only by
    its terms, and some models run only on passages. Both take batches.

    The vector's `modifier` must suit the embedder, and nothing checks it: an embedder
    giving BM25's term-frequency part needs `"idf"`, or its weights lack IDF; a learned
    model needs `"none"`, or its weights are scaled twice. As with the model itself, the
    app chooses the pairing and records it.
    """

    def embed_queries(self, texts: Sequence[str]) -> Sequence[Mapping[int, float]]:
        """Return one mapping of id to weight per question in `texts`, in order.

        Args:
            texts: The questions, as asked.

        Returns:
            One mapping per text, holding only the ids that weigh something; an id
            left out weighs nothing.
        """
        ...

    def embed_passages(self, texts: Sequence[str]) -> Sequence[Mapping[int, float]]:
        """Return one mapping of id to weight per passage in `texts`, in order.

        Args:
            texts: The passages: each chunk's `embedded_text`.

        Returns:
            One mapping per text, in the same space of ids as the questions'; an id
            left out weighs nothing.
        """
        ...


# Every kind of binding. A closed union: each `match` over it ends in `assert_never`,
# so mypy points at every place that hasn't learned a new kind.
type VectorBinding = DenseBinding | SparseBinding


@dataclass(frozen=True, kw_only=True)
class DenseBinding:
    """A dense vector of a schema, paired with the embedder that fills it.

    The pairing keeps kinds together: a `Dense` vector can only be bound to a dense
    embedder, which a mapping from vectors to embedders couldn't promise once other
    kinds exist. The retrieve stage uses the same binding to embed questions.

    Attributes:
        vector: The schema vector the rows are for.
        embedder: The model producing them, for passages and questions alike.
    """

    vector: Dense
    embedder: DenseEmbedder


@dataclass(frozen=True, kw_only=True)
class SparseBinding:
    """A sparse vector of a schema, paired with the embedder that fills it.

    As `DenseBinding` does for dense ones: a `Sparse` vector can only be bound to a
    sparse embedder, and the retrieve stage uses the same binding to embed questions.

    Attributes:
        vector: The schema vector the mappings are for.
        embedder: The model producing them, for passages and questions alike.
    """

    vector: Sparse
    embedder: SparseEmbedder


@overload
def bind(vector: Dense, *, embedder: DenseEmbedder) -> DenseBinding: ...
@overload
def bind(vector: Sparse, *, embedder: SparseEmbedder) -> SparseBinding: ...
def bind(vector: Vector, *, embedder: DenseEmbedder | SparseEmbedder) -> VectorBinding:
    """Return the binding pairing `vector` with `embedder`, its kind chosen by the vector.

    The vector already says which binding it needs, so the caller needn't choose the
    class. mypy checks the pairing at the call site: a `Dense` vector takes only a
    `DenseEmbedder`, a `Sparse` one only a `SparseEmbedder`, and the binding returned
    has the exact type. The embedder isn't checked at run time: a protocol can't be,
    short of calling the model, and a wrong one fails on its first rows (`Dense.value`,
    `Sparse.value`). Whether the vector belongs to an index is checked where the index
    is known (`SearchableIndex`, `check_fits`).

    Args:
        vector: The schema vector to fill and search.
        embedder: The model for it, for passages and questions alike.

    Returns:
        A `DenseBinding` for a `Dense` vector, a `SparseBinding` for a `Sparse` one.

    Raises:
        TypeError: `vector` isn't a vector of a kind that can be bound (e.g. a field).

    Example:
        dense = bind(KbChunks.dense, embedder=tei)
        terms = bind(KbChunks.terms, embedder=bm25)
        embed_chunks(chunks, bindings=[dense, terms])
        SearchableIndex(index=chunk_index, scans={"dense": VectorScan(binding=dense)})
    """
    # The overloads pair each vector kind with its embedder kind, so the casts only
    # restate what callers were already held to
    match vector:
        case Dense():
            return DenseBinding(vector=vector, embedder=cast(DenseEmbedder, embedder))
        case Sparse():
            return SparseBinding(vector=vector, embedder=cast(SparseEmbedder, embedder))
        case _:
            raise TypeError(
                f"bind takes a Dense or Sparse vector, got {type(vector).__name__}"
            )


def embed_chunks(
    chunks: Sequence[Chunk], *, bindings: Sequence[VectorBinding]
) -> list[EmbeddedChunk]:
    """Return each chunk paired with its vector values, embedded from `embedded_text`.

    Each binding's embedder receives every chunk's text in one batch. No bindings give
    chunks without vectors, which is right for a schema that declares none: whether the
    vectors match the schema is checked when the index writes them (`check_fits`).

    Args:
        chunks: The chunks to embed.
        bindings: Which vectors to fill, and the embedder for each.

    Returns:
        One embedded chunk per chunk, in the same order.

    Raises:
        ValueError: Two bindings are for vectors with one name (checked before anything
            is embedded), an embedder returned a different number of rows than texts,
            or a row doesn't fit its vector (see `Dense.value` and `Sparse.value`).
        TypeError: A number or weight isn't a real number, or a sparse id isn't an
            integer.

    Example:
        embedded = embed_chunks(
            chunks, bindings=[bind(KbChunks.dense, embedder=tei)]
        )
        index.put(embedded)
    """
    # Checked up front: EmbeddedChunk refuses the same, but only after every model call
    seen_names: set[str] = set()
    for binding in bindings:
        vector_name = binding.vector.name
        if vector_name in seen_names:
            raise ValueError(
                f"multiple bindings for vector {vector_name}; bind each vector once"
            )
        seen_names.add(vector_name)

    # Some services refuse an empty batch
    if not chunks:
        return []

    texts = [chunk.embedded_text for chunk in chunks]
    values_by_binding = [_embed_passages(texts, binding) for binding in bindings]

    return [
        EmbeddedChunk(
            chunk=chunk,
            vector_values=values,
        )
        for chunk, *values in zip(chunks, *values_by_binding, strict=True)
    ]


def _embed_passages(texts: Sequence[str], binding: VectorBinding) -> list[VectorValue]:
    """Return the values `binding`'s embedder gives `texts`, checked by its vector.

    Raises:
        ValueError: The embedder returned a different number of rows than texts, or a
            row doesn't fit the vector.
        TypeError: A number or weight isn't a real number, or a sparse id isn't an
            integer.
    """
    match binding:
        # One name per kind: mypy gives a variable one type in the whole function
        case DenseBinding(vector=vector, embedder=embedder):
            numbers = embedder.embed_passages(texts)
            _check_row_count(numbers, texts=texts, vector=vector)
            return [vector.value(row) for row in numbers]
        case SparseBinding(vector=vector, embedder=embedder):
            weights = embedder.embed_passages(texts)
            _check_row_count(weights, texts=texts, vector=vector)
            return [vector.value(row) for row in weights]
        case _:
            assert_never(binding)


def _check_row_count(rows: Sized, *, texts: Sequence[str], vector: Vector) -> None:
    """Check the embedder for `vector` returned one row for each of `texts`.

    Raises:
        ValueError: It returned another number, which would pair rows with the wrong
            chunks; the message names the vector, to find the embedder among several.
    """
    if len(texts) != len(rows):
        raise ValueError(
            f"the embedder for vector {vector.name!r} "
            f"returned {len(rows)} rows for {len(texts)} texts"
        )
