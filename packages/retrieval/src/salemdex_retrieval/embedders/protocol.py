"""The interface every embedder adapter implements, and how chunks are embedded with it.

An embedder turns texts into numbers. The app supplies it as an adapter, a client for a
hosted model or a wrapper around a local one, speaking plain data only, so it never
needs to know the schema. A binding pairs a schema vector with the embedder that fills
it, and the framework checks every row through that vector (`Dense.value`).

Only dense embedders exist yet. Other kinds become further protocols and bindings when
a source needs them, each binding added to `VectorBinding`.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, assert_never

from salemdex_retrieval.chunks import Chunk, EmbeddedChunk
from salemdex_retrieval.vectors import Dense, VectorValue


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


# Every kind of binding. A closed union: each `match` over it ends in `assert_never`,
# so mypy points at every place that hasn't learned a new kind.
type VectorBinding = DenseBinding


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
            or a row doesn't fit its vector (see `Dense.value`).
        TypeError: A row holds something that isn't a real number.

    Example:
        embedded = embed_chunks(
            chunks, bindings=[DenseBinding(vector=KbChunks.dense, embedder=tei)]
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
        TypeError: A row holds something that isn't a real number.
    """
    match binding:
        case DenseBinding(vector=vector, embedder=embedder):
            rows = embedder.embed_passages(texts)
            if len(texts) != len(rows):
                raise ValueError(
                    f"the embedder for vector {vector.name!r} "
                    f"returned {len(rows)} rows for {len(texts)} texts"
                )
            return [vector.value(row) for row in rows]
        case _:
            assert_never(binding)
