"""An index kept in memory: the reference behaviour for filters, writes and searches.

Use it in tests and single-process local runs. Other backends (Qdrant) must match how it
selects, writes and ranks chunks.
"""

import heapq
import math
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import Final, assert_never

from salemdex_retrieval.checks import check_positive_int
from salemdex_retrieval.chunks import Chunk, EmbeddedChunk, Hit
from salemdex_retrieval.filters import (
    Filter,
    build_chunk_matcher,
)
from salemdex_retrieval.index.protocol import (
    check_batch_matches_schema,
    check_filter_matches_schema,
    check_query_matches_schema,
)
from salemdex_retrieval.schema import Schema
from salemdex_retrieval.vectors import (
    DenseDistance,
    DenseQuery,
    DenseValue,
    SparseQuery,
    SparseValue,
    VectorQuery,
)


class InMemoryIndex:
    """An `Index` holding its chunks in memory: the reference behaviour.

    Other backends must select and write chunks as it does. It promises no more than
    they do: `replace` isn't made atomic here either.

    Not thread-safe, and not shared between processes: use the Qdrant backend when
    another process writes or reads.

    Attributes:
        schema: The schema every chunk and every filter field must belong to.
    """

    def __init__(self, *, schema: type[Schema]) -> None:
        self.schema: Final = schema
        self._embedded_chunks: dict[uuid.UUID, EmbeddedChunk] = {}

    def put(self, embedded_chunks: Iterable[EmbeddedChunk]) -> None:
        """Store `embedded_chunks`, overwriting any stored chunk with the same id (`Index.put`)."""
        for embedded in check_batch_matches_schema(embedded_chunks, schema=self.schema):
            self._embedded_chunks[embedded.chunk.id] = embedded

    def find(self, *, where: Filter) -> list[Chunk]:
        """Return the stored chunks that `where` matches (`Index.find`)."""
        check_filter_matches_schema(where, schema=self.schema)
        matches = build_chunk_matcher(where)
        return [
            embedded.chunk
            for embedded in self._embedded_chunks.values()
            if matches(embedded.chunk)
        ]

    def delete(self, *, where: Filter) -> None:
        """Remove the stored chunks that `where` matches (`Index.delete`)."""
        check_filter_matches_schema(where, schema=self.schema)
        matches = build_chunk_matcher(where)
        self._remove(matches)

    def replace(self, *, where: Filter, with_: Iterable[EmbeddedChunk]) -> None:
        """Swap the chunks `where` matches for `with_` (`Index.replace`)."""
        check_filter_matches_schema(where, schema=self.schema)
        matches = build_chunk_matcher(where)
        batch = check_batch_matches_schema(with_, schema=self.schema)
        self._remove(matches)
        for embedded in batch:
            self._embedded_chunks[embedded.chunk.id] = embedded

    def search(self, query: VectorQuery, *, where: Filter, limit: int) -> list[Hit]:
        """Return the chunks `where` matches nearest to `query` (`Index.search`).

        Scores every matching chunk, then keeps the best `limit`. Equal scores keep the
        order the chunks were stored in, but other backends don't promise that.

        For a sparse query, a chunk sharing no id with it isn't scored, as in Qdrant.
        With `modifier="idf"`, the query's weights are first multiplied by each id's IDF
        over every stored chunk, not only those `where` matches: Qdrant's default.
        """
        check_positive_int(limit, name="limit")
        check_query_matches_schema(query, schema=self.schema)
        check_filter_matches_schema(where, schema=self.schema)
        matches = build_chunk_matcher(where)

        if isinstance(query, SparseQuery) and query.vector.modifier == "idf":
            query = _weighted_by_idf(query, stored=self._embedded_chunks.values())
        hits = [
            Hit(chunk=embedded.chunk, score=_score(embedded, query=query))
            for embedded in self._embedded_chunks.values()
            if matches(embedded.chunk) and _shares_an_id(embedded, query=query)
        ]
        return heapq.nlargest(limit, hits, key=lambda hit: hit.score)

    def _remove(self, matches: Callable[[Chunk], bool]) -> None:
        self._embedded_chunks = {
            chunk_id: embedded
            for chunk_id, embedded in self._embedded_chunks.items()
            if not matches(embedded.chunk)
        }


def _score(embedded: EmbeddedChunk, *, query: VectorQuery) -> float:
    """Return the chunk's score for `query`, as its vector compares them.

    Raises:
        TypeError: The chunk's value and `query` are of different kinds, which can't
            happen: values are keyed by their own vector.
    """
    value = embedded.vector_values[query.vector]
    match value, query:
        case DenseValue(vector=vector, numbers=stored), DenseQuery(numbers=asked):
            return _dense_score(stored, asked, distance=vector.distance)
        case SparseValue(weights=stored), SparseQuery(weights=asked):
            return _sparse_score(stored, asked)
        case _:
            raise TypeError(
                f"vector {query.vector.name!r} holds a {type(value).__name__}, "
                f"not a value for {type(query).__name__}"
            )


def _dense_score(
    stored: tuple[float, ...], asked: tuple[float, ...], *, distance: DenseDistance
) -> float:
    """Return how near `stored` is to `asked` by `distance`; higher is nearer."""
    match distance:
        case "dot":
            return math.sumprod(stored, asked)
        case "cosine":
            norms = math.hypot(*stored) * math.hypot(*asked)
            # A zero vector scores 0.0, as it does in Qdrant.
            return math.sumprod(stored, asked) / norms if norms else 0.0
        case _:
            assert_never(distance)


def _sparse_score(stored: Mapping[int, float], asked: Mapping[int, float]) -> float:
    """Return the dot product of `stored` and `asked` over the ids they share."""
    return sum(
        weight * stored.get(sparse_id, 0.0) for sparse_id, weight in asked.items()
    )


def _weighted_by_idf(
    query: SparseQuery, *, stored: Iterable[EmbeddedChunk]
) -> SparseQuery:
    """Return `query` with each weight multiplied by its id's IDF over `stored`.

    As Qdrant does it: the query is weighted, the stored values are left as they are,
    and every stored chunk counts, one with no weights too.
    """
    values: list[SparseValue] = []
    for chunk in stored:
        value = chunk.vector_values[query.vector]
        if isinstance(value, SparseValue):
            values.append(value)
    document_frequencies = Counter(
        sparse_id
        for sparse_id in query.weights
        for value in values
        if sparse_id in value.weights
    )
    new_weights = {
        sparse_id: weight
        * _idf(document_frequencies[sparse_id], chunk_count=len(values))
        for sparse_id, weight in query.weights.items()
    }

    return SparseQuery(vector=query.vector, weights=MappingProxyType(new_weights))


def _idf(document_frequency: int, *, chunk_count: int) -> float:
    """Return BM25's IDF for an id held by `document_frequency` of `chunk_count` chunks.

    Qdrant's formula, `ln((N - n + 0.5) / (n + 0.5) + 1)`, with N the chunk count and n
    the document frequency: always above 0, and smaller the more chunks hold the id. A
    document, in IR's terms, is what an index stores: here a chunk.
    """
    return math.log(
        1 + (chunk_count - document_frequency + 0.5) / (document_frequency + 0.5)
    )


def _shares_an_id(embedded: EmbeddedChunk, *, query: VectorQuery) -> bool:
    """Return whether the chunk can be a hit for `query`.

    A sparse value can only when it shares an id with the query, whatever its weight,
    as Qdrant finds sparse hits through the ids; a dense one always can.

    Raises:
        TypeError: The chunk's value and `query` are of different kinds, which can't
            happen: values are keyed by their own vector.
    """
    value = embedded.vector_values[query.vector]
    match value, query:
        case DenseValue(), DenseQuery():
            return True
        case SparseValue(weights=stored), SparseQuery(weights=asked):
            return not asked.keys().isdisjoint(stored)
        case _:
            raise TypeError(
                f"vector {query.vector.name!r} holds a {type(value).__name__}, "
                f"not a value for {type(query).__name__}"
            )
