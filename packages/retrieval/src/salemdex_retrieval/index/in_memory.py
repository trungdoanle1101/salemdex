"""An index kept in memory: the reference behaviour for filters, writes and searches.

Use it in tests and single-process local runs. Other backends (Qdrant) must match how it
selects, writes and ranks chunks.
"""

import heapq
import math
import uuid
from collections.abc import Callable, Iterable
from typing import Final, assert_never

from salemdex_retrieval.chunks import Chunk, EmbeddedChunk, Hit
from salemdex_retrieval.filters import And, AnyOf, Equals, Everything, Filter, Not, Or
from salemdex_retrieval.index.protocol import (
    check_batch,
    check_filter,
    check_limit,
    check_query,
)
from salemdex_retrieval.schema import Schema
from salemdex_retrieval.vectors import DenseDistance, DenseValue, VectorQuery


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
        for embedded in check_batch(embedded_chunks, schema=self.schema):
            self._embedded_chunks[embedded.chunk.id] = embedded

    def find(self, *, where: Filter) -> list[Chunk]:
        """Return the stored chunks that `where` matches (`Index.find`)."""
        check_filter(where, schema=self.schema)
        matches = self._build_matcher(where)
        return [
            embedded.chunk
            for embedded in self._embedded_chunks.values()
            if matches(embedded.chunk)
        ]

    def delete(self, *, where: Filter) -> None:
        """Remove the stored chunks that `where` matches (`Index.delete`)."""
        check_filter(where, schema=self.schema)
        matches = self._build_matcher(where)
        self._remove(matches)

    def replace(self, *, where: Filter, with_: Iterable[EmbeddedChunk]) -> None:
        """Swap the chunks `where` matches for `with_` (`Index.replace`)."""
        check_filter(where, schema=self.schema)
        matches = self._build_matcher(where)
        batch = check_batch(with_, schema=self.schema)
        self._remove(matches)
        for embedded in batch:
            self._embedded_chunks[embedded.chunk.id] = embedded

    def search(self, query: VectorQuery, *, where: Filter, limit: int) -> list[Hit]:
        """Return the chunks `where` matches nearest to `query` (`Index.search`).

        Scores every matching chunk, then keeps the best `limit`. Equal scores keep the
        order the chunks were stored in, but other backends don't promise that.
        """
        check_limit(limit)
        check_query(query, schema=self.schema)
        check_filter(where, schema=self.schema)
        matches = self._build_matcher(where)
        hits = [
            Hit(chunk=embedded.chunk, score=_score(embedded, query=query))
            for embedded in self._embedded_chunks.values()
            if matches(embedded.chunk)
        ]
        return heapq.nlargest(limit, hits, key=lambda hit: hit.score)

    def _build_matcher(self, where: Filter) -> Callable[[Chunk], bool]:
        """Return a function telling whether a chunk matches `where`.

        `where` must already be checked by `check_filter`: a field outside the schema
        would raise `KeyError` on the first chunk.
        """
        match where:
            case Equals(field=field, value=value):
                return lambda chunk: chunk.field_values[field] == value
            case AnyOf(field=field, values=values):
                return lambda chunk: not values.isdisjoint(chunk.field_values[field])
            case And(filters=filters):
                parts = [self._build_matcher(f) for f in filters]
                return lambda chunk: all(part(chunk) for part in parts)
            case Or(filters=filters):
                parts = [self._build_matcher(f) for f in filters]
                return lambda chunk: any(part(chunk) for part in parts)
            case Not(filter=f):
                part = self._build_matcher(f)
                return lambda chunk: not part(chunk)
            case Everything():
                return lambda chunk: True
            case _:
                assert_never(where)

    def _remove(self, matches: Callable[[Chunk], bool]) -> None:
        self._embedded_chunks = {
            chunk_id: embedded
            for chunk_id, embedded in self._embedded_chunks.items()
            if not matches(embedded.chunk)
        }


def _score(embedded: EmbeddedChunk, *, query: VectorQuery) -> float:
    """Return the chunk's score for `query`, by the distance of the query's vector."""
    vector_value = embedded.vector_values[query.vector]
    match vector_value:
        case DenseValue(vector=v, numbers=stored):
            return _dense_score(stored, query.numbers, distance=v.distance)
        case _:
            assert_never(vector_value)


def _dense_score(
    stored: tuple[float, ...], query: tuple[float, ...], *, distance: DenseDistance
) -> float:
    """Return how near `stored` is to `query` by `distance`; higher is nearer."""
    match distance:
        case "dot":
            return math.sumprod(stored, query)
        case "cosine":
            norms = math.hypot(*stored) * math.hypot(*query)
            # A zero vector scores 0.0, as it does in Qdrant.
            return math.sumprod(stored, query) / norms if norms else 0.0
        case _:
            assert_never(distance)
