"""The interface every index backend implements, and the checks they share.

Each backend (in memory, Qdrant) must behave as `Index` describes; the shared tests in
`tests/index` hold every backend to it.
"""

import uuid
from collections.abc import Iterable
from typing import Protocol, assert_never

from salemdex_retrieval.chunks import Chunk, EmbeddedChunk, Hit
from salemdex_retrieval.errors import SchemaMismatchError
from salemdex_retrieval.filters import And, AnyOf, Equals, Filter, Not, Or
from salemdex_retrieval.schema import Schema
from salemdex_retrieval.vectors import VectorQuery


class Index(Protocol):
    """Chunks of one schema: stored by id, selected by filters, ranked by queries.

    Each chunk is stored with its vector values; reads return it without them.

    Every filter is checked against the schema before any chunk is read or changed,
    and every write is checked in full before any chunk is stored: a write either
    happens whole or not at all.
    """

    @property
    def schema(self) -> type[Schema]:
        """The schema every chunk and every filter field must belong to; never changes."""
        ...

    def put(self, embedded_chunks: Iterable[EmbeddedChunk]) -> None:
        """Store `embedded_chunks`, overwriting any stored chunk with the same id.

        Writing the same chunks again changes nothing, so a retried write is safe.
        Reads return the chunks without their vectors.

        Args:
            embedded_chunks: The chunks to store, each with its vector values.

        Raises:
            SchemaMismatchError: A chunk doesn't fit the schema. Nothing is stored.
            ValueError: Two chunks share an id. Nothing is stored.
        """
        ...

    def find(self, *, where: Filter) -> list[Chunk]:
        """Return the stored chunks that `where` matches, in no promised order.

        This reads without any asker's access filter. Only indexing code, tests and
        the engine may call it; anything serving a user searches through a session.

        Args:
            where: Which chunks to return.

        Raises:
            SchemaMismatchError: `where` uses a field that isn't in the schema.
        """
        ...

    def delete(self, *, where: Filter) -> None:
        """Remove the stored chunks that `where` matches.

        Args:
            where: Which chunks to remove.

        Raises:
            SchemaMismatchError: `where` uses a field that isn't in the schema.
                Nothing is removed.
        """
        ...

    def replace(self, *, where: Filter, with_: Iterable[EmbeddedChunk]) -> None:
        """Remove the chunks that `where` matches and store `with_` in their place.

        Like SQL's DELETE … WHERE then INSERT: the new chunks needn't match `where`.
        A stored chunk with a new chunk's id is overwritten, as in `put`. A backend
        that can't make the two steps atomic (Qdrant) writes the new chunks first: a
        reader may briefly see old and new chunks together, but never neither.

        Example:
            index.replace(where=KbChunks.document_id.eq(doc_id), with_=new_chunks)

        Args:
            where: Which chunks to remove.
            with_: The chunks to store instead, each with its vector values.

        Raises:
            SchemaMismatchError: `where` uses a field that isn't in the schema, or a
                new chunk doesn't fit it. Nothing is changed.
            ValueError: Two new chunks share an id. Nothing is changed.
        """
        ...

    def search(self, query: VectorQuery, *, where: Filter, limit: int) -> list[Hit]:
        """Return the chunks `where` matches nearest to `query`, best first.

        `where` is applied before ranking, never after: a chunk it doesn't match takes
        no place among the hits, so `limit` is filled by matching chunks whenever there
        are enough. Hits with equal scores come in no promised order.

        Like `find`, this applies no asker's access filter of its own. The engine
        passes it in `where`; anything serving a user searches through a session.

        Example:
            index.search(KbChunks.dense.query(numbers), where=access, limit=20)

        Args:
            query: What to search for, and in which of the schema's vectors.
            where: Which chunks may be returned. Required, so that no search can
                forget its filter.
            limit: The most hits to return; a positive `int`.

        Returns:
            At most `limit` hits, each holding a chunk without its vectors and its
            score for `query`.

        Raises:
            SchemaMismatchError: `query` searches a vector that isn't in the schema,
                or `where` uses a field that isn't. Nothing is read.
            ValueError: `limit` isn't a positive `int`. Nothing is read.
        """
        ...


def check_batch(
    embedded_chunks: Iterable[EmbeddedChunk], *, schema: type[Schema]
) -> list[EmbeddedChunk]:
    """Return `embedded_chunks` as a list, after checking each against `schema`.

    Every backend calls it before storing anything, so a write is checked in full
    first and nothing is stored when one chunk is wrong.

    Args:
        embedded_chunks: The chunks about to be written, with their vector values.
        schema: The schema of the index they're written to.

    Returns:
        The same chunks, in the same order.

    Raises:
        SchemaMismatchError: A chunk's fields or vectors don't fit the schema.
        ValueError: Two chunks share an id.
    """
    batch = list(embedded_chunks)
    seen: set[uuid.UUID] = set()
    for embedded in batch:
        chunk_id = embedded.chunk.id
        if chunk_id in seen:
            raise ValueError(f"duplicate chunk {chunk_id} in batch")
        embedded.check_fits(schema)
        seen.add(chunk_id)
    return batch


def check_filter(where: Filter, *, schema: type[Schema]) -> None:
    """Check that every field `where` uses is one of `schema`'s fields.

    Every backend calls it before reading or changing anything, so a wrong filter
    changes nothing. Fields are compared as objects, not by name, as in
    `Chunk.check_fits`. Values aren't checked again: `eq()` and `any_of()` checked them
    against their field when the filter was built.

    Args:
        where: The filter about to be run.
        schema: The schema of the index it runs on.

    Raises:
        SchemaMismatchError: `where` uses a field that isn't in `schema`, or another
            schema's field with the same name. The message names the first one found.
    """
    schema_fields = schema.fields()
    match where:
        case Equals(field=field) | AnyOf(field=field):
            if schema_fields.get(field.name) is field:
                return
            in_another = (
                " (another schema's field)" if field.name in schema_fields else ""
            )
            raise SchemaMismatchError(
                f"field {field.name!r}{in_another} not in schema {schema.__name__}"
            )
        case And(filters=filters) | Or(filters=filters):
            for f in filters:
                check_filter(f, schema=schema)
        case Not(filter=inner):
            check_filter(inner, schema=schema)
        case _:
            assert_never(where)


def check_query(query: VectorQuery, *, schema: type[Schema]) -> None:
    """Check that `query` searches one of `schema`'s vectors.

    Every backend calls it before reading anything. Vectors are compared as objects,
    not by name, as fields are in `check_filter`. The numbers aren't checked again:
    `Dense.query` checked them against their vector when the query was built.

    Args:
        query: The query about to be run.
        schema: The schema of the index it runs on.

    Raises:
        SchemaMismatchError: The query's vector isn't in `schema`, or is another
            schema's vector with the same name.
    """
    vector = query.vector
    schema_vectors = schema.vectors()
    if vector is not schema_vectors.get(vector.name):
        in_another = (
            " (another schema's vector)" if vector.name in schema_vectors else ""
        )
        raise SchemaMismatchError(
            f"vector {vector.name!r}{in_another} not in schema {schema.__name__}"
        )


def check_limit(limit: int) -> None:
    """Check that `limit` is a positive `int`.

    Every backend calls it before reading anything, so none has to decide what a limit
    of 0 means.

    Args:
        limit: The most hits a search may return.

    Raises:
        ValueError: `limit` isn't an `int` (a `bool` doesn't count), or is below 1.
    """
    if type(limit) is not int or limit <= 0:
        raise ValueError(f"limit must be a positive integer, got {limit}")
