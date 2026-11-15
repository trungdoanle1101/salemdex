"""An index stored in a Qdrant collection.

Each chunk is one point: the chunk id is the point id, each vector value is one of the
point's named vectors, and the payload is laid out as `payload.py` describes. Filters
run inside Qdrant (`filters.py`), for searches too: a search ranks only the points its
filter matches.

Open an index with `QdrantIndex.open`, which checks the collection first: it creates the
collection and its payload indexes when they're missing, and refuses to open when what
Qdrant holds contradicts the schema, whether a vector, a payload index or the stored
points.
"""

import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Final, Self, assert_never

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from salemdex_retrieval.chunks import Chunk, EmbeddedChunk, Hit
from salemdex_retrieval.errors import SchemaMismatchError
from salemdex_retrieval.filters import Filter
from salemdex_retrieval.index.protocol import (
    check_batch,
    check_filter,
    check_limit,
    check_query,
)
from salemdex_retrieval.index.qdrant.filters import build_qdrant_filter
from salemdex_retrieval.index.qdrant.payload import (
    WRITTEN_FIELDS,
    decode_payload,
    encode_payload,
    field_key,
    payload_schema_type,
    written_fields_of,
)
from salemdex_retrieval.schema import Schema
from salemdex_retrieval.vectors import (
    Dense,
    DenseDistance,
    DenseQuery,
    DenseValue,
    VectorQuery,
    VectorValue,
)

# Room for every expected `written_fields` entry and this many others; more means the
# collection holds something else entirely, and listing them wouldn't help.
_MAX_EXTRA_ENTRIES = 100

_QDRANT_DISTANCES: dict[DenseDistance, models.Distance] = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
}


# What Qdrant stores under a vector's name, and takes as a query for it. Widens with
# each kind of vector: list[list[float]] for multi-dense, models.SparseVector for
# sparse.
type _QdrantVectorData = list[float]


class QdrantIndex:
    """An `Index` storing its chunks as the points of one Qdrant collection.

    Behaves as `InMemoryIndex` does; the shared contract tests hold it to that. Every
    write waits until Qdrant has applied it, so a `find` or `search` right after sees
    it. One difference: once a collection is large, `search` is approximate (see
    `search`).

    Attributes:
        schema: The schema every chunk and every filter field must belong to.
    """

    def __init__(
        self, *, schema: type[Schema], client: QdrantClient, collection: str
    ) -> None:
        """Wrap `collection` as an index, without checking it.

        Use `open`, which checks the collection first; this skips those checks.

        Args:
            schema: The schema every chunk and every filter field must belong to.
            client: An open client. The index doesn't close it; its owner does.
            collection: The collection's name, already checked by `open`.
        """
        self.schema: Final = schema
        self._client = client
        self._collection = collection

    @classmethod
    def open(
        cls, *, schema: type[Schema], client: QdrantClient, collection: str
    ) -> Self:
        """Open the index stored in `collection`, after checking it against `schema`.

        Creates the collection if it's missing, with a named vector for each of the
        schema's vectors, then checks the collection has exactly those: no vector
        missing or extra, and each with the schema's settings. Then creates a payload
        index for each `indexed=True` field and for `written_fields` that has none; an
        index the schema doesn't ask for is left alone. Last, checks every stored point
        was written with exactly the schema's fields, by counting `written_fields`
        entries in Qdrant rather than reading the points. Safe to call again: a
        collection that's already right is unchanged.

        Args:
            schema: The schema every chunk and every filter field must belong to.
            client: An open client. The index doesn't close it; its owner does.
            collection: The collection's name.

        Returns:
            The index, ready for `put`, `find`, `delete`, `replace` and `search`.

        Raises:
            SchemaMismatchError: The collection's vectors differ from the schema's:
                one is missing, extra or set differently, or the collection holds one
                unnamed vector. Qdrant can't change them in place, so re-index into a
                new collection; nothing is created. Or a field's payload index has
                another type than the schema needs; nothing is created, so drop that
                index and open again. Or some stored points lack a field of the
                schema, or hold one it doesn't define; the message counts them per
                field. Re-index into a new collection.
            TypeError: The schema declares a kind of vector this backend can't store.
        """
        _ensure_collection(client=client, collection=collection, schema=schema)
        _check_vectors_fit_schema(client=client, collection=collection, schema=schema)
        _ensure_payload_indexes(client=client, collection=collection, schema=schema)
        _check_points_fit_schema(client=client, collection=collection, schema=schema)
        return cls(schema=schema, client=client, collection=collection)

    def put(self, embedded_chunks: Iterable[EmbeddedChunk]) -> None:
        """Store `embedded_chunks`, overwriting any stored chunk with the same id (`Index.put`)."""
        self._store(check_batch(embedded_chunks, schema=self.schema))

    def find(self, *, where: Filter) -> list[Chunk]:
        """Return the stored chunks that `where` matches (`Index.find`).

        Raises:
            SchemaMismatchError: `where` uses a field that isn't in the schema, or a
                matched point doesn't hold a chunk of the schema.
        """
        check_filter(where, schema=self.schema)
        records = self._scroll_all(build_qdrant_filter(where))
        return [self._to_chunk(record) for record in records]

    def delete(self, *, where: Filter) -> None:
        """Remove the stored chunks that `where` matches (`Index.delete`)."""
        check_filter(where, schema=self.schema)
        self._delete(build_qdrant_filter(where))

    def replace(self, *, where: Filter, with_: Iterable[EmbeddedChunk]) -> None:
        """Swap the chunks `where` matches for `with_` (`Index.replace`).

        Qdrant can't make the two steps atomic, so the new chunks are written first and
        the matched chunks they didn't overwrite are removed after: a reader may see
        old and new chunks together for a moment, but never neither. If the removal
        fails, both stay stored; repeating the call finishes it.
        """
        check_filter(where, schema=self.schema)
        batch = check_batch(with_, schema=self.schema)
        self._store(batch)
        self._delete(_left_over(where, kept=[embedded.chunk.id for embedded in batch]))

    def search(self, query: VectorQuery, *, where: Filter, limit: int) -> list[Hit]:
        """Return the chunks `where` matches nearest to `query` (`Index.search`).

        One request: Qdrant applies `where` while it searches, never after. Scores are
        computed on the float32 numbers Qdrant stores, so they may differ from the
        in-memory index's in the last digits.

        Exact only while the collection is small: Qdrant compares every vector until
        the collection outgrows its indexing threshold, then searches an HNSW graph,
        which is approximate. A restrictive `where` (a reader's few groups) can then
        return fewer than `limit` hits or miss the nearest. A payload index on each
        field `where` uses keeps such searches accurate, since Qdrant can then count
        the matching points and compare them all when they're few; so declare every
        field the engine filters every search by with `indexed=True`.

        Raises:
            SchemaMismatchError: `query` searches a vector that isn't in the schema, or
                `where` uses a field that isn't; nothing is read. Or a matched point
                doesn't hold a chunk of the schema; the message names it.
            ValueError: `limit` isn't a positive `int`; nothing is read.
        """
        check_limit(limit)
        check_query(query, schema=self.schema)
        check_filter(where, schema=self.schema)
        response = self._client.query_points(
            collection_name=self._collection,
            query=_to_qdrant_query(query),
            query_filter=build_qdrant_filter(where),
            limit=limit,
            using=query.vector.name,
            with_payload=True,
            with_vectors=False,
        )
        return [
            Hit(
                chunk=self._to_chunk(point),
                score=point.score,
            )
            for point in response.points
        ]

    def _store(self, batch: list[EmbeddedChunk]) -> None:
        """Write embedded chunks already checked against the schema, in one request."""
        # Qdrant refuses an update with no points.
        if not batch:
            return
        points = [
            models.PointStruct(
                id=embedded.chunk.id,
                payload=encode_payload(embedded.chunk),
                vector={
                    vector.name: _to_qdrant_value(value)
                    for vector, value in embedded.vector_values.items()
                },
            )
            for embedded in batch
        ]
        self._client.upsert(collection_name=self._collection, points=points, wait=True)

    def _scroll_all(self, qdrant_filter: models.Filter) -> list[models.Record]:
        """Return every point `qdrant_filter` matches, reading page by page."""
        records: list[models.Record] = []
        offset: models.ExtendedPointId | None = None
        while True:
            page, offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=qdrant_filter,
                offset=offset,
                with_payload=True,
                with_vectors=False,
                limit=1000,
            )
            records.extend(page)
            if offset is None:
                return records

    def _to_chunk(self, record: models.Record | models.ScoredPoint) -> Chunk:
        """Return the chunk a point holds.

        Raises:
            SchemaMismatchError: The point doesn't hold a chunk of the schema.
        """
        # No payload reads as an empty one, so the error names the missing keys.
        return decode_payload(
            point_id=record.id, payload=record.payload or {}, schema=self.schema
        )

    def _delete(self, qdrant_filter: models.Filter) -> None:
        """Remove the points `qdrant_filter` matches."""
        self._client.delete(
            collection_name=self._collection, points_selector=qdrant_filter, wait=True
        )


@dataclass(frozen=True, kw_only=True)
class _VectorConfigs:
    """A schema's vectors as Qdrant settings, split by the slot Qdrant keeps them in.

    Attributes:
        dense: The settings for `vectors_config`, by vector name.
        sparse: The settings for `sparse_vectors_config`, by vector name.
    """

    dense: Mapping[str, models.VectorParams]
    sparse: Mapping[str, models.SparseVectorParams]


def _ensure_collection(
    *, client: QdrantClient, collection: str, schema: type[Schema]
) -> None:
    """Create `collection` with `schema`'s vectors, unless it already exists.

    An existing collection is left as it is, even when another process created it a
    moment ago; `_check_vectors_fit_schema` checks it next.

    Raises:
        TypeError: The schema declares a kind of vector this backend can't store.
    """
    if client.collection_exists(collection):
        return

    try:
        configs = _qdrant_vector_configs(schema)
        client.create_collection(
            collection_name=collection,
            vectors_config=configs.dense,
            sparse_vectors_config=configs.sparse,
        )
    except UnexpectedResponse:
        if client.collection_exists(collection):
            return
        raise


def _ensure_payload_indexes(
    *, client: QdrantClient, collection: str, schema: type[Schema]
) -> None:
    """Create the payload index each indexed field and `written_fields` lacks.

    Checks the existing ones first, and creates nothing if any has the wrong type.

    Raises:
        SchemaMismatchError: An indexed field's payload index has the wrong type. The
            message names every such field; no index is created.
    """
    expected_types = {
        field_key(field): payload_schema_type(field)
        for field in schema.fields().values()
        if field.indexed
    }
    expected_types[WRITTEN_FIELDS] = models.PayloadSchemaType.KEYWORD
    actual_types = {
        field_name: index_info.data_type
        for field_name, index_info in client.get_collection(
            collection
        ).payload_schema.items()
    }
    to_create: list[tuple[str, models.PayloadSchemaType]] = []
    problems: list[str] = []
    for key, expected in expected_types.items():
        actual = actual_types.get(key)
        if actual is None:
            to_create.append((key, expected))
        elif actual != expected:
            problems.append(f"field {key}: expected {expected}, actual: {actual}")

    if problems:
        header = f"collection {collection}: mismatch between Schema {schema.__name__} and Qdrant payload:"
        raise SchemaMismatchError("\n".join([header, *problems]))

    for key, field_type in to_create:
        client.create_payload_index(
            collection_name=collection,
            field_name=key,
            field_schema=field_type,
            wait=True,
        )


def _check_vectors_fit_schema(
    *, client: QdrantClient, collection: str, schema: type[Schema]
) -> None:
    """Check `collection` holds exactly `schema`'s vectors, each with its settings.

    Raises:
        SchemaMismatchError: A vector is missing, extra or set differently; the
            message lists every one. Or the collection holds one unnamed vector.
    """
    expected = _qdrant_vector_configs(schema)
    params = client.get_collection(collection).config.params
    actual_dense = _named_dense_params(params, collection=collection, schema=schema)
    actual_sparse = params.sparse_vectors or {}
    problems = [
        *_vector_problems(expected.dense, actual_dense, describe=_describe_dense),
        *_vector_problems(expected.sparse, actual_sparse, describe=_describe_sparse),
    ]

    if problems:
        header = (
            f"collection {collection}: vectors don't match schema {schema.__name__}; "
            "re-index into a new collection"
        )
        raise SchemaMismatchError("\n".join([header, *problems]))


def _qdrant_vector_configs(schema: type[Schema]) -> _VectorConfigs:
    """Return the Qdrant settings for each of `schema`'s vectors.

    Raises:
        TypeError: The schema declares a kind of vector this backend can't store.
    """
    schema_vectors = schema.vectors()
    dense: dict[str, models.VectorParams] = {}
    sparse: dict[str, models.SparseVectorParams] = {}
    for name, vector in schema_vectors.items():
        if isinstance(vector, Dense):
            dense[name] = models.VectorParams(
                size=vector.size, distance=_QDRANT_DISTANCES[vector.distance]
            )
        else:
            raise TypeError(f"unknown vector type {type(vector).__name__}")
    return _VectorConfigs(dense=dense, sparse=sparse)


def _named_dense_params(
    params: models.CollectionParams, *, collection: str, schema: type[Schema]
) -> dict[str, models.VectorParams]:
    """Return the collection's dense vector settings, by vector name.

    Raises:
        SchemaMismatchError: The collection holds one unnamed vector, while this
            index writes named ones.
    """
    match params.vectors:
        case None:
            return {}
        case dict():
            return params.vectors
        case models.VectorParams():
            vector_names = ", ".join(name for name in schema.vectors())
            raise SchemaMismatchError(
                f"collection {collection} holds one unnamed vector, "
                f"but schema {schema.__name__} declares named vectors ({vector_names}); "
                "re-index into a new collection"
            )


def _vector_problems[P](
    expected: Mapping[str, P],
    actual: Mapping[str, P],
    *,
    describe: Callable[[P], str],
) -> list[str]:
    """Return how the vectors in one Qdrant slot differ from what the schema needs.

    Vectors are matched by name. Two with the same name differ when `describe` gives
    them different descriptions, so whatever it leaves out isn't compared.
    """
    problems: list[str] = []
    for name, expected_params in expected.items():
        actual_params = actual.get(name)
        if actual_params is None:
            problems.append(f"{name}: the schema declares it; the collection lacks it")
            continue

        expected_settings = describe(expected_params)
        actual_settings = describe(actual_params)
        if expected_settings != actual_settings:
            problems.append(
                f"{name}: schema has {expected_settings}; "
                f"collection has {actual_settings}"
            )
    for name in actual:
        if name not in expected:
            problems.append(
                f"{name}: the collection has it; the schema doesn't declare it"
            )
    return problems


def _describe_dense(params: models.VectorParams) -> str:
    """Return the settings of a dense vector this index sets, as messages show them."""
    multivector = params.multivector_config
    rows = (
        "one row"
        if multivector is None
        else f"many rows ({multivector.comparator.value})"
    )
    return f"size {params.size}, distance {params.distance.value}, {rows}"


def _describe_sparse(params: models.SparseVectorParams) -> str:
    """Return the settings of a sparse vector this index sets, as messages show them."""
    # A collection created without a modifier reads back None, not Modifier.NONE.
    modifier = params.modifier or models.Modifier.NONE
    return f"modifier {modifier.value}"


def _check_points_fit_schema(
    *, client: QdrantClient, collection: str, schema: type[Schema]
) -> None:
    """Check every point in `collection` was written with exactly `schema`'s fields.

    Counts the points, then each `written_fields` entry in one facet request, so no
    point is read. Counting first means a write between the two requests can't make a
    healthy collection look broken; a delete between them still can.

    Raises:
        SchemaMismatchError: Some points lack an entry the schema expects, or hold one
            it doesn't define, or there are too many unknown entries to list. The
            message counts the points for each entry.
    """
    total = client.count(
        collection_name=collection,
        exact=True,
    ).count
    expected = written_fields_of(schema)
    limit = len(expected) + _MAX_EXTRA_ENTRIES
    points_by_written_field = _count_points_by_written_field(
        client=client, collection=collection, limit=limit
    )
    if len(points_by_written_field) == limit:
        raise SchemaMismatchError(
            f"collection {collection} has over {_MAX_EXTRA_ENTRIES} {WRITTEN_FIELDS} entries "
            f"schema {schema.__name__} doesn't define; "
            "it holds another schema's points or wasn't written by this index"
        )

    problems: list[str] = []
    for entry in sorted(expected):
        if (count := total - points_by_written_field.get(entry, 0)) > 0:
            problems.append(f"{entry}: {count} points lack it")

    points_by_written_field = dict(sorted(points_by_written_field.items()))
    for entry, count in points_by_written_field.items():
        if entry not in expected:
            problems.append(
                f"{entry}: {count} points have it, but the schema doesn't define it"
            )

    if problems:
        header = f"collection {collection}: {WRITTEN_FIELDS} don't match against schema {schema.__name__}"

        raise SchemaMismatchError("\n".join([header, *problems]))


def _count_points_by_written_field(
    *, client: QdrantClient, collection: str, limit: int
) -> dict[str, int]:
    """Return how many points hold each `written_fields` entry, for `limit` entries."""
    hits = client.facet(collection, key=WRITTEN_FIELDS, limit=limit, exact=True).hits
    return {str(hit.value): hit.count for hit in hits}


def _left_over(where: Filter, *, kept: list[uuid.UUID]) -> models.Filter:
    """Return a filter matching what `where` matches, except the chunks in `kept`."""
    return models.Filter(
        must=[build_qdrant_filter(where)],
        must_not=[models.HasIdCondition(has_id=kept)],
    )


def _to_qdrant_query(query: VectorQuery) -> _QdrantVectorData:
    """Return `query` in the form `query_points` takes for its kind of vector."""
    match query:
        case DenseQuery(numbers=numbers):
            return list(numbers)
        case _:
            assert_never(query)


def _to_qdrant_value(value: VectorValue) -> _QdrantVectorData:
    """Return `value` in the form a point stores for its kind of vector."""
    match value:
        case DenseValue(numbers=numbers):
            return list(numbers)
        case _:
            assert_never(value)
