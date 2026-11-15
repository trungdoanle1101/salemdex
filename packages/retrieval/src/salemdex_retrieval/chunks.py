"""The pieces of a source that an index stores and search returns."""

import uuid
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, assert_never

from salemdex_retrieval.declarations import Declaration
from salemdex_retrieval.errors import SchemaMismatchError
from salemdex_retrieval.schema import Field, FieldValue, ManyValues, OneValue, Schema
from salemdex_retrieval.vectors import Vector, VectorValue


@dataclass(frozen=True, kw_only=True, init=False)
class Chunk:
    """A piece of a source, as an index stores it and search returns it.

    Attributes:
        id: Unique across the whole index. Derive it from what identifies the piece
            (e.g. document, version, position), so writing it again overwrites it.
        text: The source's own words, returned unchanged by search. Left out of
            `repr`, so it can't reach logs.
        embedded_text: What the model embeds: the text with context added, or the
            text itself. Left out of `repr`.
        field_values: The chunk's value for each field, keyed by the field: one value
            for a `One`, a tuple for a `Many`. Read-only.
    """

    id: uuid.UUID
    text: str = field(repr=False)
    embedded_text: str = field(repr=False)
    field_values: Mapping[Field[Any], Any]

    def __init__(
        self,
        *,
        id: uuid.UUID,
        text: str,
        embedded_text: str | None = None,
        field_values: Iterable[FieldValue],
    ) -> None:
        """Build a chunk, keeping its own read-only copy of the field values.

        Args:
            id: Unique across the whole index.
            text: The source's own words.
            embedded_text: What the model embeds, such as the headings path followed
                by the text. Left out, the text itself is embedded.
            field_values: One value per field, built by `One.value` and `Many.value`.

        Raises:
            TypeError: `id` isn't a `uuid.UUID`.
            ValueError: Two field values have one name: the same field twice, or
                fields from different schemas.
        """
        if type(id) is not uuid.UUID:
            raise TypeError("id must be of type uuid.UUID")

        by_field: dict[Field[Any], Any] = {}
        seen_names: set[str] = set()
        for field_value in field_values:
            name = field_value.field.name
            if name in seen_names:
                raise ValueError(f"duplicate field: {name}")
            seen_names.add(name)

            match field_value:
                case OneValue(field=key, value=value):
                    by_field[key] = value
                case ManyValues(field=key, values=values):
                    by_field[key] = values
                case _:
                    assert_never(field_value)

        object.__setattr__(self, "id", id)
        object.__setattr__(self, "text", text)
        object.__setattr__(
            self, "embedded_text", embedded_text if embedded_text is not None else text
        )
        object.__setattr__(self, "field_values", MappingProxyType(by_field))

    def check_fits(self, schema: type[Schema]) -> None:
        """Check that the chunk has a value for each of `schema`'s fields, and no other.

        Fields are compared as objects, not by name: a field from another schema with
        the same name doesn't fit. An empty `Many` counts as a value.

        Args:
            schema: The schema of the index the chunk is written to or read from.

        Raises:
            SchemaMismatchError: The chunk holds fields `schema` doesn't have, or lacks
                some it has. One error names the chunk id and every such field.
        """
        problems = _field_problems(self, schema)
        if problems:
            raise SchemaMismatchError(f"chunk {self.id}: {'; '.join(problems)}")


@dataclass(frozen=True, kw_only=True, init=False)
class EmbeddedChunk:
    """A chunk with its vector values, as an index writes it.

    Only writes take it: reads return the `Chunk` alone, since nothing that reads a
    chunk needs its vectors back.

    Attributes:
        chunk: The chunk itself.
        vector_values: The chunk's value for each vector, keyed by the vector.
            Read-only.
    """

    chunk: Chunk
    vector_values: Mapping[Vector, VectorValue]

    def __init__(self, *, chunk: Chunk, vector_values: Iterable[VectorValue]) -> None:
        """Pair `chunk` with its vector values, keeping a read-only copy of them.

        Args:
            chunk: The chunk to write.
            vector_values: One value per vector, built by e.g. `Dense.value` from
                the embedding of `chunk.embedded_text`.

        Raises:
            ValueError: Two vector values have one name: the same vector twice, or
                vectors from different schemas.
        """
        by_vector: dict[Vector, VectorValue] = {}
        seen_names: set[str] = set()
        for vector_value in vector_values:
            name = vector_value.vector.name
            if name in seen_names:
                raise ValueError(f"duplicate vector: {name}")
            seen_names.add(name)

            by_vector[vector_value.vector] = vector_value

        object.__setattr__(self, "chunk", chunk)
        object.__setattr__(self, "vector_values", MappingProxyType(by_vector))

    def check_fits(self, schema: type[Schema]) -> None:
        """Check the chunk has a value for each of `schema`'s fields and vectors only.

        Both are compared as objects, as in `Chunk.check_fits`. A value's size needs
        no check here: it was checked against its own vector when built.

        Args:
            schema: The schema of the index the chunk is written to.

        Raises:
            SchemaMismatchError: The chunk holds fields or vectors `schema` doesn't
                have, or lacks some it has. One error names the chunk id and every
                such field and vector.
        """
        field_problems = _field_problems(self.chunk, schema)

        vector_problems = _problems(
            list(self.vector_values), schema.vectors(), schema=schema, kind="vector"
        )
        problems = [*field_problems, *vector_problems]
        if problems:
            raise SchemaMismatchError(
                f"embedded chunk {self.chunk.id}: {'; '.join(problems)}"
            )


@dataclass(frozen=True, kw_only=True)
class Hit:
    """A chunk a search found, with its score for the query.

    Holds the `Chunk` alone, as `find` returns it: nothing that reads a hit needs its
    vectors.

    Attributes:
        chunk: The chunk found.
        score: How near the chunk is to the query, by the vector's distance: higher
            is nearer, for both cosine and dot. Comparable only with the other hits of
            the same search, never across vectors or questions.
    """

    chunk: Chunk
    score: float


def _field_problems(chunk: Chunk, schema: type[Schema]) -> list[str]:
    """Return what's wrong with the chunk's fields for `schema`; none when they fit."""
    return _problems(
        list(chunk.field_values),
        schema.fields(),
        schema=schema,
        kind="field",
    )


def _problems(
    given: Collection[Declaration],
    declared: Mapping[str, Declaration],
    *,
    schema: type[Schema],
    kind: Literal["field", "vector"],
) -> list[str]:
    """Return how `given` differs from `declared`: the extra ones, then the missing.

    Declarations are compared as objects. Each problem is labelled by `kind`
    ("fields missing: …"); there are none when the two match.
    """
    extra = [d for d in given if declared.get(d.name) is not d]
    missing = [d for d in declared.values() if d not in given]

    if not extra and not missing:
        return []

    problems: list[str] = []
    if extra:
        problems.append(
            f"{kind}s not in {schema.__name__}: {_describe_extra(extra, declared, kind)}"
        )
    if missing:
        problems.append(f"{kind}s missing: {', '.join(d.name for d in missing)}")
    return problems


def _describe_extra(
    extra: list[Declaration],
    declared: Mapping[str, Declaration],
    kind: Literal["field", "vector"],
) -> str:
    """Return the extra declarations' names, marking any named like a declared one."""
    return ", ".join(
        f"{d.name} (another schema's {kind})" if d.name in declared else d.name
        for d in extra
    )
