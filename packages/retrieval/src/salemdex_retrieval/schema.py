"""The typed fields a chunk holds, the checked values written for them, and `Schema`.

A schema subclasses `Schema` and declares `One` and `Many` fields, and the vectors of
`vectors.py`. Each field checks values where they enter: for writes (`value`) and for
filters (`eq`, `any_of`).
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, ClassVar, Final

from salemdex_retrieval.declarations import Declaration
from salemdex_retrieval.errors import SchemaError, SchemaMismatchError
from salemdex_retrieval.filters import AnyOf, Equals, In
from salemdex_retrieval.vectors import Vector

_VALUE_TYPES = (str, int, float, bool, uuid.UUID, datetime)

# Backends store integers in 64 bits (Qdrant: int64); Python's int has no limit.
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


class Field[T](Declaration):
    """A typed, named field of a schema.

    `One` and `Many` add what differs by kind: how a chunk's value is given, and which
    filters apply. A field takes its name from the schema class body it's declared in.

    Every value given to a field, for a write or a filter, is checked: its type must be
    exactly `value_type` (an `int` is widened for a `float` field), a `datetime` must
    have a timezone, a `float` must be finite and an `int` must fit in 64 bits.

    Attributes:
        value_type: The type every value must have exactly.
        indexed: Whether backends build an index on the field for filtering.
    """

    _kind: ClassVar[str] = "field"

    def __init__(self, value_type: type[T], *, indexed: bool = False) -> None:
        """Declare a field whose values are of `value_type`.

        Args:
            value_type: One of `str`, `int`, `float`, `bool`, `UUID`, `datetime`.
                Subclasses, such as `bool` for `int` or a `StrEnum` for `str`, are
                refused.
            indexed: Whether backends build an index on the field for filtering.

        Raises:
            TypeError: `value_type` isn't one of the supported types.
        """
        super().__init__()
        if value_type not in _VALUE_TYPES:
            supported_types = ", ".join(t.__name__ for t in _VALUE_TYPES)
            raise TypeError(
                f"type {value_type!r} is not supported; expected {supported_types}"
            )
        self.value_type: Final = value_type
        self.indexed: Final = indexed

    def __repr__(self) -> str:
        type_name = type(self).__name__
        arguments = (
            f"{self.value_type.__name__}, name={self._name!r}, indexed={self.indexed}"
        )
        return f"{type_name}({arguments})"

    def _check(self, value: object, *, position: int | None = None) -> T:
        """Return `value` checked against this field, an `int` widened for a `float`.

        Args:
            value: The value to check.
            position: The value's position within the given values, named in errors.

        Returns:
            The value, as a `float` when an `int` was given for a `float` field.

        Raises:
            TypeError: The value isn't exactly of the field's type, or is a
                `datetime` without a timezone.
            ValueError: A `float` isn't finite, or an `int` is outside 64 bits.
        """
        if type(value) is int and self.value_type is float:
            try:
                value = float(value)
            except OverflowError:
                raise ValueError(
                    f"{self._label(position)} is an int too large for a float; "
                    f"expected a finite number"
                ) from None
        if type(value) is not self.value_type:
            raise TypeError(
                f"{self._label(position)} is {type(value).__name__!r}, "
                f"expected {self.value_type.__name__}"
            )
        if isinstance(value, int) and not _INT64_MIN <= value <= _INT64_MAX:
            raise ValueError(
                f"{self._label(position)} is outside 64 bits; "
                f"expected {_INT64_MIN} to {_INT64_MAX}"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                f"{self._label(position)} is {value!r}; expected a finite number"
            )

        if isinstance(value, datetime) and value.utcoffset() is None:
            raise TypeError(f"{self._label(position)} is missing a timezone")
        return value

    def _label(self, position: int | None) -> str:
        """Return the field's name for messages, with `[position]` within the given values."""
        return self.name + (f"[{position}]" if position is not None else "")

    def _check_iterable(self, values: Iterable[T], *, owner: str) -> None:
        """Refuse an iterable that would be taken apart into characters, ints or keys.

        Raises:
            TypeError: `values` is a `str`, `bytes` or mapping.
        """
        if isinstance(values, (str, bytes, Mapping)):
            message = (
                f"{owner} needs a collection of values, not {type(values).__name__!r}"
            )
            raise TypeError(message)


class One[T](Field[T]):
    """A field holding exactly one value per chunk."""

    def value(self, value: T) -> OneValue[T]:
        """Return a chunk's value for this field.

        Args:
            value: The chunk's value.

        Returns:
            The checked value paired with this field, for `Chunk(field_values=...)`.

        Raises:
            TypeError: The value is of the wrong type (see `Field`).
            ValueError: The value can't be stored (see `Field`).
        """
        return OneValue(field=self, value=self._check(value))

    def eq(self, value: T) -> Equals[T]:
        """Return a filter matching chunks whose value for this field equals `value`.

        Args:
            value: The value to match.

        Returns:
            An `Equals` filter.

        Raises:
            TypeError: The value is of the wrong type (see `Field`).
            ValueError: The value can't be stored (see `Field`).
        """
        return Equals(field=self, value=self._check(value))

    def in_(self, values: Iterable[T]) -> In[T]:
        """Return a filter matching chunks whose value for this field is in `values`.

        Args:
            values: The values to match, in any iterable but a `str`, `bytes` or
                mapping. Order and repeats don't matter; no values matches no chunk.

        Returns:
            An `In` filter.

        Raises:
            TypeError: `values` is a `str`, `bytes` or mapping, or a value is of the
                wrong type (see `Field`).
            ValueError: A value can't be stored (see `Field`).
        """
        self._check_iterable(values, owner="in_(...)")
        return In(
            field=self,
            values=frozenset(
                self._check(value, position=i) for i, value in enumerate(values)
            ),
        )


class Many[T](Field[T]):
    """A field holding any number of values per chunk, possibly none."""

    def value(self, values: Iterable[T]) -> ManyValues[T]:
        """Return a chunk's values for this field.

        Order and repeats are kept. No values is allowed, and is how a chunk says it
        has none.

        Args:
            values: The chunk's values, in any iterable but a `str`, `bytes` or
                mapping.

        Returns:
            The checked values, as a tuple paired with this field, for
            `Chunk(field_values=...)`.

        Raises:
            TypeError: `values` is a `str`, `bytes` or mapping, or a value is of the
                wrong type (see `Field`).
            ValueError: A value can't be stored (see `Field`).
        """
        self._check_iterable(values, owner="value(...)")

        return ManyValues(
            field=self,
            values=tuple(
                self._check(value, position=i) for i, value in enumerate(values)
            ),
        )

    def any_of(self, values: Iterable[T]) -> AnyOf[T]:
        """Return a filter matching chunks with at least one value in `values`.

        Args:
            values: The values to match, in any iterable but a `str`, `bytes` or
                mapping. Order and repeats don't matter; no values matches no chunk.

        Returns:
            An `AnyOf` filter.

        Raises:
            TypeError: `values` is a `str`, `bytes` or mapping, or a value is of the
                wrong type (see `Field`).
            ValueError: A value can't be stored (see `Field`).
        """
        self._check_iterable(values, owner="any_of(...)")

        return AnyOf(
            field=self,
            values=frozenset(
                self._check(value, position=i) for i, value in enumerate(values)
            ),
        )


@dataclass(frozen=True, kw_only=True)
class OneValue[T]:
    """A chunk's value for a `One` field, built by `One.value`.

    Attributes:
        field: The field the value is for.
        value: The value, checked against the field.
    """

    field: One[T]
    value: T


@dataclass(frozen=True, kw_only=True)
class ManyValues[T]:
    """A chunk's values for a `Many` field, built by `Many.value`.

    Attributes:
        field: The field the values are for.
        values: The values, each checked against the field, in order; possibly none.
    """

    field: Many[T]
    values: tuple[T, ...]


# A chunk holds values for fields of different types, so T is Any here; a backend
# matches on the two classes to tell one value from several.
type FieldValue = OneValue[Any] | ManyValues[Any]


class Schema:
    """Base class for a schema: the fields and vectors of every chunk in an index.

    Declare `One` and `Many` fields and vectors such as `Dense` in the class body.
    Fields and vectors share one set of names. They come only from the class body and
    from `Schema` bases. A declaration in a plain mixin isn't part of the schema, so
    shared ones belong in a `Schema` preset.

    Example:
        class KbChunks(Schema):
            document_id = One(UUID, indexed=True)
            headings = Many(str)
            groups = Many(UUID, indexed=True)
            dense = Dense(size=1024, distance="cosine")
    """

    _declarations: Mapping[str, Declaration] = MappingProxyType({})
    _fields: Mapping[str, Field[Any]] = MappingProxyType({})
    _vectors: Mapping[str, Vector] = MappingProxyType({})

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Collect the subclass's fields and vectors, inherited ones first.

        Raises:
            SchemaError: Two bases give different declarations one name (whether
                fields, vectors or one of each), the class body hides an inherited
                declaration, a declaration takes a name `Schema` uses, or one is
                neither a field nor a vector.
        """
        super().__init_subclass__(**kwargs)
        declarations: dict[str, Declaration] = {}
        fields: dict[str, Field[Any]] = {}
        vectors: dict[str, Vector] = {}
        for base in cls.__bases__:
            if not issubclass(base, Schema):
                continue
            for name, declaration in base._declarations.items():
                if name in declarations and declarations[name] is not declaration:
                    raise SchemaError(
                        f"{cls.__name__} inherits two different declarations named {name!r}; "
                        "rename one in its preset"
                    )
                declarations[name] = declaration

        for name, value in vars(cls).items():
            if name in declarations:
                raise SchemaError(
                    f"{cls.__name__}.{name} hides an inherited declaration; "
                    "inherited declarations can't be redefined or hidden"
                )

            if isinstance(value, Declaration):
                if name in dir(Schema):
                    raise SchemaError(f"{name!r} is a reserved name; use another name")
                declarations[name] = value

        for name, declaration in declarations.items():
            if isinstance(declaration, Field):
                fields[name] = declaration
            elif isinstance(declaration, Vector):
                vectors[name] = declaration
            else:
                raise SchemaError(
                    f"{name!r} uses an unknown declaration of type {type(declaration).__name__}"
                )

        cls._declarations = MappingProxyType(declarations)
        cls._fields = MappingProxyType(fields)
        cls._vectors = MappingProxyType(vectors)

    @classmethod
    def fields(cls) -> Mapping[str, Field[Any]]:
        """Return the schema's fields by name.

        Returns:
            A read-only mapping, inherited fields first, each in declaration order.
        """
        return cls._fields

    @classmethod
    def vectors(cls) -> Mapping[str, Vector]:
        """Return the schema's vectors by name.

        Returns:
            A read-only mapping, inherited vectors first, each in declaration order.
        """
        return cls._vectors

    @classmethod
    def check_declares(cls, declaration: Declaration) -> None:
        """Check that `declaration`, a field or a vector, is one of this schema's.

        Declarations are compared as objects, not by name: another schema's field with
        the same name is a different field, and filtering or searching by it here would
        match nothing.

        Args:
            declaration: The field or vector about to be used with this schema.

        Raises:
            SchemaMismatchError: `declaration` isn't in this schema, or is another
                schema's with the same name.
        """
        name = declaration.name
        if cls._declarations.get(name) is declaration:
            return
        kind = declaration._kind
        in_another = f" (another schema's {kind})" if name in cls._declarations else ""
        raise SchemaMismatchError(
            f"{kind} {name!r}{in_another} not in schema {cls.__name__}"
        )
