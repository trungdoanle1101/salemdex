"""The vectors a chunk is searched by, and the checked values written and searched with.

A schema declares its vectors next to its fields, e.g. `dense = Dense(size=1024,
distance="cosine")`. Each vector checks numbers where they enter: for writes (`value`)
and for searches (`query`). The caller embeds; a backend only stores and compares.

Only `Dense` exists yet. Other kinds (sparse, multi-vector) become further subclasses
of `Vector` when a source needs them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import ClassVar, Final, Literal, get_args

from salemdex_retrieval.checks import to_finite_float
from salemdex_retrieval.declarations import Declaration
from salemdex_retrieval.errors import SchemaError

type DenseDistance = Literal["cosine", "dot"]

_DENSE_DISTANCES = get_args(DenseDistance.__value__)

type VectorValue = DenseValue
type VectorQuery = DenseQuery


class Vector(Declaration):
    """A named vector of a schema: one way of searching its chunks.

    Subclasses add what differs by kind: the shape of the numbers and how they're
    compared.
    """

    _kind: ClassVar[str] = "vector"


class Dense(Vector):
    """A vector of a fixed number of floats, such as one sentence embedding.

    Attributes:
        size: How many numbers every value and query holds.
        distance: How a backend compares a query with stored values. For `"cosine"`
            the numbers needn't be normalised; the backend does it.
    """

    def __init__(self, *, size: int, distance: DenseDistance) -> None:
        """Declare a dense vector of `size` numbers, compared by `distance`.

        Args:
            size: How many numbers the embedder produces, e.g. 1024 for bge-m3.
            distance: `"cosine"` or `"dot"`.

        Raises:
            SchemaError: `size` isn't a positive `int`, or `distance` isn't one of the
                supported ones.
        """
        super().__init__()
        if not (type(size) is int and size > 0):
            raise SchemaError(f"size must be a positive integer, got {size}")
        if distance not in _DENSE_DISTANCES:
            available_distances = ", ".join(_DENSE_DISTANCES)
            raise SchemaError(f"distance must be one of {available_distances}")
        self.size: Final = size
        self.distance: Final = distance

    def __repr__(self) -> str:
        type_name = type(self).__name__
        arguments = f"size={self.size}, distance={self.distance!r}, name={self._name!r}"

        return f"{type_name}({arguments})"

    def value(self, numbers: Iterable[float]) -> DenseValue:
        """Return a chunk's value for this vector, checked.

        Args:
            numbers: The embedding of the chunk's `embedded_text`. Any iterable of
                real numbers, so a numpy row can be passed as it is.

        Returns:
            The value, holding the numbers as a tuple of `float`.

        Raises:
            TypeError: A number isn't a real number (a `str` or a `bool`, say).
            ValueError: There aren't exactly `size` numbers, or one isn't finite.
        """
        return DenseValue(vector=self, numbers=self._check(numbers))

    def query(self, numbers: Iterable[float]) -> DenseQuery:
        """Return a query to search this vector with, checked.

        Args:
            numbers: The embedding of the question, from the same embedder as the
                stored values.

        Returns:
            The query, holding the numbers as a tuple of `float`.

        Raises:
            TypeError: A number isn't a real number (a `str` or a `bool`, say).
            ValueError: There aren't exactly `size` numbers, or one isn't finite.
        """
        return DenseQuery(vector=self, numbers=self._check(numbers))

    def _check(self, values: Iterable[float]) -> tuple[float, ...]:
        """Return `values` as a tuple of `float`, checked against the vector.

        Raises:
            TypeError: A number isn't a real number.
            ValueError: There aren't exactly `size` numbers, or one isn't finite.
        """
        values = list(values)
        if len(values) != self.size:
            raise ValueError(
                f"values (size {len(values)}) does not match vector size {self.size}"
            )

        return tuple(
            to_finite_float(value, name=f"Dense[{position}]")
            for position, value in enumerate(values)
        )


@dataclass(frozen=True, kw_only=True)
class DenseValue:
    """A chunk's checked value for a dense vector, built by `Dense.value`.

    Attributes:
        vector: The vector the value is for.
        numbers: Exactly `vector.size` finite floats.
    """

    vector: Dense
    numbers: tuple[float, ...] = field(repr=False)


@dataclass(frozen=True, kw_only=True)
class DenseQuery:
    """A checked query for a dense vector, built by `Dense.query`.

    Attributes:
        vector: The vector to search: a backend compares only with its values.
        numbers: Exactly `vector.size` finite floats.
    """

    vector: Dense
    numbers: tuple[float, ...] = field(repr=False)
