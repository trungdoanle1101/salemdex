"""The vectors a chunk is searched by, and the checked values written and searched with.

A schema declares its vectors next to its fields, e.g. `dense = Dense(size=1024,
distance="cosine")` or `sparse = Sparse(modifier="idf")`. Each vector checks numbers
where they enter: for writes (`value`) and for searches (`query`). The caller embeds; a
backend only stores and compares.

Two kinds exist: `Dense`, a fixed number of floats, and `Sparse`, a weight for each of a
few ids out of many. Others (multi-vector) become further subclasses of `Vector` when a
source needs them, each a sibling of the others, never a subclass.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType
from typing import ClassVar, Final, Literal, get_args

from salemdex_retrieval.checks import to_finite_float
from salemdex_retrieval.declarations import Declaration
from salemdex_retrieval.errors import SchemaError

type DenseDistance = Literal["cosine", "dot"]
type SparseModifier = Literal["none", "idf"]

_DENSE_DISTANCES = get_args(DenseDistance.__value__)
_SPARSE_MODIFIERS = get_args(SparseModifier.__value__)
_UINT32_MAX = (1 << 32) - 1

type VectorValue = DenseValue | SparseValue
type VectorQuery = DenseQuery | SparseQuery


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


class Sparse(Vector):
    """A vector of weights for a few ids out of many, such as a passage's terms.

    Each id stands for a dimension of the model's own space: a token of its vocabulary,
    or a word's hash. Turning terms into ids is the embedder's job, so any model fits,
    learned (SPLADE) or counted (BM25). A query and a stored value are compared by the
    dot product over the ids they share, and a stored value sharing none with a query
    isn't a hit.

    Attributes:
        modifier: What a backend does to the weights before comparing. `"none"` takes
            them as they are, as a learned model's are meant to be; `"idf"` multiplies
            each query weight by its id's inverse document frequency over the index, so
            an embedder giving BM25's term-frequency part gets BM25.
    """

    def __init__(self, *, modifier: SparseModifier) -> None:
        """Declare a sparse vector whose weights `modifier` adjusts when searched.

        Args:
            modifier: `"none"` or `"idf"`.

        Raises:
            SchemaError: `modifier` isn't one of the supported ones.
        """
        super().__init__()
        if modifier not in _SPARSE_MODIFIERS:
            available_modifiers = ", ".join(_SPARSE_MODIFIERS)
            raise SchemaError(f"modifier must be one of {available_modifiers}")

        self.modifier: Final = modifier

    def __repr__(self) -> str:
        type_name = type(self).__name__
        arguments = f"modifier={self.modifier!r}, name={self._name!r}"
        return f"{type_name}({arguments})"

    def value(self, weights: Mapping[int, float]) -> SparseValue:
        """Return a chunk's value for this vector, checked.

        Args:
            weights: The weight of each id in the chunk's `embedded_text`, from the
                model's passage side. Ids and weights may be numpy numbers; an id left
                out weighs nothing.

        Returns:
            The value, holding a read-only copy of the weights as plain `int` ids and
            `float` weights.

        Raises:
            TypeError: An id isn't an integer (a `str` or a `bool`, say), or a weight
                isn't a real number.
            ValueError: An id is outside 0 to 2**32 - 1, or a weight isn't finite.
        """
        return SparseValue(vector=self, weights=MappingProxyType(self._check(weights)))

    def query(self, weights: Mapping[int, float]) -> SparseQuery:
        """Return a query to search this vector with, checked.

        Args:
            weights: The weight of each id in the question, from the model's query
                side, with the same ids as the stored values.

        Returns:
            The query, holding a read-only copy of the weights as plain `int` ids and
            `float` weights.

        Raises:
            TypeError: An id isn't an integer (a `str` or a `bool`, say), or a weight
                isn't a real number.
            ValueError: An id is outside 0 to 2**32 - 1, or a weight isn't finite.
        """
        return SparseQuery(vector=self, weights=MappingProxyType(self._check(weights)))

    def _check(self, weights: Mapping[int, float]) -> dict[int, float]:
        """Return a copy of `weights` with plain `int` ids and `float` weights, checked.

        Raises:
            TypeError: An id isn't an integer, or a weight isn't a real number.
            ValueError: An id is outside 0 to 2**32 - 1, or a weight isn't finite.
        """
        checked: dict[int, float] = {}
        for key, value in weights.items():
            sparse_id = _to_sparse_id(key)
            checked[sparse_id] = to_finite_float(value, name=f"Sparse[{sparse_id!r}]")
        return checked


@dataclass(frozen=True, kw_only=True)
class SparseValue:
    """A chunk's checked value for a sparse vector, built by `Sparse.value`.

    Attributes:
        vector: The vector the value is for.
        weights: Each id's weight, read-only; an id left out weighs nothing.
    """

    vector: Sparse
    weights: Mapping[int, float] = field(repr=False)


@dataclass(frozen=True, kw_only=True)
class SparseQuery:
    """A checked query for a sparse vector, built by `Sparse.query`.

    Attributes:
        vector: The vector to search: a backend compares only with its values.
        weights: Each id's weight, read-only; an id left out weighs nothing.
    """

    vector: Sparse
    weights: Mapping[int, float] = field(repr=False)


def _to_sparse_id(key: object) -> int:
    """Return `key` as a plain `int`, after checking it is an id Qdrant can store.

    Qdrant stores ids as unsigned 32-bit integers. Any integer is taken (a numpy one
    too), except a `bool`.

    Raises:
        TypeError: `key` isn't an integer.
        ValueError: `key` is outside 0 to 2**32 - 1.
    """
    if not isinstance(key, Integral) or isinstance(key, bool):
        raise TypeError(f"weight keys must be integers, got {key!r}")
    sparse_id = int(key)
    if not (0 <= sparse_id <= _UINT32_MAX):
        raise ValueError(f"weight keys must satisfy 0 <= key < 2^32, got {sparse_id!r}")
    return sparse_id
