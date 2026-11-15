"""Declaring dense vectors, and the checks on the numbers written and searched with.

How a schema collects its vectors, next to its fields, is in `test_schema.py`.
"""

import dataclasses
import math
import re
from collections.abc import Callable, Iterable
from typing import Any

import pytest

from salemdex_retrieval.errors import SchemaError
from salemdex_retrieval.schema import Schema
from salemdex_retrieval.vectors import Dense, DenseQuery, DenseValue


class Chunks(Schema):
    dense = Dense(size=3, distance="cosine")


# Declaring


@pytest.mark.parametrize("distance", ["cosine", "dot"])
def test_a_dense_vector_keeps_its_size_and_distance(distance: Any) -> None:
    vector = Dense(size=1024, distance=distance)

    assert vector.size == 1024
    assert vector.distance == distance


@pytest.mark.parametrize(
    "size",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(True, id="bool"),
        pytest.param(3.0, id="float"),
        pytest.param("3", id="str"),
    ],
)
def test_a_dense_vector_rejects_a_size_that_is_not_a_positive_int(size: Any) -> None:
    with pytest.raises(SchemaError, match="size must be a positive integer"):
        Dense(size=size, distance="cosine")


def test_a_dense_vector_rejects_an_unknown_distance_naming_the_supported_ones() -> None:
    manhattan: Any = "manhattan"

    with pytest.raises(SchemaError, match="distance must be one of cosine, dot"):
        Dense(size=3, distance=manhattan)


def test_a_vector_takes_its_name_from_the_schema_class_body() -> None:
    assert Chunks.dense.name == "dense"


def test_a_vector_outside_a_schema_class_body_has_no_name_and_says_how_to_declare_it() -> (
    None
):
    with pytest.raises(
        SchemaError,
        match=re.escape("Dense has no name; declare it in a schema class body")
        + r".*my_vector = Dense\(\.\.\.\)",
    ):
        _ = Dense(size=3, distance="cosine").name


def test_sharing_a_vector_between_two_names_is_rejected() -> None:
    with pytest.raises(SchemaError, match="already the vector 'dense'.*'sentence'"):

        class Twice(Schema):
            dense = Dense(size=3, distance="cosine")
            sentence = dense


def test_a_vector_shows_its_size_distance_and_name() -> None:
    assert repr(Chunks.dense) == "Dense(size=3, distance='cosine', name='dense')"
    assert (
        repr(Dense(size=2, distance="dot"))
        == "Dense(size=2, distance='dot', name=None)"
    )


# Values and queries: the same checks for both

type Build = Callable[[Dense, Iterable[Any]], DenseValue | DenseQuery]

BUILDS = [
    pytest.param(Dense.value, id="value"),
    pytest.param(Dense.query, id="query"),
]


def test_a_value_pairs_the_vector_with_its_numbers() -> None:
    value = Chunks.dense.value([0.5, 1.0, 2.0])

    assert value == DenseValue(vector=Chunks.dense, numbers=(0.5, 1.0, 2.0))


def test_a_query_pairs_the_vector_with_its_numbers() -> None:
    query = Chunks.dense.query([0.5, 1.0, 2.0])

    assert query == DenseQuery(vector=Chunks.dense, numbers=(0.5, 1.0, 2.0))


@pytest.mark.parametrize("build", BUILDS)
def test_ints_are_kept_as_floats(build: Build) -> None:
    numbers = build(Chunks.dense, [1, 2, 3]).numbers

    assert numbers == (1.0, 2.0, 3.0)
    assert all(type(number) is float for number in numbers)


@pytest.mark.parametrize("build", BUILDS)
def test_any_iterable_of_numbers_is_accepted_such_as_a_generator(build: Build) -> None:
    numbers = build(Chunks.dense, (n / 2 for n in range(3))).numbers

    assert numbers == (0.0, 0.5, 1.0)


@pytest.mark.parametrize("build", BUILDS)
def test_a_numpy_row_is_accepted_as_plain_floats(build: Build) -> None:
    numpy = pytest.importorskip("numpy")
    row = numpy.array([0.5, 1.0, 2.0], dtype=numpy.float32)

    numbers = build(Chunks.dense, row).numbers

    assert numbers == (0.5, 1.0, 2.0)
    assert all(type(number) is float for number in numbers)


@pytest.mark.parametrize("build", BUILDS)
@pytest.mark.parametrize(
    "numbers",
    [
        pytest.param([1.0, 2.0], id="too few"),
        pytest.param([1.0, 2.0, 3.0, 4.0], id="too many"),
        pytest.param([], id="none"),
    ],
)
def test_the_wrong_count_of_numbers_is_rejected(
    build: Build, numbers: list[float]
) -> None:
    with pytest.raises(
        ValueError, match=rf"size {len(numbers)}\) does not match vector size 3"
    ):
        build(Chunks.dense, numbers)


@pytest.mark.parametrize("build", BUILDS)
@pytest.mark.parametrize(
    ("numbers", "label"),
    [
        pytest.param("123", "Dense[0] is '1'", id="a string, read as characters"),
        pytest.param([1.0, True, 3.0], "Dense[1] is True", id="a bool"),
        pytest.param([1.0, 2.0, None], "Dense[2] is None", id="None"),
    ],
)
def test_a_number_that_is_not_real_is_rejected_naming_its_position(
    build: Build, numbers: Iterable[Any], label: str
) -> None:
    with pytest.raises(TypeError, match=re.escape(label)):
        build(Chunks.dense, numbers)


@pytest.mark.parametrize("build", BUILDS)
@pytest.mark.parametrize(
    "wrong",
    [math.nan, math.inf, -math.inf],
    ids=["nan", "inf", "-inf"],
)
def test_a_number_that_is_not_finite_is_rejected_naming_its_position(
    build: Build, wrong: float
) -> None:
    with pytest.raises(
        ValueError, match=re.escape(f"Dense[1] is {wrong!r}; expected a finite number")
    ):
        build(Chunks.dense, [1.0, wrong, 3.0])


@pytest.mark.parametrize("build", BUILDS)
def test_a_value_or_query_cannot_be_changed_after_it_is_built(build: Build) -> None:
    built = build(Chunks.dense, [1.0, 2.0, 3.0])

    with pytest.raises(dataclasses.FrozenInstanceError):
        built.numbers = (0.0, 0.0, 0.0)  # type: ignore[misc]
