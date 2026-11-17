import math
import re
from fractions import Fraction

import pytest

from salemdex_retrieval.checks import (
    check_positive_int,
    to_finite_float,
    to_stripped_str,
)


@pytest.mark.parametrize("value", [1, 50, 2**63])
def test_a_positive_int_passes(value: int) -> None:
    check_positive_int(value, name="depth")


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(True, id="a bool"),
        pytest.param(2.0, id="a float"),
        pytest.param("3", id="a string"),
        pytest.param(None, id="None"),
    ],
)
def test_anything_but_a_positive_int_is_refused_naming_the_value(
    value: object,
) -> None:
    with pytest.raises(
        ValueError, match=f"depth must be a positive integer, got {value!r}"
    ):
        check_positive_int(value, name="depth")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(0.5, 0.5, id="a float"),
        pytest.param(3, 3.0, id="an int"),
        pytest.param(Fraction(1, 4), 0.25, id="another real number"),
        pytest.param(-0.0, -0.0, id="negative zero"),
    ],
)
def test_a_finite_real_number_comes_back_as_a_float(
    value: object, expected: float
) -> None:
    result = to_finite_float(value, name="score")

    assert result == expected
    assert type(result) is float


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(True, id="a bool"),
        pytest.param("0.5", id="a string"),
        pytest.param(None, id="None"),
        pytest.param(1j, id="a complex number"),
    ],
)
def test_a_value_that_isnt_a_real_number_is_refused_naming_it(value: object) -> None:
    with pytest.raises(TypeError, match=f"^score is {value!r}; expected a float$"):
        to_finite_float(value, name="score")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_number_that_isnt_finite_is_refused_naming_it(value: float) -> None:
    with pytest.raises(
        ValueError, match=f"^score is {value!r}; expected a finite number$"
    ):
        to_finite_float(value, name="score")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("dense", "dense", id="plain text"),
        pytest.param("  dense\n", "dense", id="whitespace at either end"),
        pytest.param("tiêu đề chương", "tiêu đề chương", id="inner spaces"),
    ],
)
def test_text_comes_back_without_whitespace_at_either_end(
    value: str, expected: str
) -> None:
    assert to_stripped_str(value, name="name") == expected


@pytest.mark.parametrize("value", ["", "   ", "\n\t"])
def test_a_str_without_text_is_refused_naming_it(value: str) -> None:
    with pytest.raises(
        ValueError, match=f"^name is {re.escape(repr(value))}; expected some text$"
    ):
        to_stripped_str(value, name="name")


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(None, id="None"),
        pytest.param(1, id="an int"),
        pytest.param(b"dense", id="bytes"),
    ],
)
def test_a_value_that_isnt_a_str_is_refused_naming_it(value: object) -> None:
    with pytest.raises(TypeError, match=f"^name is {value!r}; expected a str$"):
        to_stripped_str(value, name="name")
