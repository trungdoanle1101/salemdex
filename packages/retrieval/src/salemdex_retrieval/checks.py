"""Checks on plain values, shared by every layer: counts and sizes, numbers, names.

Each check knows nothing about schemas, indexes or stages; the caller passes the name
the value goes by, so the message names what the caller wrote. Checks that a field or
vector belongs to a schema live on `Schema`; the checks of the `Index` contract live
beside it, in `index/protocol.py`.
"""

import math
from numbers import Real


def check_positive_int(value: object, *, name: str) -> None:
    """Check that `value` is an `int` of at least 1.

    The type is compared exactly, so a `bool` is refused although it is an `int`.

    Args:
        value: The value to check, e.g. a limit, a depth or a size.
        name: What the value is called where it was given, named in the message.

    Raises:
        ValueError: `value` isn't an `int`, or is below 1.
    """
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def to_finite_float(value: object, *, name: str) -> float:
    """Return `value` as a `float`, after checking it is a finite real number.

    Any real number is taken (an `int`, a numpy float), except a `bool`.

    Args:
        value: The number to check, e.g. one number of a vector or a score.
        name: What the number is called where it was given, named in the message.

    Returns:
        The number as a `float`.

    Raises:
        TypeError: `value` isn't a real number (a `str` or a `bool`, say).
        ValueError: `value` is NaN or infinite.
    """
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} is {value!r}; expected a float")
    # JSON has no NaN or infinity, and a NaN compares false with everything
    if not math.isfinite(value):
        raise ValueError(f"{name} is {value!r}; expected a finite number")
    return float(value)


def to_stripped_str(value: object, *, name: str) -> str:
    """Return `value` without whitespace at either end, after checking it has text.

    Whitespace at either end is taken for a slip and dropped, so `" dense"` and
    `"dense"` are one name.

    Args:
        value: The text to check, e.g. a name.
        name: What the text is called where it was given, named in the message.

    Returns:
        The text, stripped.

    Raises:
        TypeError: `value` isn't a `str`.
        ValueError: `value` is empty or only whitespace.
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} is {value!r}; expected a str")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{name} is {value!r}; expected some text")
    return stripped
