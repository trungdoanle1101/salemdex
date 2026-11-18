"""Filters choosing which chunks a search or a write applies to.

Build them from a schema's fields (`One.eq`, `One.in_`, `Many.any_of`), which check the values,
combine them with `and_`, `or_` and `not_`, or match every chunk with `Everything()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, get_args, get_origin

if TYPE_CHECKING:
    from salemdex_retrieval.schema import Many, One

# Any filter, whatever its field's type: what searches, writes and backends accept.
type Filter = Equals[Any] | In[Any] | AnyOf[Any] | And | Or | Not | Everything


@dataclass(frozen=True, kw_only=True)
class Equals[T]:
    """A filter matching chunks whose value for a `One` field equals `value`.

    Build it with `One.eq`, which checks the value against the field.

    Attributes:
        field: The field to compare.
        value: The value to match.
    """

    field: One[T]
    value: T


@dataclass(frozen=True, kw_only=True)
class In[T]:
    """A filter matching chunks whose value for a `One` field is one of `values`.

    Build it with `One.in_`, which checks each value against the field.

    Attributes:
        field: The field to compare.
        values: The values to match; none matches no chunk.
    """

    field: One[T]
    values: frozenset[T]


@dataclass(frozen=True, kw_only=True)
class AnyOf[T]:
    """A filter matching chunks holding at least one of `values` in a `Many` field.

    Build it with `Many.any_of`, which checks each value against the field.

    Attributes:
        field: The field to look in.
        values: The values to match; none matches no chunk.
    """

    field: Many[T]
    values: frozenset[T]


@dataclass(frozen=True, kw_only=True)
class Not:
    """A filter matching the chunks that `filter` doesn't match.

    Build it with `not_`.

    Attributes:
        filter: The filter to negate.
    """

    filter: Filter

    def __post_init__(self) -> None:
        check_is_filter(self.filter, owner="Not(...)")


def not_(ft: Filter, /) -> Not:
    """Return a filter matching the chunks that `ft` doesn't match.

    Example:
        not_(KbChunks.version.eq(2))

    Raises:
        TypeError: `ft` isn't a filter.
    """
    return Not(filter=ft)


@dataclass(frozen=True, kw_only=True)
class And:
    """A filter matching the chunks that every one of `filters` matches.

    Build it with `and_`. It holds at least one filter: an empty AND would match
    every chunk, so a `delete(where=…)` built from an empty list would empty the index.

    Attributes:
        filters: The filters that must all match.
    """

    filters: tuple[Filter, ...]

    def __post_init__(self) -> None:
        _check_filters(self.filters, owner="And(...)")


def and_(*filters: Filter) -> And:
    """Return a filter matching the chunks that all of `filters` match.

    Example:
        and_(KbChunks.document_id.eq(doc_id), not_(KbChunks.version.eq(2)))

    Raises:
        ValueError: No filter is given.
        TypeError: An argument isn't a filter.
    """
    return And(filters=filters)


@dataclass(frozen=True, kw_only=True)
class Or:
    """A filter matching the chunks that at least one of `filters` matches.

    Build it with `or_`. It holds at least one filter, like `And`.

    Attributes:
        filters: The filters of which one must match.
    """

    filters: tuple[Filter, ...]

    def __post_init__(self) -> None:
        _check_filters(self.filters, owner="Or(...)")


def or_(*filters: Filter) -> Or:
    """Return a filter matching the chunks that any of `filters` matches.

    Example:
        or_(KbChunks.groups.any_of(my_groups), KbChunks.owner_id.eq(me))

    Raises:
        ValueError: No filter is given.
        TypeError: An argument isn't a filter.
    """
    return Or(filters=filters)


@dataclass(frozen=True)
class Everything:
    """A filter matching every chunk, like SQL with no `WHERE`.

    Accepted wherever a filter is, writes included: `delete(where=Everything())`
    empties the index, as `DELETE FROM chunks` empties a table. It has to be written
    out, so the intent is clear where it's used. An empty `and_()` is refused instead,
    because it comes from a list of conditions that turned out empty, by accident.

    It names no field, so it fits every schema. A session given it as its access
    filter lets every asker see every chunk.

    Example:
        index.delete(where=Everything())
    """


def check_is_filter(value: object, *, owner: str) -> None:
    """Check that `value` is a filter, for filters arriving untyped at run time.

    Fields don't define `__eq__`, so `KbChunks.version == 2` gives `False`, not an
    error, and a filter built at run time (from config, or by an agent) arrives as
    `Any`. mypy passes both; this catches them where they enter.

    Args:
        value: What should be a filter.
        owner: Where it was given, starting the message, e.g. `"Not(...)"`.

    Raises:
        TypeError: `value` isn't one of the `Filter` classes.
    """
    if not isinstance(value, _FILTER_CLASSES):
        raise TypeError(f"{owner}: {type(value).__name__} is not a valid Filter")


def _check_filters(filters: tuple[Filter, ...], *, owner: str) -> None:
    """Check that `filters` holds at least one filter, and only filters.

    Raises:
        ValueError: `filters` is empty.
        TypeError: An element isn't a filter.
    """
    if not filters:
        raise ValueError(f"{owner}: needs at least one Filter")
    for i, filter in enumerate(filters):
        check_is_filter(filter, owner=f"{owner}, filters[{i}]")


# The classes in `Filter`, for isinstance; read from the alias so a new node can't be
# missed. Here at the bottom because evaluating the alias needs every class defined.
_FILTER_CLASSES = tuple(get_origin(a) or a for a in get_args(Filter.__value__))
