import dataclasses
import re
import uuid
from typing import Any

import pytest

from salemdex_retrieval.chunks import Chunk
from salemdex_retrieval.errors import SchemaMismatchError
from salemdex_retrieval.filters import (
    And,
    AnyOf,
    Equals,
    Everything,
    In,
    Not,
    Or,
    and_,
    build_chunk_matcher,
    check_is_filter,
    not_,
    or_,
)
from salemdex_retrieval.schema import Many, One, Schema


class Chunks(Schema):
    document_id = One(uuid.UUID, indexed=True)
    version = One(int)
    position = One(int)
    groups = Many(uuid.UUID, indexed=True)
    headings = Many(str)


class Titles(Schema):
    title = One(str)


DOCUMENT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
HR = uuid.UUID("00000000-0000-0000-0000-00000000000a")
LEGAL = uuid.UUID("00000000-0000-0000-0000-00000000000b")


def test_eq_describes_the_field_and_the_value() -> None:
    where = Chunks.document_id.eq(DOCUMENT_ID)

    assert where == Equals(field=Chunks.document_id, value=DOCUMENT_ID)
    assert where.field.name == "document_id"


def test_eq_on_different_fields_gives_different_filters() -> None:
    assert Chunks.version.eq(1) != Chunks.position.eq(1)


def test_eq_rejects_a_value_not_of_the_field_type() -> None:
    with pytest.raises(TypeError, match="document_id is 'str', expected UUID"):
        Chunks.document_id.eq(str(DOCUMENT_ID))  # type: ignore[arg-type]


def test_a_filter_cannot_be_changed_after_it_is_built() -> None:
    where = Equals(field=Chunks.version, value=1)

    with pytest.raises(dataclasses.FrozenInstanceError):
        where.value = 2  # type: ignore[misc]


def test_equals_takes_its_parts_by_keyword_only() -> None:
    with pytest.raises(TypeError):
        Equals(Chunks.version, 1)  # type: ignore[call-arg]


def test_any_of_describes_the_field_and_the_values() -> None:
    where = Chunks.groups.any_of([HR, LEGAL])

    assert where == AnyOf(field=Chunks.groups, values=frozenset({HR, LEGAL}))
    assert where.field.name == "groups"


def test_any_of_ignores_the_order_and_repeats_of_its_values() -> None:
    assert Chunks.groups.any_of([HR, LEGAL]) == Chunks.groups.any_of([LEGAL, HR, HR])


def test_any_of_keeps_its_own_copy_of_the_values() -> None:
    group_ids = [HR]
    where = Chunks.groups.any_of(group_ids)

    group_ids.append(LEGAL)

    assert where == AnyOf(field=Chunks.groups, values=frozenset({HR}))


def test_any_of_accepts_any_iterable_such_as_a_generator() -> None:
    where = Chunks.groups.any_of(group_id for group_id in [HR, LEGAL])

    assert where == AnyOf(field=Chunks.groups, values=frozenset({HR, LEGAL}))


def test_any_of_names_the_position_of_a_value_not_of_the_field_type() -> None:
    with pytest.raises(TypeError, match=re.escape("groups[1] is 'str', expected UUID")):
        Chunks.groups.any_of([HR, str(LEGAL)])  # type: ignore[list-item]


def test_any_of_with_no_values_keeps_no_values() -> None:
    assert Chunks.groups.any_of([]) == AnyOf(field=Chunks.groups, values=frozenset())


@pytest.mark.parametrize(
    ("wrong", "type_name"),
    [("intro", "str"), (b"ab", "bytes"), ({"a": 1}, "dict")],
    ids=["str", "bytes", "dict"],
)
def test_any_of_rejects_a_string_bytes_or_mapping_naming_its_type(
    wrong: Any, type_name: str
) -> None:
    message = f"any_of(...) needs a collection of values, not {type_name!r}"
    with pytest.raises(TypeError, match=re.escape(message)):
        Chunks.headings.any_of(wrong)


def test_in_describes_the_field_and_the_values() -> None:
    where = Chunks.version.in_([1, 2])

    assert where == In(field=Chunks.version, values=frozenset({1, 2}))
    assert where.field.name == "version"


def test_in_ignores_the_order_and_repeats_of_its_values() -> None:
    assert Chunks.version.in_([1, 2]) == Chunks.version.in_([2, 1, 1])


def test_in_keeps_its_own_copy_of_the_values() -> None:
    versions = [1]
    where = Chunks.version.in_(versions)

    versions.append(2)

    assert where == In(field=Chunks.version, values=frozenset({1}))


def test_in_accepts_any_iterable_such_as_a_generator() -> None:
    where = Chunks.version.in_(version for version in [1, 2])

    assert where == In(field=Chunks.version, values=frozenset({1, 2}))


def test_in_names_the_position_of_a_value_not_of_the_field_type() -> None:
    with pytest.raises(TypeError, match=re.escape("version[1] is 'str', expected int")):
        Chunks.version.in_([1, "2"])  # type: ignore[list-item]


def test_in_with_no_values_keeps_no_values() -> None:
    assert Chunks.version.in_([]) == In(field=Chunks.version, values=frozenset())


@pytest.mark.parametrize(
    ("wrong", "type_name"),
    [("intro", "str"), (b"ab", "bytes"), ({"a": 1}, "dict")],
    ids=["str", "bytes", "dict"],
)
def test_in_rejects_a_string_bytes_or_mapping_naming_its_type(
    wrong: Any, type_name: str
) -> None:
    message = f"in_(...) needs a collection of values, not {type_name!r}"
    with pytest.raises(TypeError, match=re.escape(message)):
        Titles.title.in_(wrong)


def test_and_holds_its_filters_in_order() -> None:
    first, second = Chunks.document_id.eq(DOCUMENT_ID), Chunks.version.eq(2)

    assert and_(first, second) == And(filters=(first, second))


def test_or_holds_its_filters_in_order() -> None:
    first, second = Chunks.groups.any_of([HR]), Chunks.version.eq(2)

    assert or_(first, second) == Or(filters=(first, second))


def test_not_holds_the_filter_it_negates() -> None:
    negated = Chunks.version.eq(2)

    assert not_(negated) == Not(filter=negated)


def test_and_or_accept_a_single_filter() -> None:
    only = Chunks.version.eq(2)

    assert and_(only).filters == (only,)
    assert or_(only).filters == (only,)


def test_combinators_nest() -> None:
    where = and_(
        Chunks.document_id.eq(DOCUMENT_ID),
        not_(or_(Chunks.version.eq(1), not_(Chunks.version.eq(2)))),
    )

    assert isinstance(where.filters[1], Not)
    assert isinstance(where.filters[1].filter, Or)


def test_everything_is_a_filter_the_combinators_accept() -> None:
    everything = Everything()

    assert and_(everything, Chunks.version.eq(2)).filters[0] is everything
    assert or_(Chunks.version.eq(2), everything).filters[1] is everything
    assert not_(everything).filter is everything


def test_everything_equals_another_everything() -> None:
    assert Everything() == Everything()
    assert not_(Everything()) == not_(Everything())


@pytest.mark.parametrize(
    "where",
    [
        pytest.param(Chunks.version.eq(2), id="eq"),
        pytest.param(Chunks.version.in_([2]), id="in_"),
        pytest.param(Chunks.groups.any_of([HR]), id="any_of"),
        pytest.param(and_(Chunks.version.eq(2)), id="and_"),
        pytest.param(or_(Chunks.version.eq(2)), id="or_"),
        pytest.param(not_(Chunks.version.eq(2)), id="not_"),
        pytest.param(Everything(), id="Everything"),
    ],
)
def test_check_is_filter_accepts_every_kind_of_filter(where: object) -> None:
    check_is_filter(where, owner="access")


def test_check_is_filter_names_where_the_value_was_given_and_its_type() -> None:
    with pytest.raises(
        TypeError, match=re.escape("access: bool is not a valid Filter")
    ):
        check_is_filter(Chunks.version == 2, owner="access")  # type: ignore[comparison-overlap]


@pytest.mark.parametrize(
    ("build", "owner"),
    [pytest.param(and_, "And", id="and_"), pytest.param(or_, "Or", id="or_")],
)
def test_and_or_reject_no_filters(build: Any, owner: str) -> None:
    with pytest.raises(ValueError, match=re.escape(f"{owner}(...): needs at least")):
        build()


def test_and_rejects_no_filters_even_from_an_empty_list() -> None:
    conditions: list[Equals[Any]] = []

    with pytest.raises(ValueError, match="at least one"):
        and_(*conditions)


@pytest.mark.parametrize(
    ("build", "owner"),
    [pytest.param(and_, "And", id="and_"), pytest.param(or_, "Or", id="or_")],
)
def test_and_or_name_the_position_and_type_of_an_argument_that_is_not_a_filter(
    build: Any, owner: str
) -> None:
    # A bool is what comparing a field with == gives, since fields don't define __eq__
    wrong = Chunks.version == 2  # type: ignore[comparison-overlap]
    message = f"{owner}(...), filters[1]: bool is not a valid Filter"

    with pytest.raises(TypeError, match=re.escape(message)):
        build(Chunks.version.eq(1), wrong)


@pytest.mark.parametrize(
    ("wrong", "type_name"),
    [(True, "bool"), ("version = 2", "str"), (None, "NoneType")],
    ids=["bool", "str", "None"],
)
def test_not_names_the_type_of_an_argument_that_is_not_a_filter(
    wrong: Any, type_name: str
) -> None:
    with pytest.raises(
        TypeError, match=re.escape(f"Not(...): {type_name} is not a valid Filter")
    ):
        not_(wrong)


def test_a_combinator_built_directly_is_checked_too() -> None:
    with pytest.raises(ValueError, match="at least one"):
        And(filters=())

    with pytest.raises(TypeError, match="not a valid Filter"):
        Or(filters=(Chunks.version.eq(1), "version = 2"))  # type: ignore[arg-type]


# build_chunk_matcher


def chunk(
    *,
    document_id: uuid.UUID = DOCUMENT_ID,
    version: int = 1,
    groups: tuple[uuid.UUID, ...] = (HR,),
) -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        text="text",
        field_values=[
            Chunks.document_id.value(document_id),
            Chunks.version.value(version),
            Chunks.position.value(0),
            Chunks.groups.value(groups),
            Chunks.headings.value(()),
        ],
    )


OTHER_DOCUMENT = uuid.UUID("00000000-0000-0000-0000-000000000002")


@pytest.mark.parametrize(
    ("where", "expected"),
    [
        pytest.param(Chunks.version.eq(1), True, id="eq, same value"),
        pytest.param(Chunks.version.eq(2), False, id="eq, other value"),
        pytest.param(Chunks.version.in_([1, 3]), True, id="in_, among the values"),
        pytest.param(Chunks.version.in_([2, 3]), False, id="in_, not among them"),
        pytest.param(Chunks.version.in_([]), False, id="in_, no values"),
        pytest.param(Chunks.groups.any_of([HR, LEGAL]), True, id="any_of, sharing one"),
        pytest.param(Chunks.groups.any_of([LEGAL]), False, id="any_of, sharing none"),
        pytest.param(Chunks.groups.any_of([]), False, id="any_of, no values"),
        pytest.param(
            and_(Chunks.version.eq(1), Chunks.document_id.eq(DOCUMENT_ID)),
            True,
            id="and_, both match",
        ),
        pytest.param(
            and_(Chunks.version.eq(1), Chunks.document_id.eq(OTHER_DOCUMENT)),
            False,
            id="and_, one doesn't",
        ),
        pytest.param(
            or_(Chunks.version.eq(2), Chunks.document_id.eq(DOCUMENT_ID)),
            True,
            id="or_, one matches",
        ),
        pytest.param(
            or_(Chunks.version.eq(2), Chunks.document_id.eq(OTHER_DOCUMENT)),
            False,
            id="or_, neither matches",
        ),
        pytest.param(not_(Chunks.version.eq(2)), True, id="not_, inner doesn't match"),
        pytest.param(not_(Chunks.version.eq(1)), False, id="not_, inner matches"),
        pytest.param(Everything(), True, id="Everything"),
    ],
)
def test_a_chunk_matcher_tells_whether_a_chunk_matches_the_filter(
    where: Any, expected: bool
) -> None:
    assert build_chunk_matcher(where)(chunk()) is expected


def test_a_chunk_matcher_matches_nested_filters() -> None:
    where = and_(
        or_(Chunks.version.eq(2), Chunks.groups.any_of([HR])),
        not_(Chunks.document_id.eq(OTHER_DOCUMENT)),
    )
    matches = build_chunk_matcher(where)

    assert matches(chunk())
    assert not matches(chunk(document_id=OTHER_DOCUMENT))
    assert not matches(chunk(groups=(LEGAL,)))


def test_a_chunk_matcher_is_built_once_and_used_on_many_chunks() -> None:
    matches = build_chunk_matcher(Chunks.version.in_([1, 2]))

    found = [matches(chunk(version=version)) for version in (1, 2, 3, 1)]

    assert found == [True, True, False, True]


def test_any_of_never_matches_a_chunk_with_no_values_for_its_field() -> None:
    assert not build_chunk_matcher(Chunks.groups.any_of([HR]))(chunk(groups=()))


def test_a_chunk_matcher_refuses_a_chunk_without_the_filters_field() -> None:
    # The filter must be checked against the chunks' schema first; a chunk of
    # another schema is an error, never quietly no match
    title = Chunk(id=uuid.uuid4(), text="text", field_values=[Titles.title.value("A")])

    with pytest.raises(SchemaMismatchError):
        build_chunk_matcher(Chunks.version.eq(1))(title)


def test_a_chunk_matcher_compares_fields_as_objects_not_by_name() -> None:
    class Other(Schema):
        version = One(int)

    with pytest.raises(SchemaMismatchError):
        build_chunk_matcher(Other.version.eq(1))(chunk())
