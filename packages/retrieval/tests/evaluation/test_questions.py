import dataclasses
from typing import Any

import pytest

from salemdex_retrieval.evaluation.questions import KeyPoint, Question, Quote
from salemdex_retrieval.filters import Everything, Filter, or_
from salemdex_retrieval.schema import One, Schema

# Invented text only: the real question sets are restricted and never appear in tests


class Notes(Schema):
    document = One(str, indexed=True)


IN_HANDBOOK = Notes.document.eq("handbook")
IN_GUIDE = Notes.document.eq("guide")

TEXTS_WITHOUT_TEXT = ["", "   ", "\n\t"]


def quote(
    text: str = "staff may work from home", *, where: Filter = IN_HANDBOOK
) -> Quote:
    return Quote(text=text, where=where)


def key_point(*quotes: Quote) -> KeyPoint:
    return KeyPoint(quotes=quotes or (quote(),))


def question(
    *,
    id: str = "q-01",
    text: str = "Can staff work from home?",
    tags: tuple[str, ...] = ("lookup",),
    key_points: tuple[KeyPoint, ...] | None = None,
) -> Question:
    return Question(
        id=id,
        text=text,
        tags=tags,
        key_points=(key_point(),) if key_points is None else key_points,
    )


# Question


def test_a_question_keeps_its_id_text_tags_and_key_points() -> None:
    needed = (key_point(),)

    asked = question(
        id="q-07", text="Who signs leave?", tags=("lookup",), key_points=needed
    )

    assert asked.id == "q-07"
    assert asked.text == "Who signs leave?"
    assert asked.tags == ("lookup",)
    assert asked.key_points == needed


def test_a_question_strips_its_id_text_and_tags() -> None:
    asked = question(id=" q-07\n", text="  Who signs leave? ", tags=(" lookup\t",))

    assert asked.id == "q-07"
    assert asked.text == "Who signs leave?"
    assert asked.tags == ("lookup",)


def test_a_question_keeps_its_tags_as_a_sorted_tuple() -> None:
    asked = question(tags=("situational", "lookup", "several_places"))

    assert asked.tags == ("lookup", "several_places", "situational")
    assert isinstance(asked.tags, tuple)


def test_a_question_reads_its_tags_more_than_once() -> None:
    asked = question(tags=("paraphrase", "lookup"))

    assert "lookup" in asked.tags
    assert list(asked.tags) == list(asked.tags) == ["lookup", "paraphrase"]


def test_two_questions_with_the_same_tags_in_another_order_are_equal() -> None:
    assert question(tags=("a", "b")) == question(tags=("b", "a"))


def test_a_question_may_have_no_tags() -> None:
    assert question(tags=()).tags == ()


@pytest.mark.parametrize("id", TEXTS_WITHOUT_TEXT)
def test_a_question_refuses_an_id_without_text(id: str) -> None:
    with pytest.raises(ValueError, match="id"):
        question(id=id)


@pytest.mark.parametrize("text", TEXTS_WITHOUT_TEXT)
def test_a_question_refuses_a_text_without_text(text: str) -> None:
    with pytest.raises(ValueError, match="text"):
        question(text=text)


@pytest.mark.parametrize("field", ["id", "text"])
def test_a_question_refuses_an_id_or_text_that_isnt_a_str(field: str) -> None:
    with pytest.raises(TypeError, match=field):
        question(**{field: 7})  # type: ignore[arg-type]


@pytest.mark.parametrize("tag", TEXTS_WITHOUT_TEXT)
def test_a_question_refuses_a_tag_without_text_naming_its_position(tag: str) -> None:
    with pytest.raises(ValueError, match=r"tags\[1\]"):
        question(tags=("lookup", tag))


def test_a_question_refuses_a_tag_that_isnt_a_str() -> None:
    with pytest.raises(TypeError, match=r"tags\[0\]"):
        question(tags=(7,))  # type: ignore[arg-type]


def test_a_question_refuses_a_tag_given_twice() -> None:
    with pytest.raises(ValueError, match="duplicate tag 'lookup'"):
        question(tags=("lookup", "paraphrase", "lookup"))


def test_a_question_refuses_tags_alike_once_stripped() -> None:
    with pytest.raises(ValueError, match="duplicate tag 'lookup'"):
        question(tags=("lookup", " lookup "))


def test_a_question_with_key_points_is_answerable() -> None:
    assert question(key_points=(key_point(),)).is_answerable


def test_a_question_without_key_points_is_unanswerable() -> None:
    assert not question(key_points=()).is_answerable


def test_a_question_can_need_key_points_from_several_documents() -> None:
    rule = key_point(quote("leave needs a signature", where=IN_HANDBOOK))
    exception = key_point(quote("except for one day", where=IN_GUIDE))

    asked = question(key_points=(rule, exception))

    assert asked.key_points == (rule, exception)


def test_a_question_cant_be_changed() -> None:
    asked = question()

    with pytest.raises(dataclasses.FrozenInstanceError):
        asked.id = "q-99"  # type: ignore[misc]


# KeyPoint


def test_a_key_point_keeps_its_quotes_in_order() -> None:
    first = quote("leave needs a signature", where=IN_HANDBOOK)
    second = quote("leave is signed by a manager", where=IN_GUIDE)

    assert KeyPoint(quotes=(first, second)).quotes == (first, second)


def test_a_key_point_refuses_no_quotes() -> None:
    with pytest.raises(ValueError, match="at least"):
        KeyPoint(quotes=())


def test_a_key_point_cant_be_changed() -> None:
    needed = key_point()

    with pytest.raises(dataclasses.FrozenInstanceError):
        needed.quotes = ()  # type: ignore[misc]


# Quote


def test_a_quote_keeps_its_text_and_filter() -> None:
    copied = Quote(text="staff may work from home", where=IN_HANDBOOK)

    assert copied.text == "staff may work from home"
    assert copied.where == IN_HANDBOOK


def test_a_quote_strips_its_text() -> None:
    assert quote("  staff may work from home\n").text == "staff may work from home"


def test_a_quote_keeps_whitespace_and_case_inside_its_text() -> None:
    # Normalising belongs to grading, which treats both sides alike
    assert quote("Staff  may\nwork").text == "Staff  may\nwork"


@pytest.mark.parametrize("text", TEXTS_WITHOUT_TEXT)
def test_a_quote_refuses_a_text_without_text(text: str) -> None:
    with pytest.raises(ValueError, match="text"):
        quote(text)


def test_a_quote_refuses_a_text_that_isnt_a_str() -> None:
    with pytest.raises(TypeError, match="text"):
        Quote(text=7, where=IN_HANDBOOK)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "where",
    [
        pytest.param(False, id="a comparison of a field"),
        pytest.param(None, id="None"),
        pytest.param("handbook", id="a str"),
    ],
)
def test_a_quote_refuses_a_where_that_isnt_a_filter(where: Any) -> None:
    # `Notes.document == "handbook"` gives False: fields don't define `__eq__`
    with pytest.raises(TypeError, match="Quote"):
        Quote(text="staff may work from home", where=where)


@pytest.mark.parametrize(
    "where",
    [
        pytest.param(Everything(), id="anywhere"),
        pytest.param(or_(IN_HANDBOOK, IN_GUIDE), id="either of two documents"),
    ],
)
def test_a_quote_takes_any_filter(where: Filter) -> None:
    assert quote(where=where).where == where


def test_two_quotes_with_the_same_text_and_filter_are_equal() -> None:
    assert quote() == quote()
    assert quote(where=IN_HANDBOOK) != quote(where=IN_GUIDE)
