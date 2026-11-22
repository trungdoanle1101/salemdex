import uuid

import pytest

from salemdex_retrieval.chunks import Chunk
from salemdex_retrieval.evaluation.judges.quote import QuoteJudge
from salemdex_retrieval.evaluation.questions import KeyPoint, Question, Quote
from salemdex_retrieval.filters import Filter
from salemdex_retrieval.schema import One, Schema

# Invented text only: the real question sets are restricted and never appear in tests


class Notes(Schema):
    document = One(str, indexed=True)


IN_HANDBOOK = Notes.document.eq("handbook")
IN_GUIDE = Notes.document.eq("guide")


def chunk(
    text: str, *, document: str = "handbook", embedded_text: str | None = None
) -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        text=text,
        embedded_text=embedded_text,
        field_values=[Notes.document.value(document)],
    )


def key_point(*texts: str, where: Filter = IN_HANDBOOK) -> KeyPoint:
    return KeyPoint(quotes=tuple(Quote(text=text, where=where) for text in texts))


def question(*key_points: KeyPoint) -> Question:
    return Question(
        id="q-01", text="Can staff work from home?", tags=(), key_points=key_points
    )


def judge_one(asked: Question, judged: Chunk) -> tuple[str, tuple[int, ...]]:
    [judgment] = QuoteJudge().judge(asked, [judged])
    return judgment.grade, judgment.key_points


RULE = key_point("staff may work from home")
EXCEPTION = key_point("except on fridays")


# QuoteJudge


def test_the_quote_judge_is_named_quote() -> None:
    assert QuoteJudge().name == "quote"


def test_a_chunk_holding_the_quote_answers_its_key_point() -> None:
    assert judge_one(question(RULE), chunk("Staff may work from home.")) == (
        "answered",
        (0,),
    )


def test_a_chunk_without_the_quote_is_unrelated() -> None:
    assert judge_one(question(RULE), chunk("The office opens at 8.")) == (
        "unrelated",
        (),
    )


def test_every_chunk_is_unrelated_to_an_unanswerable_question() -> None:
    judgments = QuoteJudge().judge(
        question(), [chunk("Staff may work from home."), chunk("The office opens.")]
    )

    assert [(j.grade, j.key_points) for j in judgments] == [
        ("unrelated", ()),
        ("unrelated", ()),
    ]


def test_any_one_of_a_key_points_quotes_supplies_it() -> None:
    # The quotes are alternatives: the same point stated in different words
    either = key_point("staff may work from home", "remote work is allowed")

    assert judge_one(question(either), chunk("Staff may work from home.")) == (
        "answered",
        (0,),
    )
    assert judge_one(question(either), chunk("Remote work is allowed.")) == (
        "answered",
        (0,),
    )


def test_a_key_point_stated_in_two_documents_is_supplied_by_either() -> None:
    either = KeyPoint(
        quotes=(
            Quote(text="staff may work from home", where=IN_HANDBOOK),
            Quote(text="employees can work remotely", where=IN_GUIDE),
        )
    )

    in_guide = chunk("Employees can work remotely.", document="guide")

    assert judge_one(question(either), in_guide) == ("answered", (0,))


def test_a_chunk_supplying_one_key_point_of_two_answers() -> None:
    # The grade is about this chunk; whether every point was found is the
    # scorecard's question, across the top passages
    assert judge_one(question(RULE, EXCEPTION), chunk("Staff may work from home.")) == (
        "answered",
        (0,),
    )


def test_a_chunk_supplying_several_key_points_names_them_all() -> None:
    both = chunk("Staff may work from home, except on Fridays.")

    assert judge_one(question(RULE, EXCEPTION), both) == ("answered", (0, 1))


def test_key_points_are_named_by_their_position_in_the_question() -> None:
    assert judge_one(question(RULE, EXCEPTION), chunk("Except on Fridays.")) == (
        "answered",
        (1,),
    )


def test_a_quote_outside_its_where_supplies_nothing() -> None:
    only_in_guide = key_point("staff may work from home", where=IN_GUIDE)

    assert judge_one(question(only_in_guide), chunk("Staff may work from home.")) == (
        "unrelated",
        (),
    )


def test_a_quote_found_only_in_the_embedded_text_supplies_nothing() -> None:
    # The embedded text adds the headings path, which a reader never sees as the
    # passage's words
    headed = chunk("Staff may rest.", embedded_text="Leave policy > Staff may rest.")

    assert judge_one(question(key_point("leave policy")), headed) == ("unrelated", ())


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("STAFF MAY WORK FROM HOME", id="another case"),
        pytest.param("Staff may\nwork  from\thome", id="other whitespace"),
        pytest.param("Staff may work from home", id="a non-breaking space"),
    ],
)
def test_a_quote_matches_whatever_its_case_and_whitespace(text: str) -> None:
    assert judge_one(question(RULE), chunk(text)) == ("answered", (0,))


def test_a_quote_matches_whether_written_composed_or_decomposed() -> None:
    # "é" as one character (NFC) in the chunk, as "e" and a combining accent (NFD)
    # in the quote
    decomposed = key_point("cho phép")

    assert judge_one(question(decomposed), chunk("Được cho phép.")) == (
        "answered",
        (0,),
    )


def test_a_quote_split_across_two_chunks_is_in_neither() -> None:
    judgments = QuoteJudge().judge(
        question(RULE), [chunk("Staff may work"), chunk("from home.")]
    )

    assert [j.grade for j in judgments] == ["unrelated", "unrelated"]


def test_the_quote_judge_judges_every_chunk_in_order() -> None:
    chunks = [
        chunk("The office opens at 8."),
        chunk("Staff may work from home."),
        chunk("Except on Fridays."),
    ]

    judgments = QuoteJudge().judge(question(RULE, EXCEPTION), chunks)

    assert [j.chunk_id for j in judgments] == [c.id for c in chunks]
    assert [j.key_points for j in judgments] == [(), (0,), (1,)]


def test_the_quote_judge_names_the_question_and_itself_on_each_judgment() -> None:
    [judgment] = QuoteJudge().judge(question(RULE), [chunk("Staff may work.")])

    assert judgment.question_id == "q-01"
    assert judgment.judge == "quote"


def test_the_quote_judge_takes_chunks_from_a_generator() -> None:
    chunks = (chunk(text) for text in ["Staff may work from home.", "Other."])

    judgments = QuoteJudge().judge(question(RULE), chunks)

    assert [j.grade for j in judgments] == ["answered", "unrelated"]


def test_the_quote_judge_given_no_chunks_judges_nothing() -> None:
    assert QuoteJudge().judge(question(RULE), []) == []
