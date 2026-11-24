import dataclasses
import uuid
from typing import Any

import pytest

from salemdex_retrieval.evaluation.judges.protocol import (
    Assessment,
    Grade,
    Judge,
    Judgment,
)
from salemdex_retrieval.evaluation.judges.quote import QuoteJudge

CHUNK_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
TEXTS_WITHOUT_TEXT = ["", "   ", "\n\t"]


def judgment(
    *,
    question_id: str = "q-01",
    grade: Grade = "answered",
    key_points: tuple[int, ...] = (0,),
    judge: str = "quote",
) -> Judgment:
    return Judgment(
        question_id=question_id,
        chunk_id=CHUNK_ID,
        grade=grade,
        key_points=key_points,
        judge=judge,
    )


def test_a_judgment_keeps_what_it_was_given() -> None:
    judged = judgment(question_id="q-07", grade="answered", key_points=(0, 2))

    assert judged.question_id == "q-07"
    assert judged.chunk_id == CHUNK_ID
    assert judged.grade == "answered"
    assert judged.key_points == (0, 2)
    assert judged.judge == "quote"


def test_a_judgment_strips_its_question_id_and_judge() -> None:
    judged = judgment(question_id=" q-07\n", judge="  quote ")

    assert judged.question_id == "q-07"
    assert judged.judge == "quote"


@pytest.mark.parametrize("grade", ["unrelated", "related", "answered"])
def test_a_judgment_of_any_grade_may_name_no_key_points(grade: Grade) -> None:
    # "answered" without key points: a judge that can't say which point it supplies
    assert judgment(grade=grade, key_points=()).key_points == ()


@pytest.mark.parametrize("grade", ["unrelated", "related"])
def test_only_an_answered_judgment_may_name_key_points(grade: Grade) -> None:
    with pytest.raises(ValueError, match="only a judgment of grade 'answered'"):
        judgment(grade=grade, key_points=(0,))


def test_a_judgment_refuses_a_key_point_named_twice() -> None:
    with pytest.raises(ValueError, match="duplicate key point: 1"):
        judgment(key_points=(0, 1, 1))


def test_a_judgment_refuses_a_negative_key_point() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        judgment(key_points=(0, -1))


@pytest.mark.parametrize(
    "key_point",
    [
        pytest.param(True, id="a bool"),
        pytest.param(1.0, id="a float"),
        pytest.param("1", id="a str"),
    ],
)
def test_a_judgment_refuses_a_key_point_that_isnt_an_int(key_point: Any) -> None:
    with pytest.raises(TypeError, match="key points must be integers"):
        judgment(key_points=(key_point,))


@pytest.mark.parametrize("field", ["question_id", "judge"])
@pytest.mark.parametrize("value", TEXTS_WITHOUT_TEXT)
def test_a_judgment_refuses_a_question_id_or_judge_without_text(
    field: str, value: str
) -> None:
    with pytest.raises(ValueError, match=field):
        judgment(**{field: value})  # type: ignore[arg-type]


def test_a_judgment_cant_be_changed() -> None:
    judged = judgment()

    with pytest.raises(dataclasses.FrozenInstanceError):
        judged.grade = "unrelated"  # type: ignore[misc]


def test_the_quote_judge_is_a_judge() -> None:
    # mypy checks the assignment: QuoteJudge has every member the protocol names
    judge: Judge = QuoteJudge()

    assert judge.name == "quote"


def assessment_of(
    *judgments: Judgment, judge: str = "quote", judged_depth: int = 10
) -> Assessment:
    return Assessment(
        judge=judge,
        judgments={(j.question_id, j.chunk_id): j for j in judgments},
        judged_depth=judged_depth,
    )


def test_an_assessment_finds_a_judgment_by_question_and_chunk() -> None:
    judged = judgment(question_id="q-07")

    assert assessment_of(judged).judgment_of("q-07", CHUNK_ID) is judged


def test_a_chunk_never_judged_has_no_judgment_rather_than_unrelated() -> None:
    assessment = assessment_of(judgment(question_id="q-07"))

    assert assessment.judgment_of("q-08", CHUNK_ID) is None
    assert assessment.judgment_of("q-07", uuid.uuid4()) is None


def test_an_assessment_strips_its_judges_name() -> None:
    assert assessment_of(judge="  quote ").judge == "quote"


@pytest.mark.parametrize("name", TEXTS_WITHOUT_TEXT)
def test_an_assessment_refuses_a_judge_without_a_name(name: str) -> None:
    with pytest.raises(ValueError):
        assessment_of(judge=name)


def test_an_assessment_refuses_a_judgment_by_another_judge() -> None:
    with pytest.raises(ValueError, match="judged by 'llm'"):
        assessment_of(judgment(judge="llm"), judge="quote")


def test_an_assessment_refuses_a_judgment_filed_under_another_key() -> None:
    judged = judgment(question_id="q-07")

    with pytest.raises(ValueError, match="filed under"):
        Assessment(
            judge="quote", judgments={("q-08", CHUNK_ID): judged}, judged_depth=10
        )


@pytest.mark.parametrize("judged_depth", [0, -1, True])
def test_an_assessment_refuses_a_judged_depth_that_isnt_a_positive_int(
    judged_depth: int,
) -> None:
    with pytest.raises(ValueError, match="judged_depth"):
        assessment_of(judged_depth=judged_depth)


def test_an_assessment_cannot_be_changed() -> None:
    assessment = assessment_of()

    with pytest.raises(dataclasses.FrozenInstanceError):
        assessment.judge = "llm"  # type: ignore[misc]
