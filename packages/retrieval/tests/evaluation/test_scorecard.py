import uuid
from collections.abc import Callable

import pytest

from salemdex_retrieval.engine.results import Found
from salemdex_retrieval.engine.stages import source
from salemdex_retrieval.engine.statements import search
from salemdex_retrieval.evaluation.judges.protocol import Grade, Judgment
from salemdex_retrieval.evaluation.questions import KeyPoint, Question, Quote
from salemdex_retrieval.evaluation.runs import ReturnedChunk
from salemdex_retrieval.evaluation.scorecard import (
    RankedChunk,
    Scorecard,
    ScoredQuestion,
)
from salemdex_retrieval.filters import Everything

# Invented text only: the real question sets are restricted and never appear in tests

STATEMENT = search().retrieve(source("body", depth=10)).take()
JUDGED_DEPTH = 10

# One letter per returned chunk, best first: A answered, R related, U unrelated
type Grades = str

GRADE_OF_LETTER: dict[str, Grade] = {"A": "answered", "R": "related", "U": "unrelated"}


def scored(
    question_id: str, grades: Grades, *, answerable: bool = True
) -> ScoredQuestion:
    """Return a row for a question whose returned chunks carry `grades`, best first."""
    question = Question(
        id=question_id,
        text="Can staff work from home?",
        tags=(),
        key_points=(
            (KeyPoint(quotes=(Quote(text="two days a week", where=Everything()),)),)
            if answerable
            else ()
        ),
    )
    return ScoredQuestion(
        question=question,
        chunks=tuple(
            ranked(question_id, letter, rank=rank)
            for rank, letter in enumerate(grades, start=1)
        ),
    )


def ranked(question_id: str, letter: str, *, rank: int) -> RankedChunk:
    chunk_id = uuid.uuid4()
    grade = GRADE_OF_LETTER[letter]
    return RankedChunk(
        returned=ReturnedChunk(
            chunk_id=chunk_id, trail=(Found(source="body", rank=rank, score=0.5),)
        ),
        judgment=Judgment(
            question_id=question_id,
            chunk_id=chunk_id,
            grade=grade,
            key_points=(0,) if grade == "answered" else (),
            judge="quote",
        ),
    )


def scorecard_of(*questions: ScoredQuestion) -> Scorecard:
    return Scorecard(
        statement=STATEMENT,
        questions=questions,
        questions_left_out=(),
        judged_depth=JUDGED_DEPTH,
    )


# q1 answered at rank 2, q2 at rank 8, q3 not in the judged depth
EXAMPLE = scorecard_of(
    scored("q1", "UAUUAUUUUU"),
    scored("q2", "UUUUUUUAUU"),
    scored("q3", "UUUUUUUUUU"),
)

RANK_METRICS: list[Callable[[Scorecard, int], float]] = [
    Scorecard.hit_rate_at,
    Scorecard.mean_reciprocal_rank_at,
]


def test_a_questions_rank_is_that_of_its_first_answered_chunk_counted_from_one() -> (
    None
):
    assert scored("q1", "UAUA").first_answered_rank == 2


def test_a_question_with_no_answered_chunk_has_no_rank() -> None:
    assert scored("q1", "UUU").first_answered_rank is None


def test_a_related_chunk_doesnt_answer() -> None:
    assert scored("q1", "RRA").first_answered_rank == 3


def test_the_hit_rate_is_the_share_of_questions_answered_within_k() -> None:
    assert EXAMPLE.hit_rate_at(5) == pytest.approx(1 / 3)
    assert EXAMPLE.hit_rate_at(10) == pytest.approx(2 / 3)


def test_the_hit_rate_leaves_unanswerable_questions_out() -> None:
    card = scorecard_of(scored("q1", "AU"), scored("q4", "UU", answerable=False))

    assert card.hit_rate_at(1) == 1.0


def test_the_mean_reciprocal_rank_averages_one_over_each_questions_rank() -> None:
    assert EXAMPLE.mean_reciprocal_rank_at(10) == pytest.approx((1 / 2 + 1 / 8) / 3)


def test_the_mean_reciprocal_rank_counts_a_rank_below_k_as_zero() -> None:
    assert EXAMPLE.mean_reciprocal_rank_at(5) == pytest.approx((1 / 2) / 3)


def test_the_mean_reciprocal_rank_leaves_unanswerable_questions_out() -> None:
    card = scorecard_of(scored("q1", "UA"), scored("q4", "UU", answerable=False))

    assert card.mean_reciprocal_rank_at(2) == pytest.approx(1 / 2)


@pytest.mark.parametrize("metric", RANK_METRICS)
def test_a_rank_metric_refuses_a_cut_off_past_the_judged_depth(
    metric: Callable[[Scorecard, int], float],
) -> None:
    with pytest.raises(ValueError, match=str(JUDGED_DEPTH)):
        metric(EXAMPLE, JUDGED_DEPTH + 1)


@pytest.mark.parametrize("metric", RANK_METRICS)
def test_a_rank_metric_accepts_a_cut_off_at_the_judged_depth(
    metric: Callable[[Scorecard, int], float],
) -> None:
    metric(EXAMPLE, JUDGED_DEPTH)


@pytest.mark.parametrize("metric", RANK_METRICS)
@pytest.mark.parametrize("k", [0, -1, True])
def test_a_rank_metric_refuses_a_cut_off_that_isnt_a_positive_int(
    metric: Callable[[Scorecard, int], float], k: int
) -> None:
    with pytest.raises(ValueError, match="k must be a positive integer"):
        metric(EXAMPLE, k)


@pytest.mark.parametrize("metric", RANK_METRICS)
def test_a_rank_metric_refuses_a_card_with_only_unanswerable_questions(
    metric: Callable[[Scorecard, int], float],
) -> None:
    card = scorecard_of(scored("q4", "UU", answerable=False))

    with pytest.raises(ValueError, match="no answerable questions"):
        metric(card, 1)


@pytest.mark.parametrize("metric", RANK_METRICS)
def test_a_rank_metric_refuses_a_card_with_no_questions(
    metric: Callable[[Scorecard, int], float],
) -> None:
    with pytest.raises(ValueError, match="no answerable questions"):
        metric(scorecard_of(), 1)
