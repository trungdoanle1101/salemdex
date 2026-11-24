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


def question_of(
    question_id: str, *, key_point_count: int = 1, tags: tuple[str, ...] = ()
) -> Question:
    """Return a question with `key_point_count` key points; none is unanswerable."""
    return Question(
        id=question_id,
        text="Can staff work from home?",
        tags=tags,
        key_points=tuple(
            KeyPoint(quotes=(Quote(text=f"key point {n}", where=Everything()),))
            for n in range(key_point_count)
        ),
    )


def scored(
    question_id: str,
    grades: Grades,
    *,
    answerable: bool = True,
    tags: tuple[str, ...] = (),
) -> ScoredQuestion:
    """Return a row for a one-key-point question whose chunks carry `grades`."""
    return ScoredQuestion(
        question=question_of(
            question_id, key_point_count=1 if answerable else 0, tags=tags
        ),
        chunks=tuple(
            ranked(
                question_id,
                grade=GRADE_OF_LETTER[letter],
                key_points=(0,) if letter == "A" else (),
                rank=rank,
            )
            for rank, letter in enumerate(grades, start=1)
        ),
    )


def covering(
    question_id: str, *supplied: tuple[int, ...] | None, key_point_count: int
) -> ScoredQuestion:
    """Return a row whose chunks, best first, supply these key points.

    Each entry is one chunk: the key points it answers, or None for an unrelated
    chunk. An empty tuple is an answered chunk that names no key point.
    """
    return ScoredQuestion(
        question=question_of(question_id, key_point_count=key_point_count),
        chunks=tuple(
            ranked(
                question_id,
                grade="unrelated" if key_points is None else "answered",
                key_points=key_points or (),
                rank=rank,
            )
            for rank, key_points in enumerate(supplied, start=1)
        ),
    )


def ranked(
    question_id: str, *, grade: Grade, key_points: tuple[int, ...], rank: int
) -> RankedChunk:
    chunk_id = uuid.uuid4()
    return RankedChunk(
        returned=ReturnedChunk(
            chunk_id=chunk_id, trail=(Found(source="body", rank=rank, score=0.5),)
        ),
        judgment=Judgment(
            question_id=question_id,
            chunk_id=chunk_id,
            grade=grade,
            key_points=key_points,
            judge="quote",
        ),
    )


def scorecard_of(
    *questions: ScoredQuestion, left_out: tuple[Question, ...] = ()
) -> Scorecard:
    return Scorecard(
        statement=STATEMENT,
        questions=questions,
        questions_left_out=left_out,
        judged_depth=JUDGED_DEPTH,
    )


# q1 answered at rank 2, q2 at rank 8, q3 not in the judged depth
EXAMPLE = scorecard_of(
    scored("q1", "UAUUAUUUUU"),
    scored("q2", "UUUUUUUAUU"),
    scored("q3", "UUUUUUUUUU"),
)

# Every metric that takes a cut-off checks it, and its subset, the same way
CUT_OFF_METRICS: list[Callable[[Scorecard, int], object]] = [
    Scorecard.hit_rate_at,
    Scorecard.mean_reciprocal_rank_at,
    Scorecard.all_key_points_rate_at,
    Scorecard.questions_missed_at,
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


@pytest.mark.parametrize("metric", CUT_OFF_METRICS)
def test_a_metric_refuses_a_cut_off_past_the_judged_depth(
    metric: Callable[[Scorecard, int], object],
) -> None:
    with pytest.raises(ValueError, match=str(JUDGED_DEPTH)):
        metric(EXAMPLE, JUDGED_DEPTH + 1)


@pytest.mark.parametrize("metric", CUT_OFF_METRICS)
def test_a_metric_accepts_a_cut_off_at_the_judged_depth(
    metric: Callable[[Scorecard, int], object],
) -> None:
    metric(EXAMPLE, JUDGED_DEPTH)


@pytest.mark.parametrize("metric", CUT_OFF_METRICS)
@pytest.mark.parametrize("k", [0, -1, True])
def test_a_metric_refuses_a_cut_off_that_isnt_a_positive_int(
    metric: Callable[[Scorecard, int], object], k: int
) -> None:
    with pytest.raises(ValueError, match="k must be a positive integer"):
        metric(EXAMPLE, k)


@pytest.mark.parametrize("metric", CUT_OFF_METRICS)
def test_a_metric_refuses_a_card_with_only_unanswerable_questions(
    metric: Callable[[Scorecard, int], object],
) -> None:
    card = scorecard_of(scored("q4", "UU", answerable=False))

    with pytest.raises(ValueError, match="no answerable questions"):
        metric(card, 1)


@pytest.mark.parametrize("metric", CUT_OFF_METRICS)
def test_a_metric_refuses_a_card_with_no_questions(
    metric: Callable[[Scorecard, int], object],
) -> None:
    with pytest.raises(ValueError, match="no answerable questions"):
        metric(scorecard_of(), 1)


# All key points: every key point supplied within k, by one chunk or several


def test_the_all_key_points_rank_is_where_the_chunks_first_cover_every_key_point() -> (
    None
):
    row = covering("q1", (0,), None, (0,), (1,), key_point_count=2)

    assert row.all_key_points_rank == 4


def test_one_chunk_supplying_every_key_point_covers_them_at_its_rank() -> None:
    assert covering("q1", None, (0, 1), key_point_count=2).all_key_points_rank == 2


def test_a_question_whose_key_points_are_never_all_supplied_has_no_rank() -> None:
    row = covering("q1", (0,), (0,), None, key_point_count=2)

    assert row.all_key_points_rank is None


def test_with_one_key_point_the_all_key_points_rank_is_the_first_answered_rank() -> (
    None
):
    row = scored("q1", "URA")

    assert row.all_key_points_rank == row.first_answered_rank == 3


def test_the_all_key_points_rate_counts_a_question_once_its_last_key_point_is_in() -> (
    None
):
    # key point 0 at rank 2, key point 1 at rank 7
    card = scorecard_of(
        covering("q1", None, (0,), None, None, None, None, (1,), key_point_count=2)
    )

    assert card.all_key_points_rate_at(5) == 0.0
    assert card.all_key_points_rate_at(7) == 1.0


@pytest.mark.parametrize("k", range(1, JUDGED_DEPTH + 1))
def test_with_one_key_point_per_question_the_all_key_points_rate_is_the_hit_rate(
    k: int,
) -> None:
    assert EXAMPLE.all_key_points_rate_at(k) == EXAMPLE.hit_rate_at(k)


@pytest.mark.parametrize("k", range(1, JUDGED_DEPTH + 1))
def test_the_all_key_points_rate_is_never_above_the_hit_rate(k: int) -> None:
    card = scorecard_of(
        covering("q1", None, (0,), None, None, None, None, (1,), key_point_count=2),
        covering("q2", None, None, (1, 0), key_point_count=2),
        scored("q3", "UUUUA"),
    )

    assert card.all_key_points_rate_at(k) <= card.hit_rate_at(k)


def test_the_all_key_points_rate_leaves_unanswerable_questions_out() -> None:
    card = scorecard_of(
        covering("q1", (0, 1), key_point_count=2), scored("q4", "UU", answerable=False)
    )

    assert card.all_key_points_rate_at(1) == 1.0


def test_the_all_key_points_rate_refuses_an_answered_chunk_naming_no_key_point() -> (
    None
):
    # A judge that can't say which key point a chunk supplies (an LLM's): counting it
    # as supplying none would score the question too low
    card = scorecard_of(covering("q1", (), (0,), key_point_count=1))

    with pytest.raises(ValueError, match="key point"):
        card.all_key_points_rate_at(2)


def test_the_all_key_points_rate_refuses_an_unnamed_key_point_below_k_too() -> None:
    # Naming key points is the judge's way, so the rate is readable at every k or none
    card = scorecard_of(covering("q1", (0,), (), key_point_count=1))

    with pytest.raises(ValueError, match="key point"):
        card.all_key_points_rate_at(1)


def test_an_unanswerable_question_doesnt_need_its_answered_chunks_to_name_key_points() -> (
    None
):
    # An LLM judge may grade a chunk answered for a question with no key points
    card = scorecard_of(
        covering("q1", (0,), key_point_count=1), covering("q4", (), key_point_count=0)
    )

    assert card.all_key_points_rate_at(1) == 1.0


# Questions missed: the rows to look into


def test_the_questions_missed_are_the_answerable_ones_not_answered_within_k() -> None:
    assert [row.question.id for row in EXAMPLE.questions_missed_at(5)] == ["q2", "q3"]
    assert [row.question.id for row in EXAMPLE.questions_missed_at(10)] == ["q3"]


def test_the_questions_missed_are_the_cards_rows_with_their_chunks() -> None:
    (missed,) = EXAMPLE.questions_missed_at(10)

    assert missed is EXAMPLE.questions[2]


def test_the_questions_missed_leave_unanswerable_questions_out() -> None:
    card = scorecard_of(scored("q1", "UU"), scored("q4", "UU", answerable=False))

    assert [row.question.id for row in card.questions_missed_at(2)] == ["q1"]


@pytest.mark.parametrize("k", range(1, JUDGED_DEPTH + 1))
def test_the_questions_missed_are_the_ones_the_hit_rate_doesnt_count(k: int) -> None:
    answerable = len(EXAMPLE.questions)

    assert len(EXAMPLE.questions_missed_at(k)) == pytest.approx(
        answerable * (1 - EXAMPLE.hit_rate_at(k))
    )


# Tags: a card over one kind of question


TAGGED = scorecard_of(
    scored("q1", "A", tags=("paraphrase",)),
    scored("q2", "U", tags=("exact",)),
    scored("q3", "A", tags=("exact", "paraphrase")),
    left_out=(
        question_of("q4", tags=("paraphrase",)),
        question_of("q5", tags=("table",)),
    ),
)


def test_a_card_for_a_tag_keeps_only_the_rows_and_left_out_questions_with_it() -> None:
    paraphrase = TAGGED.for_tag("paraphrase")

    assert [row.question.id for row in paraphrase.questions] == ["q1", "q3"]
    assert [question.id for question in paraphrase.questions_left_out] == ["q4"]


def test_a_card_for_a_tag_keeps_the_statement_and_the_judged_depth() -> None:
    exact = TAGGED.for_tag("exact")

    assert exact.statement == TAGGED.statement
    assert exact.judged_depth == TAGGED.judged_depth


def test_a_card_for_a_tag_scores_only_its_questions() -> None:
    assert TAGGED.for_tag("paraphrase").hit_rate_at(1) == 1.0
    assert TAGGED.for_tag("exact").hit_rate_at(1) == pytest.approx(1 / 2)


def test_a_tag_whose_questions_were_all_left_out_still_gives_a_card() -> None:
    table = TAGGED.for_tag("table")

    assert table.questions == ()
    assert [question.id for question in table.questions_left_out] == ["q5"]


def test_a_card_for_a_tag_refuses_a_tag_no_question_has() -> None:
    with pytest.raises(ValueError, match="'typo'"):
        TAGGED.for_tag("typo")


def test_a_cards_tags_are_sorted_once_each_left_out_questions_included() -> None:
    assert TAGGED.tags == ("exact", "paraphrase", "table")
