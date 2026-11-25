import pytest

from salemdex_retrieval.evaluation.comparison import Comparison, RankChange
from salemdex_retrieval.evaluation.scorecard import ScoredQuestion
from tests.evaluation.support import (
    JUDGED_DEPTH,
    question_of,
    scorecard_of,
    scored,
)

# Invented text only: the real question sets are restricted and never appear in tests


def comparison_of(
    candidate: list[ScoredQuestion], baseline: list[ScoredQuestion]
) -> Comparison:
    """Return the comparison of two cards built from these rows."""
    return Comparison(
        candidate=scorecard_of(*candidate), baseline=scorecard_of(*baseline)
    )


def ids(changes: tuple[RankChange, ...]) -> list[str]:
    return [change.question.id for change in changes]


# One change of each kind; ranks: baseline → candidate
EXAMPLE = comparison_of(
    candidate=[
        scored("up", "AUU"),  # 3 → 1
        scored("down", "UUA"),  # 1 → 3
        scored("found", "UAU"),  # missed → 2
        scored("lost", "UUU"),  # 2 → missed
        scored("same", "UAU"),  # 2 → 2
        scored("never", "UUU"),  # missed → missed
    ],
    baseline=[
        scored("up", "UUA"),
        scored("down", "AUU"),
        scored("found", "UUU"),
        scored("lost", "UAU"),
        scored("same", "UAU"),
        scored("never", "UUU"),
    ],
)


# A rank change on its own: smaller is better, and missed is worst


@pytest.mark.parametrize(
    ("baseline_rank", "candidate_rank", "expected"),
    [
        (3, 1, "improved"),
        (None, 2, "improved"),
        (1, 3, "worsened"),
        (2, None, "worsened"),
        (2, 2, "unchanged"),
        (None, None, "unchanged"),
    ],
)
def test_a_rank_change_improves_when_the_candidate_ranks_the_answer_higher(
    baseline_rank: int | None, candidate_rank: int | None, expected: str
) -> None:
    change = RankChange(
        question=question_of("q1"),
        baseline_rank=baseline_rank,
        candidate_rank=candidate_rank,
    )

    kind = (
        "improved"
        if change.is_improvement
        else "worsened"
        if change.is_regression
        else "unchanged"
    )
    assert kind == expected


def test_no_rank_change_is_both_an_improvement_and_a_regression() -> None:
    ranks = [None, 1, 2, 3]
    for baseline_rank in ranks:
        for candidate_rank in ranks:
            change = RankChange(
                question=question_of("q1"),
                baseline_rank=baseline_rank,
                candidate_rank=candidate_rank,
            )
            assert not (change.is_improvement and change.is_regression)


# The three lists


def test_the_questions_improved_are_those_the_candidate_ranks_higher() -> None:
    assert ids(EXAMPLE.questions_improved_at(3)) == ["up", "found"]


def test_the_questions_worsened_are_those_the_candidate_ranks_lower() -> None:
    assert ids(EXAMPLE.questions_worsened_at(3)) == ["down", "lost"]


def test_the_questions_unchanged_include_those_both_runs_missed() -> None:
    assert ids(EXAMPLE.questions_unchanged_at(3)) == ["same", "never"]


@pytest.mark.parametrize("k", range(1, JUDGED_DEPTH + 1))
def test_every_answerable_question_is_in_exactly_one_list(k: int) -> None:
    listed = (
        ids(EXAMPLE.questions_improved_at(k))
        + ids(EXAMPLE.questions_worsened_at(k))
        + ids(EXAMPLE.questions_unchanged_at(k))
    )

    assert sorted(listed) == sorted(
        row.question.id for row in EXAMPLE.candidate.answerable_questions
    )


def test_each_change_carries_the_question_and_both_ranks() -> None:
    [up, found] = EXAMPLE.questions_improved_at(3)

    assert (up.question.id, up.baseline_rank, up.candidate_rank) == ("up", 3, 1)
    assert (found.baseline_rank, found.candidate_rank) == (None, 2)


def test_unanswerable_questions_are_left_out() -> None:
    comparison = comparison_of(
        candidate=[scored("q1", "A"), scored("q4", "A", answerable=False)],
        baseline=[scored("q1", "U"), scored("q4", "U", answerable=False)],
    )

    assert ids(comparison.questions_improved_at(1)) == ["q1"]
    assert ids(comparison.questions_unchanged_at(1)) == []


# The cut-off: past k, a rank counts as missed


def test_a_move_below_k_is_unchanged_at_k() -> None:
    # 3 → 2: both missed in the top 1, so unchanged there; improved in the top 3
    comparison = comparison_of(
        candidate=[scored("q1", "UAU")], baseline=[scored("q1", "UUA")]
    )

    assert ids(comparison.questions_unchanged_at(1)) == ["q1"]
    assert ids(comparison.questions_improved_at(3)) == ["q1"]


def test_a_move_into_the_top_k_is_an_improvement_at_k() -> None:
    # 3 → 1: in the top 1 for the candidate only
    comparison = comparison_of(
        candidate=[scored("q1", "AUU")], baseline=[scored("q1", "UUA")]
    )

    [change] = comparison.questions_improved_at(1)
    assert (change.baseline_rank, change.candidate_rank) == (None, 1)


@pytest.mark.parametrize("k", range(1, JUDGED_DEPTH + 1))
def test_the_changes_account_for_the_difference_in_mean_reciprocal_rank(
    k: int,
) -> None:
    def reciprocal(rank: int | None) -> float:
        return 0.0 if rank is None else 1.0 / rank

    moved = EXAMPLE.questions_improved_at(k) + EXAMPLE.questions_worsened_at(k)
    answerable = len(EXAMPLE.candidate.answerable_questions)

    assert sum(
        reciprocal(change.candidate_rank) - reciprocal(change.baseline_rank)
        for change in moved
    ) / answerable == pytest.approx(
        EXAMPLE.candidate.mean_reciprocal_rank_at(k)
        - EXAMPLE.baseline.mean_reciprocal_rank_at(k)
    )


@pytest.mark.parametrize("k", [0, JUDGED_DEPTH + 1])
def test_a_comparison_refuses_a_cut_off_its_cards_refuse(k: int) -> None:
    with pytest.raises(ValueError, match="k"):
        EXAMPLE.questions_improved_at(k)


# Tags


TAGGED = comparison_of(
    candidate=[
        scored("q1", "A", tags=("paraphrase",)),
        scored("q2", "U", tags=("exact",)),
        scored("q3", "A", tags=("exact", "paraphrase")),
    ],
    baseline=[
        scored("q1", "U", tags=("paraphrase",)),
        scored("q2", "A", tags=("exact",)),
        scored("q3", "A", tags=("exact", "paraphrase")),
    ],
)


def test_a_comparison_for_a_tag_compares_only_its_questions() -> None:
    paraphrase = TAGGED.for_tag("paraphrase")

    assert ids(paraphrase.questions_improved_at(1)) == ["q1"]
    assert ids(paraphrase.questions_worsened_at(1)) == []
    assert ids(paraphrase.questions_unchanged_at(1)) == ["q3"]


def test_a_comparison_for_a_tag_refuses_a_tag_no_question_has() -> None:
    with pytest.raises(ValueError, match="'typo'"):
        TAGGED.for_tag("typo")
