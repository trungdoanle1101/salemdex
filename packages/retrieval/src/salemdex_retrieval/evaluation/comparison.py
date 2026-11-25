"""Two runs compared question by question: which questions a change helped or hurt.

A scorecard says how well one run did overall; a comparison says where two runs
differ, so a developer can look at the questions a change moved. Each question's
outcome is its first answered rank within a cut-off `k`, the rank every scorecard
metric reads, so the questions improved and worsened at `k` are exactly the ones that
make up the difference in the two cards' mean reciprocal rank at `k`.

Both cards come from one assessment, so the two runs are graded alike: the same judge,
the same judged depth, and the same questions.
"""

from __future__ import annotations

from dataclasses import dataclass

from salemdex_retrieval.evaluation.questions import Question
from salemdex_retrieval.evaluation.scorecard import Scorecard


@dataclass(frozen=True, kw_only=True)
class RankChange:
    """Where one question's first answer ranked in each run, within a cut-off.

    A smaller rank is better, and None, no answered chunk within the cut-off, is worst.

    Attributes:
        question: The question, whole.
        baseline_rank: The baseline's first answered rank, from 1; None if it had no
            answered chunk within the cut-off.
        candidate_rank: The candidate's first answered rank, from 1; None if it had no
            answered chunk within the cut-off.

    Example:
        for change in comparison.questions_worsened_at(5):
            print(change.question.id, change.baseline_rank, change.candidate_rank)
    """

    question: Question
    baseline_rank: int | None
    candidate_rank: int | None

    @property
    def is_improvement(self) -> bool:
        """Whether the candidate ranks the answer higher, or alone finds one."""
        if self.candidate_rank is None:
            return False
        return self.baseline_rank is None or self.candidate_rank < self.baseline_rank

    @property
    def is_regression(self) -> bool:
        """Whether the candidate ranks the answer lower, or alone misses one."""
        if self.baseline_rank is None:
            return False
        return self.candidate_rank is None or self.baseline_rank < self.candidate_rank


@dataclass(frozen=True, kw_only=True)
class Comparison:
    """Two runs' scorecards, compared question by question.

    Build it with `Evaluator.compare`, which scores both runs on one assessment; built
    directly, nothing checks that the two cards can be compared.

    Every method with a cut-off `k` lists the answerable questions only, in question
    order, and refuses a `k` either card refuses: one that isn't a positive `int` or
    is past the judged depth, or a card with no answerable question.

    Attributes:
        candidate: The card of the run being tried, e.g. with a rerank.
        baseline: The card of the run it's measured against.

    Example:
        comparison = evaluator.compare(reranked, baseline=dense, assessment=assessment)
        comparison.questions_improved_at(5)
        comparison.for_tag("paraphrase").questions_worsened_at(5)
    """

    candidate: Scorecard
    baseline: Scorecard

    def questions_improved_at(self, k: int) -> tuple[RankChange, ...]:
        """Return the questions the candidate answers higher within the top `k`.

        Found by the candidate and missed by the baseline counts, and so does any move
        up within `k`: 3 → 1 at `k=1` and at `k=3`, but 3 → 2 only at `k=3`.

        Raises:
            ValueError: `k` isn't a positive `int` or is past the judged depth, or the
                cards have no answerable question.
        """
        return tuple(
            change for change in self._rank_changes_at(k) if change.is_improvement
        )

    def questions_worsened_at(self, k: int) -> tuple[RankChange, ...]:
        """Return the questions the candidate answers lower within the top `k`.

        Found by the baseline and missed by the candidate counts, and so does any
        move down within `k`.

        Raises:
            ValueError: `k` isn't a positive `int` or is past the judged depth, or the
                cards have no answerable question.
        """
        return tuple(
            change for change in self._rank_changes_at(k) if change.is_regression
        )

    def questions_unchanged_at(self, k: int) -> tuple[RankChange, ...]:
        """Return the questions both runs rank alike within the top `k`.

        The same rank, or missed by both: a question missed by both is listed here,
        so the three lists together hold every answerable question once.

        Raises:
            ValueError: `k` isn't a positive `int` or is past the judged depth, or the
                cards have no answerable question.
        """
        return tuple(
            change
            for change in self._rank_changes_at(k)
            if not (change.is_improvement or change.is_regression)
        )

    def for_tag(self, tag: str) -> Comparison:
        """Return the comparison over only the questions tagged `tag`.

        Every method then works per tag, through each card's `for_tag`.

        Raises:
            ValueError: No question on the cards, scored or left out, has the tag.
        """
        return Comparison(
            candidate=self.candidate.for_tag(tag), baseline=self.baseline.for_tag(tag)
        )

    def _rank_changes_at(self, k: int) -> tuple[RankChange, ...]:
        """Return every answerable question's two ranks within `k`, in question order.

        Raises:
            ValueError: Either card refuses `k`, or has no answerable question.
        """
        candidate_ranks = self.candidate.first_answered_ranks_within(k)
        baseline_ranks = self.baseline.first_answered_ranks_within(k)

        return tuple(
            RankChange(
                question=row.question,
                baseline_rank=baseline_ranks[row.question.id],
                candidate_rank=candidate_ranks[row.question.id],
            )
            for row in self.candidate.answerable_questions
        )
