"""The metrics over one run and its judgments, question by question.

`Evaluator.score` joins a run, an assessment and the questions once, into one row per
gradable question, so every metric reads the rows and nothing else. Rows hold only the
top `judged_depth` chunks of each question, every one judged, and no metric's cut-off
may pass that depth: below it, a chunk may be judged only because another run ranked
it higher, so a score read there would depend on which runs were judged together.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from salemdex_retrieval.checks import check_positive_int
from salemdex_retrieval.engine import Statement
from salemdex_retrieval.evaluation.judges.protocol import Judgment
from salemdex_retrieval.evaluation.questions import Question
from salemdex_retrieval.evaluation.runs import ReturnedChunk


@dataclass(frozen=True, kw_only=True)
class RankedChunk:
    """A chunk a run returned for a question, with its judgment for that question.

    Its rank is its position in its row's `chunks`, counted from 1.

    Attributes:
        returned: The chunk as the run returned it: its id and trail.
        judgment: The assessment's judgment of the chunk for the row's question.
    """

    returned: ReturnedChunk
    judgment: Judgment

    @property
    def is_answered(self) -> bool:
        """Whether the chunk was graded as supplying a key point."""
        return self.judgment.grade == "answered"


@dataclass(frozen=True, kw_only=True)
class ScoredQuestion:
    """One question of a scorecard: the question and its top chunks, judged.

    Attributes:
        question: The question, whole.
        chunks: The run's chunks for it, best first, at most the judged depth.
    """

    question: Question
    chunks: tuple[RankedChunk, ...]

    @cached_property
    def first_answered_rank(self) -> int | None:
        """The rank of the first answered chunk, from 1; None if none is answered."""
        for rank, chunk in enumerate(self.chunks, start=1):
            if chunk.is_answered:
                return rank
        return None

    @cached_property
    def all_key_points_rank(self) -> int | None:
        """The rank at which the answered chunks so far supply every key point.

        None if they never do. Read only from the key points each judgment names, so a
        judge that names none never completes a question.
        """
        seen_key_points: set[int] = set()
        all_key_points = set(range(len(self.question.key_points)))
        for rank, chunk in enumerate(self.chunks, start=1):
            if not chunk.is_answered:
                continue
            seen_key_points.update(chunk.judgment.key_points)
            if seen_key_points >= all_key_points:
                return rank

        return None

    def is_answered_within(self, k: int) -> bool:
        """Return whether an answered chunk is among the top `k`."""
        return self.first_answered_rank is not None and self.first_answered_rank <= k

    def has_all_key_points_within(self, k: int) -> bool:
        """Return whether the top `k` chunks together supply every key point."""
        return self.all_key_points_rank is not None and self.all_key_points_rank <= k


@dataclass(frozen=True, kw_only=True)
class Scorecard:
    """One run's top chunks joined with their judgments, with the metrics over them.

    Build it with `Evaluator.score`. Every metric with a cut-off `k` counts only the
    answerable questions, refuses a `k` past `judged_depth`, and refuses a card with no
    answerable question rather than returning 0. A card of unanswerable questions, or
    of none, can still be built.

    Attributes:
        statement: The statement the run ran.
        questions: One row per gradable question, in question order; unanswerable
            questions included.
        questions_left_out: The questions with a key point whose quotes no chunk
            holds, in question order: no metric counts them.
        judged_depth: How deep each question's chunks were judged; every cut-off is
            at most this.

    Example:
        scorecard = evaluator.score(run, assessment=assessment)
        scorecard.hit_rate_at(5)
        scorecard.for_tag("paraphrase").questions_missed_at(10)
    """

    statement: Statement
    questions: tuple[ScoredQuestion, ...]
    questions_left_out: tuple[Question, ...]
    judged_depth: int

    def hit_rate_at(self, k: int) -> float:
        """Return the share of answerable questions answered within the top `k`.

        Raises:
            ValueError: `k` isn't a positive `int` or is past the judged depth, or the
                card has no answerable question.
        """
        self._check_for_rank_metrics(k)
        return sum(
            question.is_answered_within(k) for question in self._answerable_questions
        ) / len(self._answerable_questions)

    def all_key_points_rate_at(self, k: int) -> float:
        """Return the share of answerable questions with every key point in the top `k`.

        Stricter than the hit rate: a question needing three key points counts only
        once all three are in, from one chunk or several.

        Raises:
            ValueError: `k` isn't a positive `int` or is past the judged depth, the card
                has no answerable question, or a chunk graded "answered" names no key
                points: its judge doesn't link them, so the rate can't be read.
        """
        self._check_for_rank_metrics(k)
        self._check_key_points_named()
        return sum(
            question.has_all_key_points_within(k)
            for question in self._answerable_questions
        ) / len(self._answerable_questions)

    def mean_reciprocal_rank_at(self, k: int) -> float:
        """Return the mean of 1 / the first answered rank, over answerable questions.

        A question with no answered chunk in the top `k` counts 0.

        Raises:
            ValueError: `k` isn't a positive `int` or is past the judged depth, or the
                card has no answerable question.
        """
        self._check_for_rank_metrics(k)
        score = 0.0
        for question in self._answerable_questions:
            if (
                question.first_answered_rank is not None
                and question.first_answered_rank <= k
            ):
                score += 1.0 / question.first_answered_rank

        return score / len(self._answerable_questions)

    def questions_missed_at(self, k: int) -> tuple[ScoredQuestion, ...]:
        """Return the answerable rows with no answered chunk in the top `k`.

        The rows themselves, in question order, so a miss can be looked into: its
        chunks, their judgments and their trails.

        Raises:
            ValueError: `k` isn't a positive `int` or is past the judged depth, or the
                card has no answerable question.
        """
        self._check_for_rank_metrics(k)
        return tuple(
            question
            for question in self._answerable_questions
            if not question.is_answered_within(k)
        )

    def for_tag(self, tag: str) -> Scorecard:
        """Return the card over only the questions tagged `tag`, left out ones too.

        Every metric then works per tag. A tag whose questions were all left out gives
        a card with no rows, which shows its labels need fixing.

        Raises:
            ValueError: No question on the card, scored or left out, has the tag.
        """
        questions = tuple(
            scored for scored in self.questions if tag in scored.question.tags
        )
        left_out = tuple(
            question for question in self.questions_left_out if tag in question.tags
        )
        if not questions and not left_out:
            raise ValueError(
                f"no question on this card has the tag {tag!r}: "
                f"its tags are {self.tags}"
            )

        return Scorecard(
            statement=self.statement,
            questions=questions,
            questions_left_out=left_out,
            judged_depth=self.judged_depth,
        )

    @cached_property
    def tags(self) -> tuple[str, ...]:
        """Every tag on the card's questions, scored or left out, sorted, each once."""
        questions = [row.question for row in self.questions]
        questions.extend(self.questions_left_out)
        return tuple(sorted({tag for question in questions for tag in question.tags}))

    @cached_property
    def _answerable_questions(self) -> tuple[ScoredQuestion, ...]:
        """The rows the metrics with a cut-off count."""
        return tuple(row for row in self.questions if row.question.is_answerable)

    def _check_for_rank_metrics(self, k: int) -> None:
        """Check `k` is a cut-off this card can be read at, and it has a row to count.

        Raises:
            ValueError: `k` isn't a positive `int` or is past the judged depth, or the
                card has no answerable question.
        """
        check_positive_int(k, name="k")
        if k > self.judged_depth:
            raise ValueError(
                f"k can only be at most {self.judged_depth} (judged_depth), got {k}"
            )
        if not self._answerable_questions:
            raise ValueError("this card has no answerable questions to rank")

    def _check_key_points_named(self) -> None:
        """Check every answered chunk names its key points, on the whole card.

        The whole card, not the top `k`: whether a judge names key points is the
        judge's way, so the rate is readable at every `k` or at none.

        Raises:
            ValueError: An answered chunk of an answerable question names none.
        """
        for row in self._answerable_questions:
            for chunk in row.chunks:
                if chunk.is_answered and not chunk.judgment.key_points:
                    raise ValueError(
                        f"question {row.question.id!r}: judge {chunk.judgment.judge!r} "
                        f"graded a chunk 'answered' without naming its key points"
                    )
