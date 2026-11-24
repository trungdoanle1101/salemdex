from dataclasses import dataclass
from functools import cached_property

from salemdex_retrieval.checks import check_positive_int
from salemdex_retrieval.engine import Statement
from salemdex_retrieval.evaluation.judges.protocol import Judgment
from salemdex_retrieval.evaluation.questions import Question
from salemdex_retrieval.evaluation.runs import ReturnedChunk


@dataclass(frozen=True, kw_only=True)
class RankedChunk:
    returned: ReturnedChunk
    judgment: Judgment

    @property
    def is_answered(self) -> bool:
        return self.judgment.grade == "answered"


@dataclass(frozen=True, kw_only=True)
class ScoredQuestion:
    question: Question
    chunks: tuple[RankedChunk, ...]

    @property
    def first_answered_rank(self) -> int | None:
        for rank, chunk in enumerate(self.chunks, start=1):
            if chunk.is_answered:
                return rank
        return None


@dataclass(frozen=True, kw_only=True)
class Scorecard:
    statement: Statement
    questions: tuple[ScoredQuestion, ...]
    questions_left_out: tuple[Question, ...]
    judged_depth: int

    def hit_rate_at(self, k: int) -> float:
        self._check_for_rank_metrics(k)
        hits = 0

        for question in self._answerable_questions:
            if question.first_answered_rank is None or question.first_answered_rank > k:
                continue
            hits += 1
        return hits / len(self._answerable_questions)

    def mean_reciprocal_rank_at(self, k: int) -> float:
        self._check_for_rank_metrics(k)
        score = 0.0
        for question in self._answerable_questions:
            if question.first_answered_rank is None or question.first_answered_rank > k:
                continue
            score += 1.0 / question.first_answered_rank
        return score / len(self._answerable_questions)

    @cached_property
    def _answerable_questions(self) -> tuple[ScoredQuestion, ...]:
        return tuple(row for row in self.questions if row.question.is_answerable)

    def _check_for_rank_metrics(self, k: int) -> None:
        check_positive_int(k, name="k")
        if k > self.judged_depth:
            raise ValueError(
                f"k can only be at most {self.judged_depth} (judged_depth), got {k}"
            )
        if not self._answerable_questions:
            raise ValueError("this card has no answerable questions to rank")
