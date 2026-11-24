"""What a judge says about a chunk, and the protocol every judge follows.

A judge reads one question and the chunks a run returned for it, and grades each chunk:
does it answer the question, and which of the question's key points does it supply? A
run records what was returned and a judgment what is relevant, kept apart (TREC's runs
and qrels), so a slow or costly judge (an LLM) judges once, and every metric and
comparison reads the same judgments.

A grade is about one chunk. Whether a question's every key point was found is decided
across its top chunks, by the scorecard, never by a judge.
"""

import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from salemdex_retrieval.checks import check_positive_int, to_stripped_str
from salemdex_retrieval.chunks import Chunk
from salemdex_retrieval.evaluation.questions import Question

# How relevant a chunk is to a question. "related" is on topic without giving the answer:
# only a judge that reads for meaning (an LLM) gives it; the quote judge never does.
type Grade = Literal["unrelated", "related", "answered"]


@dataclass(frozen=True, kw_only=True)
class Judgment:
    """A judge's grade for one chunk against one question.

    It holds ids and a grade, never text, so judgments can be logged and compared.

    Attributes:
        question_id: The question's id, as `Question.id` gives it; stripped of whitespace
            at either end.
        chunk_id: The chunk's id, as `Chunk.id` gives it.
        grade: How relevant the chunk is to the question.
        key_points: The positions, in the question's `key_points`, of those the chunk
            supplies; several when one chunk holds several (a rule and its exception).
            Only an "answered" judgment may name any, and an "answered" one may name
            none: a judge reading for meaning can say a chunk answers without saying
            which point it supplies.
        judge: The name of the judge that gave the grade, e.g. "quote"; stripped of
            whitespace at either end. Grades from two judges aren't comparable, so
            whoever combines judgments reads it.

    Example:
        Judgment(
            question_id="q-01",
            chunk_id=chunk.id,
            grade="answered",
            key_points=(0, 1),
            judge="quote",
        )
    """

    question_id: str
    chunk_id: uuid.UUID
    grade: Grade
    key_points: tuple[int, ...]
    judge: str

    def __post_init__(self) -> None:
        """Check the key points against the grade, and strip the question id and judge.

        Raises:
            TypeError: A key point isn't an `int` (a `bool` isn't one either), or the
                question id or the judge isn't a `str`.
            ValueError: A key point is negative or named twice, a grade other than
                "answered" names key points, or the question id or the judge is empty
                or only whitespace.
        """
        seen: set[int] = set()
        for key_point in self.key_points:
            if type(key_point) is not int:
                raise TypeError(f"key points must be integers, got {self.key_points!r}")
            if key_point < 0:
                raise ValueError(
                    f"key points must be non-negative integers, got {self.key_points!r}"
                )
            if key_point in seen:
                raise ValueError(f"duplicate key point: {key_point}")
            seen.add(key_point)

        if self.grade != "answered" and self.key_points:
            raise ValueError(
                "only a judgment of grade 'answered' may contain key points"
            )

        # The dataclass is frozen; this is the one place its fields are set
        object.__setattr__(
            self, "question_id", to_stripped_str(self.question_id, name="question_id")
        )
        object.__setattr__(self, "judge", to_stripped_str(self.judge, name="judge"))


@dataclass(frozen=True, kw_only=True)
class Assessment:
    """One judge's judgments of the chunks runs returned, by question and chunk.

    Every metric and comparison reads one assessment, so runs scored against it share
    the same grades. It holds one judge's grades only: grades from two judges aren't
    comparable, and combining judges is a new judge with its own name.

    Attributes:
        judge: The name of the judge every judgment came from, e.g. "quote"; stripped
            of whitespace at either end.
        judgments: Each judgment, keyed by its question id and chunk id. Read-only
            when built by the evaluator.
        judged_depth: How many of each judged run's top chunks, per question, have a
            judgment. A metric's cut-off can't exceed it: within it, every chunk of a
            judged run is judged whatever other runs shared the pool, so pooling
            never changes a run's score.

    Example:
        assessment = evaluator.judge_top_chunks(runs, depth=10, judge=QuoteJudge())
        judgment = assessment.judgment_of("kt-07", chunk_id)
    """

    judge: str
    judgments: Mapping[tuple[str, uuid.UUID], Judgment]
    judged_depth: int

    def __post_init__(self) -> None:
        """Strip the judge's name, and check the depth and every judgment's key and judge.

        Raises:
            TypeError: The judge's name isn't a `str`.
            ValueError: The judge's name is empty or only whitespace, the judged depth
                isn't a positive `int`, a judgment came from another judge, or a
                judgment is keyed by another question or chunk than its own.
        """
        # The dataclass is frozen; this is the one place its judge is set
        object.__setattr__(self, "judge", to_stripped_str(self.judge, name="judge"))
        check_positive_int(self.judged_depth, name="judged_depth")

        for key, judgment in self.judgments.items():
            if judgment.judge != self.judge:
                raise ValueError(
                    f"question {judgment.question_id!r}, chunk {judgment.chunk_id}: "
                    f"judged by {judgment.judge!r}, in an assessment by {self.judge!r}"
                )
            if key != (judgment.question_id, judgment.chunk_id):
                raise ValueError(
                    f"the judgment of question {judgment.question_id!r}, chunk "
                    f"{judgment.chunk_id} is filed under {key!r}"
                )

    def judgment_of(self, question_id: str, chunk_id: uuid.UUID) -> Judgment | None:
        """Return the judgment of `chunk_id` for `question_id`, or None if never judged.

        None isn't "unrelated": the chunk was never read, e.g. it ranked below the
        judged depth, or its run wasn't judged in this assessment. Metrics read only
        within the judged depth of a judged run, where every chunk has a judgment.

        Args:
            question_id: The question's id, as `Question.id` gives it.
            chunk_id: The chunk's id, as `Chunk.id` gives it.

        Returns:
            The judgment, or None when the judge never read the chunk for the question.
        """
        return self.judgments.get((question_id, chunk_id))


class Judge(Protocol):
    """Grades chunks against a question: by its quotes, by an LLM, or by people.

    A judge works on one question at a time, so each can batch its work as it likes (an
    LLM judge, one prompt per chunk or per question). It trusts what it's given: the
    evaluator has checked each quote's filter against the chunks' schema.
    """

    @property
    def name(self) -> str:
        """The judge's name, written on every judgment it gives, e.g. "quote"."""
        ...

    def judge(self, question: Question, chunks: Iterable[Chunk]) -> Sequence[Judgment]:
        """Return one judgment for each chunk, in the order given.

        Every chunk is judged, an irrelevant one with "unrelated": a chunk with no
        judgment is one the judge never read, which isn't the same.

        Args:
            question: The question the chunks were returned for.
            chunks: The chunks to grade, e.g. a run's top passages for the question.

        Returns:
            The judgments, one per chunk, all naming `question` and this judge.
        """
        ...
