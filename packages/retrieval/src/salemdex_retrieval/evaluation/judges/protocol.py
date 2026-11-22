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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from salemdex_retrieval.checks import to_stripped_str
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
