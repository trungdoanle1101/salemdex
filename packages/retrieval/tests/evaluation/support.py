"""Builders for scorecards written by hand, shared by the scorecard and comparison tests.

Rows are built directly, without an evaluator, so each test says exactly where its
answers rank. Invented text only: the real question sets are restricted and never
appear in tests.
"""

import uuid

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


def ranked(
    question_id: str, *, grade: Grade, key_points: tuple[int, ...], rank: int
) -> RankedChunk:
    """Return a chunk at `rank`, judged for the question with `grade`."""
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
    """Return a card of these rows, judged to `JUDGED_DEPTH`."""
    return Scorecard(
        statement=STATEMENT,
        questions=questions,
        questions_left_out=left_out,
        judged_depth=JUDGED_DEPTH,
    )
