"""What a statement returned for each question, recorded before any grading.

A run is one prepared statement executed once per question, with what came back kept
as chunk ids and trails. It says nothing about whether a chunk answers: judges decide
that afterwards, and metrics read the run and the judgments together (TREC's runs and
qrels). So a judge added later can grade an old run, and a slow judge grades each chunk
once for every metric and comparison.

A run holds every candidate the statement found, not only the top few: each metric
takes its own cut-off, and per-stage metrics need the chunks a rerank left low. It holds
no text, so it can be logged and diffed; judges read a chunk's text by its id from the
corpus.
"""

import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import assert_never

from salemdex_retrieval.engine import PreparedStatement, Statement
from salemdex_retrieval.engine.results import Trail
from salemdex_retrieval.engine.stages import Merge, Rerank, Retrieve, Take
from salemdex_retrieval.evaluation.questions import Question
from salemdex_retrieval.filters import Filter


@dataclass(frozen=True, kw_only=True)
class ReturnedChunk:
    """A chunk a statement returned, held by its id, with the trail of how it ranked.

    The engine's `Passage` without its chunk: the id is enough to match a judgment or
    read the text from the corpus, and leaving the chunk out keeps text out of the run.
    Its rank is its position in the question's tuple.

    Attributes:
        chunk_id: The chunk's id, as `Chunk.id` gives it.
        trail: Every entry the stages added, in order: which source found it at what
            rank, then its place after fusion and after the rerank.
    """

    chunk_id: uuid.UUID
    trail: Trail


@dataclass(frozen=True, kw_only=True)
class Run:
    """What one statement returned for each question of an evaluation.

    Attributes:
        statement: The statement that ran, naming its stages, roles and depths; it tells
            two runs apart in a comparison.
        chunks_by_question: Each question's id, mapped to the chunks returned for it,
            best first: every candidate the statement found. Read-only.
        evaluator_fingerprint: Names the evaluator whose questions and corpus the run
            used, so scoring or comparing refuses a run from another.

    Example:
        for returned in run.chunks_by_question["kt-07"]:
            judgments.is_answered("kt-07", returned.chunk_id)
    """

    statement: Statement
    chunks_by_question: Mapping[str, tuple[ReturnedChunk, ...]]
    evaluator_fingerprint: uuid.UUID


def run_statement(
    prepared: PreparedStatement,
    *,
    questions: Iterable[Question],
    fingerprint: uuid.UUID,
    where: Filter,
) -> Run:
    """Return what `prepared` returns for each of `questions`, ungraded.

    Each question is asked once, for every candidate the statement can find: the sum of
    its sources' depths. Unanswerable questions are asked like the others; their top
    scores serve abstention. It trusts the evaluator: question ids are unique (a second
    question with an id would replace the first's chunks), and the engine keeps every
    chunk (`AlwaysValid`), so no gate hides one.

    Args:
        prepared: The statement to run, prepared by the engine.
        questions: The questions to ask, in the order the run keeps them.
        fingerprint: The evaluator's fingerprint, recorded on the run.
        where: Which chunks may be found, e.g. `Everything()`, or one asker's access
            filter to evaluate under it.

    Returns:
        The run: the statement, each question's returned chunks, and the fingerprint.

    Raises:
        SchemaMismatchError: `where` uses a field that isn't in the schema of an index
            the statement searches.

    Example:
        run = run_statement(
            engine.prepare(dense_then_rerank),
            questions=questions,
            fingerprint=fingerprint,
            where=Everything(),
        )
    """
    every_candidate = _count_candidates(prepared.statement)
    record: dict[str, tuple[ReturnedChunk, ...]] = {}
    for question in questions:
        passages = prepared.execute(
            question=question.text, limit=every_candidate, where=where
        )
        record[question.id] = tuple(
            ReturnedChunk(chunk_id=p.chunk.id, trail=p.trail) for p in passages
        )
    return Run(
        statement=prepared.statement,
        chunks_by_question=MappingProxyType(record),
        evaluator_fingerprint=fingerprint,
    )


def _count_candidates(statement: Statement) -> int:
    """Return the most candidates `statement` can find: its sources' depths, summed."""
    count = 0
    for stage in statement.stages:
        match stage:
            case Retrieve(sources=sources):
                # A merge's union can exceed any one depth, never their sum
                count += sum(source.depth for source in sources)
            case Merge() | Rerank() | Take():
                continue
            case _:
                assert_never(stage)

    return count
