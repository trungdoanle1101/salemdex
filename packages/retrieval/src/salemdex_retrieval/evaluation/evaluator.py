"""The evaluator: one question set against one corpus, checked once, then run and graded.

Before any number means anything, question ids must be unique, every quote's `where`
must fit the index's schema, and every quote must be found in the corpus: a quote no
chunk holds would count a broken label as a retrieval miss. The evaluator does all of
it once, when it's built, and every run, judgment and score goes through it, so none
can skip a step or mix question sets.

The class is thin: what a run is, where a quote occurs, and how a chunk is judged live
in the modules below, tested there.
"""

import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import Self

from salemdex_retrieval.checks import check_positive_int
from salemdex_retrieval.chunks import Chunk
from salemdex_retrieval.engine import PreparedStatement
from salemdex_retrieval.evaluation.judges.protocol import Assessment, Judge, Judgment
from salemdex_retrieval.evaluation.occurrences import (
    QuoteOccurrences,
    find_quote_occurrences,
)
from salemdex_retrieval.evaluation.questions import Question
from salemdex_retrieval.evaluation.runs import Run, run_statement
from salemdex_retrieval.evaluation.scorecard import (
    RankedChunk,
    Scorecard,
    ScoredQuestion,
)
from salemdex_retrieval.filters import Everything, Filter
from salemdex_retrieval.index import Index
from salemdex_retrieval.index.protocol import check_filter_matches_schema
from salemdex_retrieval.schema import Schema

# A module-level default: a call in the signature would build one per definition (B008)
_EVERYTHING = Everything()


@dataclass(frozen=True, kw_only=True)
class Evaluator:
    """A question set and the corpus it's graded on, with where each quote was found.

    Build it with `from_index`, which checks the questions and reads the corpus; built
    directly, nothing ties its fields together.

    The corpus is a snapshot, read once: chunks indexed later aren't in it. Nothing
    checks that a statement it runs searches the same index; the caller prepares both
    from one.

    Attributes:
        questions: The questions, in the order given; ids unique.
        corpus: Every chunk of the index, by id, in the order the index gave them; what
            judges read a chunk's text from. Read-only.
        quote_occurrences: Where each quote occurs, one record per quote, in question
            order, then key point, then quote.
        fingerprint: Names this evaluator on every run it makes, so scoring or comparing
            refuses a run from another.

    Example:
        evaluator = Evaluator.from_index(questions, index=index)
        evaluator.quotes_not_found
        run = evaluator.run_statement(engine.prepare(dense_then_rerank))
    """

    questions: tuple[Question, ...]
    corpus: Mapping[uuid.UUID, Chunk]
    quote_occurrences: tuple[QuoteOccurrences, ...]
    fingerprint: uuid.UUID

    @classmethod
    def from_index(cls, questions: Iterable[Question], *, index: Index) -> Self:
        """Return an evaluator for `questions`, with the corpus read from `index`.

        The questions are checked before the index is read; then every chunk is read
        once, and every quote looked for in it.

        Args:
            questions: The question set; ids unique, each quote's `where` over the
                index's schema.
            index: The index holding the corpus, the same one the statements search.

        Returns:
            The evaluator, holding its own copy of the questions and the chunks.

        Raises:
            ValueError: There are no questions, or two share an id.
            SchemaMismatchError: A quote's `where` uses a field that isn't in the
                index's schema.
        """
        questions = tuple(questions)
        _check_questions(questions, schema=index.schema)

        corpus = {chunk.id: chunk for chunk in index.find(where=_EVERYTHING)}
        occurrences = find_quote_occurrences(questions, chunks=corpus.values())

        return cls(
            questions=questions,
            corpus=MappingProxyType(corpus),
            quote_occurrences=tuple(occurrences),
            fingerprint=uuid.uuid4(),
        )

    @cached_property
    def quotes_not_found(self) -> tuple[QuoteOccurrences, ...]:
        """The quotes no chunk in their place holds: labels or chunking to fix."""
        return tuple(found for found in self.quote_occurrences if not found.is_found)

    @cached_property
    def gradable_questions(self) -> tuple[Question, ...]:
        """The questions every metric counts: each key point has a quote found.

        An unanswerable question has no key points, so it's gradable: the abstention
        scores need it, and the rank metrics leave it out by `is_answerable`.
        """
        found = {
            (occurrence.question_id, occurrence.key_point)
            for occurrence in self.quote_occurrences
            if occurrence.is_found
        }
        return tuple(
            question
            for question in self.questions
            if all(
                (question.id, key_point) in found
                for key_point in range(len(question.key_points))
            )
        )

    def judge_top_chunks(
        self, runs: Iterable[Run], *, depth: int, judge: Judge
    ) -> Assessment:
        """Return `judge`'s judgments of every run's top chunks, each judged once.

        For each question, the top `depth` chunks of every run are pooled, in the order
        first seen, and the judge reads each pooled chunk once (pooling, from TREC): runs
        compared on one assessment are graded alike, and a chunk two runs share costs
        one judgment. Every question is judged, gradable or not; scoring decides what to
        leave out.

        `depth` bounds what's judged, and so the cost, and it caps every metric's
        cut-off: the assessment records it as `judged_depth`. Below it, a run's chunk
        may be judged only because another run ranked it higher, so a score read there
        would depend on which runs shared the pool. To score deeper, judge deeper.

        Args:
            runs: The runs to judge, all made by this evaluator; at least one.
            depth: How many of each run's top chunks to judge, per question.
            judge: Grades the chunks: by quotes, an LLM, or people.

        Returns:
            The assessment: one judgment per pooled chunk and question, by `judge`,
            judged to `depth`.

        Raises:
            ValueError: `depth` isn't a positive `int`; there are no runs, or one was
                made by another evaluator; a run returned a chunk that isn't in the
                corpus; or the judge didn't return one judgment, by its name and for
                the question, for each chunk it was given, or named a key point the
                question doesn't have.
        """
        check_positive_int(depth, name="depth")
        runs = tuple(runs)
        self._check_runs(runs)

        judgments: dict[tuple[str, uuid.UUID], Judgment] = {}
        for question in self.questions:
            pool = _pool_top_chunks(question.id, runs=runs, depth=depth)
            judged = judge.judge(
                question, [self._chunk_in_corpus(chunk_id) for chunk_id in pool]
            )
            _check_judged(judged, question=question, pool=pool, judge=judge.name)

            for judgment in judged:
                judgments[(judgment.question_id, judgment.chunk_id)] = judgment

        return Assessment(
            judge=judge.name, judgments=MappingProxyType(judgments), judged_depth=depth
        )

    def run_statement(
        self, prepared: PreparedStatement, *, where: Filter = _EVERYTHING
    ) -> Run:
        """Return what `prepared` returns for every question, ungraded.

        Every question is asked, gradable or not: what's left out is decided when
        scoring, so `quotes_not_found` can change without running again.

        Args:
            prepared: The statement to run, prepared on an engine searching this
                evaluator's index, with `AlwaysValid`.
            where: Which chunks may be found; `Everything()` by default, or one asker's
                access filter to evaluate under it.

        Returns:
            The run, carrying this evaluator's fingerprint.

        Raises:
            SchemaMismatchError: `where` uses a field that isn't in the schema of an
                index the statement searches.
        """
        return run_statement(
            prepared,
            questions=self.questions,
            fingerprint=self.fingerprint,
            where=where,
        )

    def score(self, run: Run, *, assessment: Assessment) -> Scorecard:
        """Return `run`'s scorecard: its top chunks joined with their judgments.

        One row per gradable question, holding its top `judged_depth` chunks; the
        other questions are kept as left out, so the card says what it didn't count.

        Args:
            run: The run to score, made by this evaluator.
            assessment: Judgments of the run's top chunks: `run` was among the runs
                passed to `judge_top_chunks`.

        Returns:
            The scorecard, read to the assessment's judged depth.

        Raises:
            ValueError: The run was made by another evaluator, or one of its chunks
                within the judged depth has no judgment in `assessment`.
        """
        self._check_runs([run])
        rows: list[ScoredQuestion] = []
        left_out: list[Question] = []
        gradable_ids = {question.id for question in self.gradable_questions}
        for question in self.questions:
            if question.id not in gradable_ids:
                left_out.append(question)
                continue

            rows.append(_score_question(question, run=run, assessment=assessment))

        return Scorecard(
            statement=run.statement,
            questions=tuple(rows),
            questions_left_out=tuple(left_out),
            judged_depth=assessment.judged_depth,
        )

    def _check_runs(self, runs: Sequence[Run]) -> None:
        """Check there's a run, and every one was made by this evaluator.

        Raises:
            ValueError: There are no runs, or one was made by another evaluator.
        """
        if not runs:
            raise ValueError("no runs to judge")
        for run in runs:
            if run.evaluator_fingerprint != self.fingerprint:
                raise ValueError(
                    "a run was made by another evaluator: judge and score runs with "
                    "the evaluator that ran them"
                )

    def _chunk_in_corpus(self, chunk_id: uuid.UUID) -> Chunk:
        """Return the corpus's chunk with id `chunk_id`.

        Raises:
            ValueError: No chunk in the corpus has the id: the statement searched
                another index than the one this evaluator read.
        """
        chunk = self.corpus.get(chunk_id)
        if chunk is None:
            raise ValueError(
                f"a run returned chunk {chunk_id}, which isn't in the corpus: was the "
                f"statement prepared on another index than the evaluator read?"
            )
        return chunk


def _check_questions(questions: tuple[Question, ...], *, schema: type[Schema]) -> None:
    """Check the set isn't empty, its ids are unique, and its quotes fit `schema`.

    Raises:
        ValueError: There are no questions, or two share an id.
        SchemaMismatchError: A quote's `where` uses a field that isn't in `schema`.
    """
    if not questions:
        raise ValueError("no questions to evaluate: the question set is empty")

    seen_ids: set[str] = set()
    for question in questions:
        if question.id in seen_ids:
            raise ValueError(f"two questions share the id {question.id!r}")
        seen_ids.add(question.id)

        for key_point in question.key_points:
            for quote in key_point.quotes:
                check_filter_matches_schema(quote.where, schema=schema)


def _pool_top_chunks(
    question_id: str, *, runs: Iterable[Run], depth: int
) -> list[uuid.UUID]:
    """Return the ids of every run's top `depth` chunks for a question, each once."""
    # A dict, not a set: it keeps the order first seen, so a judge reads the pool in
    # rank order and the same runs always give the same pool
    pool = dict.fromkeys(
        returned.chunk_id
        for run in runs
        for returned in run.chunks_by_question[question_id][:depth]
    )
    return list(pool)


def _check_judged(
    judged: Sequence[Judgment],
    *,
    question: Question,
    pool: Sequence[uuid.UUID],
    judge: str,
) -> None:
    """Check the judge returned one judgment, by its name, for each chunk it was given.

    Checked per question, before the next is judged, so a broken judge (an LLM's
    output) fails at once, naming the question.

    Raises:
        ValueError: A judgment is for another question or by another judge, or names
            a key point the question doesn't have; a chunk was judged twice; or the
            chunks judged aren't the chunks given.
    """
    seen: set[uuid.UUID] = set()
    for judgment in judged:
        if judgment.question_id != question.id:
            raise ValueError(
                f"judge {judge!r}, question {question.id!r}: returned a judgment for "
                f"question {judgment.question_id!r}"
            )
        if judgment.judge != judge:
            raise ValueError(
                f"judge {judge!r}, question {question.id!r}: returned a judgment "
                f"named for judge {judgment.judge!r}"
            )
        if judgment.chunk_id in seen:
            raise ValueError(
                f"judge {judge!r}, question {question.id!r}: judged chunk "
                f"{judgment.chunk_id} twice"
            )

        max_key_point_index = max(judgment.key_points, default=-1)
        if max_key_point_index >= len(question.key_points):
            raise ValueError(
                f"judge {judge!r}, question {question.id!r}: returned a judgment "
                f"with a key point index of {max_key_point_index}, "
                f"but the question only has {len(question.key_points)} key points"
            )

        seen.add(judgment.chunk_id)

    if seen != set(pool):
        raise ValueError(
            f"judge {judge!r}, question {question.id!r}: judged "
            f"{len(seen - set(pool))} chunks it wasn't given and left "
            f"{len(set(pool) - seen)} of the {len(pool)} given unjudged"
        )


def _score_question(
    question: Question, *, run: Run, assessment: Assessment
) -> ScoredQuestion:
    """Return the question's row: its top `judged_depth` chunks, each with its judgment.

    Raises:
        ValueError: A chunk within the depth has no judgment: the run wasn't judged in
            this assessment.
    """
    ranked_chunks: list[RankedChunk] = []
    chunks = run.chunks_by_question[question.id][: assessment.judged_depth]
    for returned in chunks:
        judgment = assessment.judgment_of(question.id, returned.chunk_id)
        if judgment is None:
            raise ValueError(
                "this run wasn't judged in this assessment: pass it to judge_top_chunks"
            )
        ranked = RankedChunk(returned=returned, judgment=judgment)
        ranked_chunks.append(ranked)
    return ScoredQuestion(question=question, chunks=tuple(ranked_chunks))
