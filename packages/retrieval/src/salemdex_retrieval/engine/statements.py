"""Search statements, built step by step: `search(question)`, then the stages in order.

A statement describes a search; the session's executor runs it. Each step returns a
new statement in a new state, and each state offers only the steps allowed next, so
mypy checks the order:

    search(question)            → Question
    Question.retrieve(source)   → Candidates
    Question.retrieve(a, b, …)  → CandidateLists   (several lists: merge them)
    CandidateLists.merge(…)     → Candidates
    Candidates.rerank(…)        → Candidates
    Candidates.take(limit)      → Search           (complete: what `execute` takes)

Statements are immutable data: a step never changes the statement it starts from, so
one statement can branch into several. A statement built at run time, without these
steps, is checked by the executor instead.

Example:
    statement = (
        search("thời hạn nộp hồ sơ")
        .retrieve(dense, sparse)
        .merge(Rrf())
        .rerank(cross_encoder, depth=30)
        .take(10)
    )
    passages = session.execute(statement)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import overload

from salemdex_retrieval.engine.stages import (
    Fusion,
    Merge,
    Rerank,
    Retrieve,
    Source,
    Stage,
    Take,
)
from salemdex_retrieval.rerankers.protocol import Reranker


@dataclass(frozen=True, kw_only=True)
class _Statement:
    """What every state holds: the question, and the stages added so far, in order.

    Private, so that no signature can accept any state: `execute` takes a `Search`.
    The states subclass it side by side, never one another, or a later state would be
    accepted where an earlier one is expected.

    The question stays out of `repr`: it can reveal what the asker is after, and a
    statement in a traceback or a log line would print it. An app that audits
    questions reads `question` itself and stores it where its own rules apply.

    Attributes:
        question: The question as asked.
        stages: The stages so far, in the order they run.
    """

    question: str = field(repr=False)
    stages: tuple[Stage, ...]

    def _add[S: _Statement](self, stage: Stage, *, becomes: type[S]) -> S:
        """Return this statement with `stage` added, as the state `becomes`."""
        return becomes(question=self.question, stages=(*self.stages, stage))


def search(question: str) -> Question:
    """Start a statement searching for answers to `question`.

    Args:
        question: The question as asked; it's embedded for each source, and judged
            against each candidate by a reranker.

    Returns:
        A statement with no stages yet, ready to retrieve.

    Raises:
        ValueError: `question` is empty or only whitespace.
    """
    # An empty question would still be embedded and reranked, giving confident
    # scores for nothing
    if not question.strip():
        raise ValueError("question is empty or only whitespace; ask something")
    return Question(question=question, stages=())


@dataclass(frozen=True, kw_only=True)
class Question(_Statement):
    """A statement holding exactly one question, with nothing retrieved yet.

    One question is what makes a single source give a single list. Query transforms
    (rewrite, split into several questions) will start from this state too.
    """

    @overload
    def retrieve(self, source: Source, /) -> Candidates: ...

    @overload
    def retrieve(
        self, first: Source, second: Source, /, *more: Source
    ) -> CandidateLists: ...

    def retrieve(self, *sources: Source) -> Candidates | CandidateLists:
        """Return the statement with a stage finding candidates in `sources`.

        One source gives one ranked list, ready for the next stages. Several give one
        list each, which must be merged first.

        Args:
            sources: Where to look; at least one, no two searching the same vector of
                the same index.

        Returns:
            `Candidates` for one source, `CandidateLists` for several.

        Raises:
            ValueError: There are no sources, or two repeat each other (see
                `Retrieve`).
            TypeError: A source isn't a `Source`.
        """
        stage = Retrieve(sources=sources)
        if len(sources) == 1:
            return self._add(stage, becomes=Candidates)
        return self._add(stage, becomes=CandidateLists)


@dataclass(frozen=True, kw_only=True)
class CandidateLists(_Statement):
    """A statement whose candidates are in several ranked lists, one per source.

    Nothing but a merge can follow: every later stage needs a single ranking.
    """

    def merge(self, fusion: Fusion) -> Candidates:
        """Return the statement with a stage fusing the lists into one.

        Args:
            fusion: How the lists are fused, e.g. `Rrf()`.

        Returns:
            The statement with one ranked list of candidates.
        """
        return self._add(Merge(fusion=fusion), becomes=Candidates)


@dataclass(frozen=True, kw_only=True)
class Candidates(_Statement):
    """A statement whose candidates are in one ranked list, ready to refine or take."""

    def rerank(self, reranker: Reranker, *, depth: int) -> Candidates:
        """Return the statement with a stage reranking the top candidates.

        Args:
            reranker: The model scoring each candidate against the question.
            depth: How many of the top candidates it scores; the rest are dropped.

        Returns:
            The statement with its candidates reordered by the reranker's scores.

        Raises:
            ValueError: `depth` isn't a positive `int`.
        """
        return self._add(Rerank(reranker=reranker, depth=depth), becomes=Candidates)

    def take(self, limit: int) -> Search:
        """Return the complete statement, keeping the best `limit` candidates.

        Args:
            limit: The most passages the search returns.

        Returns:
            The complete statement, ready for `execute`.

        Raises:
            ValueError: `limit` isn't a positive `int`.
        """
        return self._add(Take(limit=limit), becomes=Search)


@dataclass(frozen=True, kw_only=True)
class Search(_Statement):
    """A complete search statement: the only state a session executes.

    It offers no further steps. The executor still checks its stages, in case it was
    built at run time without the steps above.
    """
