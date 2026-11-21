"""Search statements, built step by step: `search()`, then the stages in order.

A statement describes how to search, for any question: it holds no question and no
limit, which are given each time it runs, so one statement serves every search, and
can be named, logged and compared. Each step returns a new statement in a new state,
and each state offers only the steps allowed next, so mypy checks the order:

    search()                    → Question
    Question.retrieve(source)   → Candidates
    Question.retrieve(a, b, …)  → CandidateLists   (several lists: merge them)
    CandidateLists.merge(…)     → Candidates
    Candidates.rerank(…)        → Candidates
    Candidates.take()           → Statement        (complete: what the engine prepares)

Statements are immutable data: a step never changes the statement it starts from, so
one statement can branch into several. A statement built at run time, without these
steps, is checked by `replay`, which takes the steps for it.

Example:
    statement = (
        search()
        .retrieve(source("dense", depth=50), source("sparse", depth=50))
        .merge(Rrf())
        .rerank("cross_encoder")
        .take()
    )
"""

from __future__ import annotations

from dataclasses import dataclass
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
from salemdex_retrieval.errors import StatementError


@dataclass(frozen=True, kw_only=True)
class _State:
    """What every state holds: the stages added so far, in order.

    Private, so that no signature can accept any state: only a complete `Statement`
    can be prepared. The states subclass it side by side, never one another, or a
    later state would be accepted where an earlier one is expected.

    Attributes:
        stages: The stages so far, in the order they run.
    """

    stages: tuple[Stage, ...]

    def _add[S: _State](self, stage: Stage, *, becomes: type[S]) -> S:
        """Return this statement with `stage` added, as the state `becomes`."""
        return becomes(stages=(*self.stages, stage))


def search() -> Question:
    """Start a statement searching for answers.

    Returns:
        A statement with no stages yet, ready to retrieve.
    """
    return Question(stages=())


@dataclass(frozen=True, kw_only=True)
class Question(_State):
    """A statement where exactly one question flows, with nothing retrieved yet.

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
            sources: Where to look, each named as the engine registered it; at least
                one, no two with the same name.

        Returns:
            `Candidates` for one source, `CandidateLists` for several.

        Raises:
            ValueError: There are no sources, or two share a name (see `Retrieve`).
            TypeError: A source isn't a `Source`.
        """
        stage = Retrieve(sources=sources)
        if len(sources) == 1:
            return self._add(stage, becomes=Candidates)
        return self._add(stage, becomes=CandidateLists)


@dataclass(frozen=True, kw_only=True)
class CandidateLists(_State):
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
class Candidates(_State):
    """A statement whose candidates are in one ranked list, ready to refine or take."""

    def rerank(self, reranker: str) -> Candidates:
        """Return the statement with a stage reranking every candidate.

        Args:
            reranker: The name of the reranker, as the engine registered it, scoring
                each candidate against the question.

        Returns:
            The statement with its candidates reordered by the reranker's scores.

        Raises:
            TypeError: `reranker` isn't a `str`.
            ValueError: `reranker` is empty or only whitespace.
        """
        return self._add(Rerank(reranker=reranker), becomes=Candidates)

    def take(self) -> Statement:
        """Return the complete statement, ending in a stage keeping the best candidates.

        How many it keeps is given each time the statement runs.

        Returns:
            The complete statement, ready to be prepared.
        """
        return self._add(Take(), becomes=Statement)


@dataclass(frozen=True, kw_only=True)
class Statement(_State):
    """A complete search statement: the only state the engine prepares.

    It offers no further steps. Preparing it checks it again, through `replay`, so one
    built at run time, without the steps above, is checked the same way.
    """


def replay(statement: Statement) -> Statement:
    """Return `statement` built again through the steps, checking the order of its stages.

    A statement built at run time (from config, by an agent) never went through the
    steps, so mypy never saw its order. Replaying takes the step each stage stands for,
    in order: every stage must be a step the state so far offers, and the last state
    must be a `Statement`. The order rules thus live only in the steps.

    Args:
        statement: The statement to check, however it was built.

    Returns:
        An equal statement, built from checked steps; run this one, not the one given.

    Raises:
        StatementError: A stage isn't one the state before it offers (or isn't a stage
            at all), or the stages stop before a `Take`.
        ValueError: A stage's settings are wrong, found as its step builds it again
            (see the steps).
        TypeError: A source isn't a `Source`, or a reranker's name isn't a `str`.
    """
    state: _State = search()
    for position, stage in enumerate(statement.stages):
        match state, stage:
            case Question(), Retrieve(sources=sources):
                state = state.retrieve(*sources)
            case CandidateLists(), Merge(fusion=fusion):
                state = state.merge(fusion)
            case Candidates(), Rerank(reranker=reranker):
                state = state.rerank(reranker)
            case Candidates(), Take():
                state = state.take()
            case _:
                raise StatementError(
                    f"stages[{position}] is a {type(stage).__name__}, "
                    f"which can't follow {type(state).__name__}"
                )
    if not isinstance(state, Statement):
        raise StatementError(
            f"the statement ends at {type(state).__name__}; end with a Take"
        )
    return state
