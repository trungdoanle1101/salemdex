"""Search statements, built step by step: `search(index)`, then the stages in order.

A statement describes how to search, for any question: it holds no question and no
limit, which are given each time it runs, so one statement serves every search, and
can be named, logged and compared. It names the one index its candidates come from,
like SQL's FROM, and every scan it runs is registered on that index. Each step
returns a new statement in a new state, and each state offers only the steps allowed
next, so mypy checks the order:

    search(index)               → Question
    Question.retrieve(scan)     → Candidates
    Question.retrieve(a, b, …)  → CandidateLists   (several lists: merge them)
    CandidateLists.merge(…)     → Candidates
    Candidates.rerank(…)        → Candidates
    Candidates.take(…)          → Statement        (complete: what the engine prepares)

Statements are immutable data: a step never changes the statement it starts from, so
one statement can branch into several. A statement built at run time, without these
steps, is checked by `replay`, which takes the steps for it.

Example:
    statement = (
        search("chunks")
        .retrieve(scan("dense", depth=50), scan("sparse", depth=50))
        .merge(Rrf())
        .rerank("cross_encoder")
        .take()
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import overload

from salemdex_retrieval.checks import to_stripped_str
from salemdex_retrieval.engine.stages import (
    Fusion,
    Merge,
    Rerank,
    Retrieve,
    Scan,
    Stage,
    Take,
    Window,
)
from salemdex_retrieval.errors import StatementError


@dataclass(frozen=True, kw_only=True)
class _State:
    """What every state holds: the index searched, and the stages added so far.

    Private, so that no signature can accept any state: only a complete `Statement`
    can be prepared. The states subclass it side by side, never one another, or a
    later state would be accepted where an earlier one is expected.

    Attributes:
        index: The name of the index the candidates come from, as the engine
            registered it; stripped of whitespace at either end.
        stages: The stages so far, in the order they run.
    """

    index: str
    stages: tuple[Stage, ...]

    def __str__(self) -> str:
        """The statement written as the steps that build it, e.g. for a log line.

        Fields print by name, so a window reads `order_by=position`. The dataclass
        `repr` stays the exact form, for debugging.

        Example:
            search('chunks').retrieve(scan('dense', depth=50)).rerank('bge').take()
        """
        steps = (_written_as_step(stage) for stage in self.stages)
        return f"search({self.index!r})" + "".join(steps)

    def _add[S: _State](self, stage: Stage, *, becomes: type[S]) -> S:
        """Return this statement with `stage` added, as the state `becomes`."""
        return becomes(index=self.index, stages=(*self.stages, stage))


def search(index: str) -> Question:
    """Start a statement searching the index registered as `index`.

    Every candidate the statement finds is a chunk of this index, as SQL's FROM names
    the table rows come from, and every scan it runs is registered on it.

    Args:
        index: The index's name in the engine, e.g. "chunks".

    Returns:
        A statement with no stages yet, ready to retrieve.

    Raises:
        TypeError: `index` isn't a `str`.
        ValueError: `index` is empty or only whitespace.
    """
    return Question(index=to_stripped_str(index, name="index"), stages=())


@dataclass(frozen=True, kw_only=True)
class Question(_State):
    """A statement where exactly one question flows, with nothing retrieved yet.

    One question is what makes a single scan give a single list. Query transforms
    (rewrite, split into several questions) will start from this state too.
    """

    @overload
    def retrieve(self, scan: Scan, /) -> Candidates: ...

    @overload
    def retrieve(self, first: Scan, second: Scan, /, *more: Scan) -> CandidateLists: ...

    def retrieve(self, *scans: Scan) -> Candidates | CandidateLists:
        """Return the statement with a stage running `scans`.

        One scan gives one ranked list, ready for the next stages. Several give one
        list each, which must be merged first.

        Args:
            scans: The scans to run, each named as registered on the statement's
                index; at least one, no two with the same name.

        Returns:
            `Candidates` for one scan, `CandidateLists` for several.

        Raises:
            ValueError: There are no scans, or two share a name (see `Retrieve`).
            TypeError: An argument isn't a `Scan`.
        """
        stage = Retrieve(scans=scans)
        if len(scans) == 1:
            return self._add(stage, becomes=Candidates)
        return self._add(stage, becomes=CandidateLists)


@dataclass(frozen=True, kw_only=True)
class CandidateLists(_State):
    """A statement whose candidates are in several ranked lists, one per scan.

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

    def take(self, *, window: Window | None = None) -> Statement:
        """Return the complete statement, ending in a stage keeping the best candidates.

        How many it keeps is given each time the statement runs.

        Args:
            window: The chunks to return around each hit (see `Window`); `None`, the
                default, returns each hit alone.

        Returns:
            The complete statement, ready to be prepared.
        """
        return self._add(Take(window=window), becomes=Statement)


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
        ValueError: The index's name is empty, or a stage's settings are wrong,
            found as its step builds it again (see the steps).
        TypeError: The index's name isn't a `str`, a scan isn't a `Scan`, or a
            reranker's name isn't a `str`.
    """
    state: _State = search(statement.index)
    for position, stage in enumerate(statement.stages):
        match state, stage:
            case Question(), Retrieve(scans=scans):
                state = state.retrieve(*scans)
            case CandidateLists(), Merge(fusion=fusion):
                state = state.merge(fusion)
            case Candidates(), Rerank(reranker=reranker):
                state = state.rerank(reranker)
            case Candidates(), Take(window=window):
                state = state.take(window=window)
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


def _written_as_step(stage: object) -> str:
    """Return `stage` written as the step that adds it, e.g. `.rerank('bge')`."""
    # `object`, not `Stage`: a statement built by hand may hold anything until it's
    # replayed, and printing it must never fail
    match stage:
        case Retrieve(scans=scans):
            written = ", ".join(f"scan({s.name!r}, depth={s.depth})" for s in scans)
            return f".retrieve({written})"
        case Merge(fusion=fusion):
            return f".merge({fusion!r})"
        case Rerank(reranker=reranker):
            return f".rerank({reranker!r})"
        case Take(window=None):
            return ".take()"
        case Take(window=Window() as window):
            return f".take(window={_written_window(window)})"
        case _:
            return f".<{stage!r}>"


def _written_window(window: Window) -> str:
    """Return `window` written as its call, its fields by name."""
    names = [field.name for field in window.partition_by]
    # A one-field tuple keeps its comma, as Python writes it
    partition_by = f"({names[0]},)" if len(names) == 1 else f"({', '.join(names)})"
    return (
        f"Window(partition_by={partition_by}, order_by={window.order_by.name}, "
        f"chunks_each_side={window.chunks_each_side})"
    )
