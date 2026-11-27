"""What a search produces: passages, each with the trail of how it was ranked.

While a statement runs, the engine moves candidates from stage to stage, and each
stage that ranks a candidate adds an entry to its trail, never changing an earlier
one. A candidate that passes the final gate becomes a passage, which
`PreparedStatement.execute` returns.

A trail holds scan names, ranks and scores, never text, so it can always be logged,
and it answers "why is this passage here?" stage by stage.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from salemdex_retrieval.checks import check_positive_int
from salemdex_retrieval.chunks import Chunk

# Every kind of trail entry. A closed union, like `Stage`: a `match` over it ends in
# `assert_never`, so mypy points at every place that hasn't learned a new kind.
type TrailEntry = Found | Fused | Reranked

# A candidate's entries, in the order they were added. A merged candidate has one
# `Found` for each scan that found it, then a `Fused`.
type Trail = tuple[TrailEntry, ...]


@dataclass(frozen=True, kw_only=True)
class Passage:
    """A chunk a search returns, with its window and the trail of how it was ranked.

    Only the engine builds passages, from the candidates that passed the final
    gate. Python can't stop other code from building one, but a passage built
    elsewhere has skipped that gate.

    Attributes:
        chunk: The chunk found, the hit; its text is the source's own words.
        window: The chunks around the hit that the statement's `Window` asks for, in
            its order, with the hit among them exactly once; only the hit when the
            statement has no window. A chunk the search may not show leaves a gap.
        trail: Every entry the stages added, in order; at least one.
    """

    chunk: Chunk
    window: tuple[Chunk, ...]
    trail: Trail

    def __post_init__(self) -> None:
        """Check the window and the trail.

        Raises:
            ValueError: The window holds a chunk twice or doesn't hold the passage's
                own chunk, or the trail is empty.
        """
        ids = [chunk.id for chunk in self.window]
        if len(ids) != len(set(ids)):
            raise ValueError("a window must hold each chunk only once")
        if self.chunk.id not in ids:
            raise ValueError("a passage's window must hold its own chunk")
        if not self.trail:
            raise ValueError("a passage needs at least one trail entry")

    @property
    def score(self) -> float:
        """The score from the last stage that ranked the passage.

        The reranker's score if there was one, else the fusion's, else its scan's:
        comparable only with the other passages of the same search.
        """
        return self.trail[-1].score

    @classmethod
    def from_candidate(
        cls, candidate: Candidate, *, window: Iterable[Chunk]
    ) -> Passage:
        """Return a passage holding the candidate's chunk and trail, and its window.

        It converts, it doesn't check the gate: call it only for a candidate that
        passed it.

        Args:
            candidate: A candidate that passed the final gate.
            window: The chunks around the candidate's chunk, in order, that chunk
                among them.

        Returns:
            The passage, with the candidate's chunk and trail unchanged.

        Raises:
            ValueError: The window holds a chunk twice or doesn't hold the candidate's
                chunk.
        """
        return cls(chunk=candidate.chunk, window=tuple(window), trail=candidate.trail)


@dataclass(frozen=True, kw_only=True)
class Candidate:
    """A chunk being ranked while a statement runs, with its trail so far.

    Each stage that ranks it adds an entry with `followed_by`; the engine turns the
    candidates that pass the final gate into passages with `Passage.from_candidate`.

    Attributes:
        chunk: The chunk found.
        trail: The entries so far, in order; at least one.
    """

    chunk: Chunk
    trail: Trail

    def __post_init__(self) -> None:
        """Check the trail.

        Raises:
            ValueError: The trail is empty.
        """
        if not self.trail:
            raise ValueError("a passage needs at least one trail entry")

    @property
    def score(self) -> float:
        """The score from the last stage that ranked the candidate.

        The next stage starts from it: a rerank scores the best candidates by it.
        """
        return self.trail[-1].score

    def followed_by(self, entry: TrailEntry) -> Candidate:
        """Return this candidate with `entry` added to the end of its trail.

        Args:
            entry: What the latest stage made of the candidate.

        Returns:
            A new candidate; this one is unchanged.
        """
        return replace(self, trail=(*self.trail, entry))


@dataclass(frozen=True, kw_only=True)
class _TrailEntry:
    """What every trail entry holds: a candidate's place and score in one ranking.

    Private, so that signatures name the closed union `TrailEntry` instead. A
    subclass that defines `__post_init__` must call `super().__post_init__()`, or it
    skips the rank check.

    Attributes:
        rank: The candidate's place in the ranking, from 1.
        score: The candidate's score in that ranking; comparable only within it.
    """

    rank: int
    score: float

    def __post_init__(self) -> None:
        """Check the rank.

        Raises:
            ValueError: `rank` isn't a positive `int`.
        """
        check_positive_int(self.rank, name="rank")


@dataclass(frozen=True, kw_only=True)
class Found(_TrailEntry):
    """A scan found the candidate: its place and score in that scan's list.

    Attributes:
        scan: The scan's name, as given to `scan`.
        rank: Its place in the scan's list.
        score: The index's score for the question, e.g. cosine similarity.
    """

    scan: str


@dataclass(frozen=True, kw_only=True)
class Fused(_TrailEntry):
    """A merge fused the scans' lists: the candidate's place and score in the result.

    Attributes:
        rank: Its place in the fused list.
        score: The fusion's score, e.g. RRF's `Σ 1 / (k + rank)`.
    """


@dataclass(frozen=True, kw_only=True)
class Reranked(_TrailEntry):
    """A reranker scored the candidate against the question.

    Attributes:
        rank: Its place among the candidates reranked.
        score: The reranker's score.
    """
