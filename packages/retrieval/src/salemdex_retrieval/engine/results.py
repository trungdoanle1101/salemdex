"""What a search produces: passages, each with the trail of how it was ranked.

While a statement runs, the executor moves candidates from stage to stage, and each
stage that ranks a candidate adds an entry to its trail, never changing an earlier
one. A candidate that passes the final gate becomes a passage, which `execute`
returns.

A trail holds source names, ranks and scores, never text, so it can always be logged,
and it answers "why is this passage here?" stage by stage.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from salemdex_retrieval.checks import check_positive_int
from salemdex_retrieval.chunks import Chunk

# Every kind of trail entry. A closed union, like `Stage`: a `match` over it ends in
# `assert_never`, so mypy points at every place that hasn't learned a new kind.
type TrailEntry = Found | Fused | Reranked

# A candidate's entries, in the order they were added. A merged candidate has one
# `Found` for each source that found it, then a `Fused`.
type Trail = tuple[TrailEntry, ...]


@dataclass(frozen=True, kw_only=True)
class Passage:
    """A chunk a search returns, with the trail of how it was ranked.

    Only the executor builds passages, from the candidates that passed the final
    gate. Python can't stop other code from building one, but a passage built
    elsewhere has skipped that gate.

    Attributes:
        chunk: The chunk found; its text is the source's own words.
        trail: Every entry the stages added, in order; at least one.
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
        """The score from the last stage that ranked the passage.

        The reranker's score if there was one, else the fusion's, else its source's:
        comparable only with the other passages of the same search.
        """
        return self.trail[-1].score

    @classmethod
    def from_candidate(cls, candidate: Candidate) -> Passage:
        """Return a passage holding the candidate's chunk and trail.

        It converts, it doesn't check: call it only for a candidate that passed the
        final gate.

        Args:
            candidate: A candidate that passed the final gate.

        Returns:
            The passage, with the candidate's chunk and trail unchanged.
        """
        return cls(chunk=candidate.chunk, trail=candidate.trail)


@dataclass(frozen=True, kw_only=True)
class Candidate:
    """A chunk being ranked while a statement runs, with its trail so far.

    Each stage that ranks it adds an entry with `followed_by`; the executor turns the
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
    """A source found the candidate: its place and score in that source's list.

    Attributes:
        source: The source's name, as given to `Source`.
        rank: Its place in the source's list.
        score: The index's score for the question, e.g. cosine similarity.
    """

    source: str


@dataclass(frozen=True, kw_only=True)
class Fused(_TrailEntry):
    """A merge fused the sources' lists: the candidate's place and score in the result.

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
