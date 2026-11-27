"""The steps a search statement is made of, as plain data: what to do, never with what.

Each stage holds only its own settings, and names the scans and rerankers it uses;
the engine holds what those names stand for (embedders, parsers, models), and gives
the stages meaning when a prepared statement runs them. Holding no objects, a statement
can be logged, compared and replayed, and one built at run time (from config, by an
agent) can name only what the app registered.

A stage checks its own settings when it's built. Checks that need the engine (a scan
is registered on the statement's index, a window's fields are in its schema) run when
the statement is prepared, so a wrong statement still fails before any model is
called or any index is read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from salemdex_retrieval.checks import check_positive_int, to_stripped_str
from salemdex_retrieval.schema import Field, One

# Every kind of stage. A closed union: the engine's `match` over it ends in
# `assert_never`, so mypy points at every place that hasn't learned a new kind.
type Stage = Retrieve | Merge | Rerank | Take

# Every way of fusing ranked lists, closed like `Stage`.
type Fusion = Rrf


@dataclass(frozen=True, kw_only=True)
class Retrieve:
    """A stage finding candidates for the question, one ranked list per scan.

    Each scan reads the statement's index for the question, with the run's filter
    (`where`). Several scans give several lists, which a `Merge` must fuse before any
    other stage. Every scan reads the same index, so a candidate is known by its chunk
    id, and one filter fits every scan.

    No two scans may share a name: each candidate's trail names the scans that found
    it. That is checked here; two rules need the registered scans, so they're checked
    when the statement is prepared:

    - every scan is registered on the statement's index;
    - no two scans search the same vector, whatever their depths: the shallower list
      would be the start of the deeper one, and fusion would count it twice.

    Attributes:
        scans: What to run, in order; at least one.
    """

    scans: tuple[Scan, ...]

    def __post_init__(self) -> None:
        """Check the scans.

        Raises:
            ValueError: There are no scans, or two share a name.
            TypeError: A scan isn't a `Scan`.
        """
        if not self.scans:
            raise ValueError("a retrieve needs at least one scan")

        seen_scans: set[str] = set()
        for position, scan in enumerate(self.scans):
            if type(scan) is not Scan:
                raise TypeError(
                    f"scans[{position}] is {type(scan).__name__}; expected a Scan"
                )

            if scan.name in seen_scans:
                raise ValueError(
                    f"two scans in one retrieve share a name: {scan.name!r}"
                )
            seen_scans.add(scan.name)


@dataclass(frozen=True, kw_only=True)
class Scan:
    """A scan a retrieve runs, named as the engine registered it, and how deep.

    Build it with `scan`. It holds only data; when the statement is prepared, the
    engine resolves the name, on the statement's index, to what the scan reads it
    with (e.g. a `VectorScan`).

    Attributes:
        name: The scan's name on the index, also what each candidate's trail calls
            it, e.g. "dense"; unique within a retrieve, and stripped of whitespace at
            either end.
        depth: The most candidates this scan returns.
    """

    name: str
    depth: int

    def __post_init__(self) -> None:
        """Strip the name, and check it and the depth.

        Raises:
            TypeError: `name` isn't a `str`.
            ValueError: `name` is empty or only whitespace, or `depth` isn't a
                positive `int`.
        """
        # The dataclass is frozen; this is the one place its name is set
        object.__setattr__(self, "name", to_stripped_str(self.name, name="name"))
        check_positive_int(self.depth, name="depth")


def scan(name: str, *, depth: int) -> Scan:
    """Return the scan registered as `name`, run for at most `depth` candidates.

    Args:
        name: The scan's name on the statement's index, e.g. "dense".
        depth: The most candidates it returns.

    Returns:
        The scan, ready for a retrieve.

    Raises:
        TypeError: `name` isn't a `str`.
        ValueError: `name` is empty or only whitespace, or `depth` isn't a positive
            `int`.

    Example:
        search("chunks").retrieve(scan("dense", depth=50), scan("sparse", depth=50))
    """
    return Scan(name=name, depth=depth)


@dataclass(frozen=True, kw_only=True)
class Merge:
    """A stage fusing the ranked lists of several scans into one.

    Attributes:
        fusion: How the lists are fused, e.g. `Rrf()`.
    """

    fusion: Fusion


@dataclass(frozen=True, kw_only=True)
class Rrf:
    """Reciprocal rank fusion: each candidate scores `Σ 1 / (k + rank)` over its lists.

    Only ranks count, never the scans' scores, which aren't on one scale (cosine,
    BM25 and max-sim differ). A candidate several scans rank high beats one a single
    scan ranks first.

    Attributes:
        k: How much a top rank outweighs a lower one; smaller favours the top. 60 is
            the usual value (Cormack et al., 2009).
    """

    k: int = 60

    def __post_init__(self) -> None:
        """Check `k`.

        Raises:
            ValueError: `k` isn't a positive `int`.
        """
        check_positive_int(self.k, name="k")


@dataclass(frozen=True, kw_only=True)
class Rerank:
    """A stage scoring every candidate again with a reranker, and reordering them.

    It scores all the candidates it receives, so its cost is set by the stages before
    it: each bound in a statement limits what a stage keeps, never what it reads.
    With one scan that's the scan's `depth`; after a merge, the union of the scans'
    lists.

    Attributes:
        reranker: The name of the reranker, as the engine registered it, that scores
            each candidate against the question; stripped of whitespace at either end.
    """

    reranker: str

    def __post_init__(self) -> None:
        """Strip the reranker's name, and check it.

        Raises:
            TypeError: `reranker` isn't a `str`.
            ValueError: `reranker` is empty or only whitespace.
        """
        # The dataclass is frozen; this is the one place its name is set
        object.__setattr__(
            self, "reranker", to_stripped_str(self.reranker, name="reranker")
        )


@dataclass(frozen=True, kw_only=True)
class Window:
    """The chunks returned around each hit: its neighbors, in reading order.

    Like a SQL window frame: the chunks of the hit's partition (the same values for
    every `partition_by` field) whose `order_by` is within `chunks_each_side` of the
    hit's. Small chunks rank precisely, and their window gives the reader the text
    around them.

    A `Take` finds the windows after the final gate and the cut, for the hits it
    returns only, in one read of the index. The run's filter and validity apply to
    every neighbor, so a window shows nothing a search couldn't; a neighbor either
    hides leaves a gap. Windows of nearby hits may overlap, and each holds all of its
    chunks.

    `order_by` should number each partition's chunks with unique, consecutive
    integers, such as their position in a document's version: a window then holds at
    most `2 * chunks_each_side + 1` chunks. Gaps in the numbering only shorten
    windows. Repeated values are allowed, ordered by chunk id, but then the bound
    doesn't hold. Every field must be one of the schema of the statement's index,
    checked when the statement is prepared.

    Attributes:
        partition_by: The fields whose values every chunk of a window shares with
            its hit, e.g. `(KbChunks.document_id, KbChunks.document_version)`; empty
            when the whole index is one sequence. It has no default, so a window that
            runs across documents is always asked for.
        order_by: The field numbering the chunks of a partition, e.g.
            `KbChunks.position`.
        chunks_each_side: The most neighbors before the hit, and after it.

    Example:
        search("chunks").retrieve(scan("dense", depth=50)).take(
            window=Window(
                partition_by=(KbChunks.document_id, KbChunks.document_version),
                order_by=KbChunks.position,
                chunks_each_side=1,
            )
        )
    """

    partition_by: tuple[One[Any], ...]
    order_by: One[int]
    chunks_each_side: int

    def __post_init__(self) -> None:
        """Check the fields' kinds, and the number of chunks each side.

        Raises:
            TypeError: `partition_by` isn't a tuple or holds a field that isn't a
                `One`, or `order_by` isn't a `One[int]`.
            ValueError: `chunks_each_side` isn't a positive `int`.
        """
        if type(self.partition_by) is not tuple:
            raise TypeError("partition_by must be a tuple")

        for partition_field in self.partition_by:
            if not isinstance(partition_field, One):
                raise TypeError(
                    "partition_by must be fields of type One; "
                    f"found a {type(partition_field).__name__} instead"
                )

        if not isinstance(self.order_by, One) or self.order_by.value_type is not int:
            value_type = (
                f"[{self.order_by.value_type.__name__}]"
                if isinstance(self.order_by, Field)
                else ""
            )
            raise TypeError(
                "order_by must be a field of type One[int], "
                f"got {type(self.order_by).__name__}{value_type}"
            )
        check_positive_int(self.chunks_each_side, name="chunks_each_side")


@dataclass(frozen=True, kw_only=True)
class Take:
    """The last stage: keep the best candidates and return them as passages.

    It holds no limit: how many passages to return is given with the question, each
    time the statement runs. The limit counts hits: a window adds chunks to each
    hit's passage, never more passages.

    Attributes:
        window: The chunks to return around each hit (see `Window`); `None` returns
            each hit alone.
    """

    window: Window | None = None

    def __post_init__(self) -> None:
        """Check the window.

        Raises:
            TypeError: `window` is neither a `Window` nor `None`.
        """
        if not (self.window is None or type(self.window) is Window):
            raise TypeError("window must be of type Window or None")
