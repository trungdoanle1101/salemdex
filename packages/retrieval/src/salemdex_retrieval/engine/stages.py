"""The steps a search statement is made of, as plain data: what to do, never with what.

Each stage holds only its own settings, and names the sources and rerankers it uses;
the engine holds what those names stand for (indexes, embedders, models), and gives
the stages meaning when a prepared statement runs them. Holding no objects, a statement
can be logged, compared and replayed, and one built at run time (from config, by an
agent) can name only what the app registered.

A stage checks its own settings when it's built. Checks that need the engine (a name
is registered, the sources of a retrieve search one index) run when the statement is
prepared, so a wrong statement still fails before any model is called or any index is
read.
"""

from __future__ import annotations

from dataclasses import dataclass

from salemdex_retrieval.checks import check_positive_int, to_stripped_str

# Every kind of stage. A closed union: the engine's `match` over it ends in
# `assert_never`, so mypy points at every place that hasn't learned a new kind.
type Stage = Retrieve | Merge | Rerank | Take

# Every way of fusing ranked lists, closed like `Stage`.
type Fusion = Rrf


@dataclass(frozen=True, kw_only=True)
class Retrieve:
    """A stage finding candidates for the question, one ranked list per source.

    Each source's question is embedded, and its index searched with the run's filter
    (`where`). Several sources give several lists, which a `Merge` must fuse before
    any other stage.

    No two sources may share a name: each candidate's trail names the sources that
    found it. That is checked here; two rules need the registered sources, so they're
    checked when the statement is prepared:

    - every source searches the same index, for now: a candidate is known by its
      chunk id, which is unique only within one index, and one index has one schema,
      so one filter fits every source;
    - no two sources search the same vector, whatever their depths: the shallower
      list would be the start of the deeper one, and fusion would count it twice.

    Attributes:
        sources: Where to look, in order; at least one.
    """

    sources: tuple[Source, ...]

    def __post_init__(self) -> None:
        """Check the sources.

        Raises:
            ValueError: There are no sources, or two share a name.
            TypeError: A source isn't a `Source`.
        """
        if not self.sources:
            raise ValueError("a retrieve needs at least one source")

        seen_sources: set[str] = set()
        for position, source in enumerate(self.sources):
            if type(source) is not Source:
                raise TypeError(
                    f"sources[{position}] is {type(source).__name__}; expected a Source"
                )

            if source.name in seen_sources:
                raise ValueError(
                    f"two sources in one retrieve share a name: {source.name!r}"
                )
            seen_sources.add(source.name)


@dataclass(frozen=True, kw_only=True)
class Source:
    """A source a retrieve searches, named as the engine registered it, and how deep.

    Build it with `source`. It holds only data; the engine resolves the name to what
    the source searches (e.g. a `VectorSource`) when the statement is prepared.

    Attributes:
        name: The source's name in the engine, also what each candidate's trail calls
            it, e.g. "dense"; unique within a retrieve, and stripped of whitespace at
            either end.
        depth: The most candidates this source returns.
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


def source(name: str, *, depth: int) -> Source:
    """Return the source registered as `name`, searched for at most `depth` candidates.

    Args:
        name: The source's name in the engine, e.g. "dense".
        depth: The most candidates it returns.

    Returns:
        The source, ready for a retrieve.

    Raises:
        TypeError: `name` isn't a `str`.
        ValueError: `name` is empty or only whitespace, or `depth` isn't a positive
            `int`.

    Example:
        search().retrieve(source("dense", depth=50), source("sparse", depth=50))
    """
    return Source(name=name, depth=depth)


@dataclass(frozen=True, kw_only=True)
class Merge:
    """A stage fusing the ranked lists of several sources into one.

    Attributes:
        fusion: How the lists are fused, e.g. `Rrf()`.
    """

    fusion: Fusion


@dataclass(frozen=True, kw_only=True)
class Rrf:
    """Reciprocal rank fusion: each candidate scores `Σ 1 / (k + rank)` over its lists.

    Only ranks count, never the sources' scores, which aren't on one scale (cosine,
    BM25 and max-sim differ). A candidate several sources rank high beats one a
    single source ranks first.

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
    With one source that's the source's `depth`; after a merge, the union of the
    sources' lists.

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
class Take:
    """The last stage: keep the best candidates and return them as passages.

    It holds no limit: how many passages to return is given with the question, each
    time the statement runs.
    """
