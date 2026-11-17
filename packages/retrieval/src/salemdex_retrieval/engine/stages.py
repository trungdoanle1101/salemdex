"""The steps a search statement is made of, as plain data: what to do, never how.

Each stage holds only its own settings; the statement holds the stages in order, and
the executor gives them meaning when it runs them. Being data, a statement can be
logged, compared and replayed, and one built at run time (from config, by an agent) is
checked the same way as one built in code.

A stage checks its settings when it's built, so a wrong statement fails before any
model is called or any index is read.
"""

from __future__ import annotations

from dataclasses import dataclass

from salemdex_retrieval.checks import check_positive_int, to_stripped_str
from salemdex_retrieval.embedders.protocol import VectorBinding
from salemdex_retrieval.index.protocol import Index
from salemdex_retrieval.rerankers.protocol import Reranker
from salemdex_retrieval.vectors import Vector

# Every kind of stage. A closed union: the executor's `match` over it ends in
# `assert_never`, so mypy points at every place that hasn't learned a new kind.
type Stage = Retrieve | Merge | Rerank | Take

# Every way of fusing ranked lists, closed like `Stage`.
type Fusion = Rrf


@dataclass(frozen=True, kw_only=True)
class Retrieve:
    """A stage finding candidates for the question, one ranked list per source.

    The executor embeds the question for each source and searches its index with the
    session's access filter. Several sources give several lists, which a `Merge`
    must fuse before any other stage.

    Every source searches the same index, for now: a candidate is known by its chunk
    id, which is unique only within one index, and one index has one schema, so one
    access filter fits every source. Indexes are compared as objects, so build one
    index object per stored collection.

    No two sources may search the same vector, whatever their depths: the shallower
    list would be the start of the deeper one, and fusion would count it twice. No two
    may share a name either: each candidate's trail names the sources that found it.

    Attributes:
        sources: Where to look, in order; at least one.
    """

    sources: tuple[Source, ...]

    def __post_init__(self) -> None:
        """Check the sources.

        Raises:
            ValueError: There are no sources, or they search different indexes, or
                two search the same vector or share a name.
            TypeError: A source isn't a `Source`.
        """
        if not self.sources:
            raise ValueError("a retrieve needs at least one source")

        seen_vectors: set[Vector] = set()
        seen_source_names: set[str] = set()
        for position, source in enumerate(self.sources):
            if type(source) is not Source:
                raise TypeError(
                    f"sources[{position}] is {type(source).__name__}; expected a Source"
                )

            if source.index is not self.sources[0].index:
                raise ValueError(
                    "two sources search different indexes, which isn't supported yet; "
                    "search one index, with one source per vector"
                )

            # One index, so the vector alone says what a source searches; vectors
            # hash by identity, as they compare
            vector = source.binding.vector
            if vector in seen_vectors:
                raise ValueError(
                    f"two sources search vector {vector.name!r}; "
                    "give it once, with the depth you need"
                )
            seen_vectors.add(vector)

            if source.name in seen_source_names:
                raise ValueError(
                    f"two sources have the same name {source.name!r}; "
                    "give them different names"
                )
            seen_source_names.add(source.name)


@dataclass(frozen=True, kw_only=True)
class Source:
    """One way of finding candidates: a vector of an index, searched with a question.

    The binding's embedder turns the question into a query for its vector; the index
    returns the nearest chunks the access filter lets through.

    Attributes:
        name: What the source is called in each candidate's trail, e.g. "dense";
            unique within a retrieve, and stripped of whitespace at either end.
        index: The index to search.
        binding: The vector to search, and the embedder for the question; the vector
            must be one of the index's schema.
        depth: The most candidates this source returns.
    """

    name: str
    index: Index
    binding: VectorBinding
    depth: int

    def __post_init__(self) -> None:
        """Strip the name, check it and the depth, and that the schema has the vector.

        Raises:
            TypeError: `name` isn't a `str`.
            ValueError: `name` is empty or only whitespace, or `depth` isn't a
                positive `int`.
            SchemaMismatchError: The vector isn't one of the index's schema, checked
                here so that no question is embedded for it.
        """
        # The dataclass is frozen; this is the one place its name is set
        object.__setattr__(self, "name", to_stripped_str(self.name, name="name"))
        check_positive_int(self.depth, name="depth")
        self.index.schema.check_declares(self.binding.vector)


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
    """A stage scoring the top candidates again with a reranker, and reordering them.

    The candidates below `depth` are dropped: a reranker can only lift what it sees,
    and each candidate it sees costs a model call.

    Attributes:
        reranker: The model scoring each candidate against the question.
        depth: How many of the top candidates it scores.
    """

    reranker: Reranker
    depth: int

    def __post_init__(self) -> None:
        """Check the depth.

        Raises:
            ValueError: `depth` isn't a positive `int`.
        """
        check_positive_int(self.depth, name="depth")


@dataclass(frozen=True, kw_only=True)
class Take:
    """The last stage: keep the best candidates and return them as passages.

    Attributes:
        limit: The most passages returned.
    """

    limit: int

    def __post_init__(self) -> None:
        """Check the limit.

        Raises:
            ValueError: `limit` isn't a positive `int`.
        """
        check_positive_int(self.limit, name="limit")
