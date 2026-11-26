"""The sources an engine registers: what a statement's source names stand for.

A statement names its sources (`source("dense", depth=50)`); the engine holds, under
each name, what that source searches and with what: an index, and either the embedder
for the question (`VectorSource`) or the app's parser turning it into a filter
(`LookupSource`). Keeping these objects here, out of statements, is what lets a
statement stay plain data.
"""

from collections.abc import Callable
from dataclasses import dataclass

from salemdex_retrieval.embedders import VectorBinding
from salemdex_retrieval.filters import Filter
from salemdex_retrieval.index import Index

# The app's function reading a question into a filter on the schema's fields, or None
# when the question names nothing to look up
type FilterParser = Callable[[str], Filter | None]

# Every kind of source. A closed union: each `match` over it ends in `assert_never`,
# so mypy points at every place that hasn't learned a new kind.
type RegisteredSource = VectorSource | LookupSource


@dataclass(frozen=True, kw_only=True)
class VectorSource:
    """A way of finding candidates: a vector of an index, searched with a question.

    The binding's embedder turns the question into a query for its vector; the index
    returns the nearest chunks the run's filter (`where`) lets through.

    Attributes:
        index: The index to search.
        binding: The vector to search, and the embedder for the question; the vector
            must be one of the index's schema.
    """

    index: Index
    binding: VectorBinding

    def __post_init__(self) -> None:
        """Check that the index's schema declares the vector.

        Raises:
            SchemaMismatchError: The vector isn't one of the index's schema, checked
                here, when the engine is built, so that no question is embedded for it.
        """
        self.index.schema.check_declares(self.binding.vector)


@dataclass(frozen=True, kw_only=True)
class LookupSource:
    """A way of finding candidates: the chunks a filter read from the question matches.

    For questions naming what they ask about by an identifier ("article 12 of the staff
    rules", an error code, a product number), which a vector ranks only roughly. The
    app's `parse_filter` reads the question into a filter on the index's fields; the
    framework knows nothing of what the fields mean. Each run:

    - `None` from the parser finds nothing, and reads no index: most questions name
      nothing, and the other sources then carry the search alone;
    - the filter is checked against the index's schema, then ANDed with the run's
      filter (`where`), so a lookup can only narrow what the asker may see, never widen
      it, however the parser built it;
    - every chunk matched scores 1.0: a lookup matches or it doesn't, so no hit is
      better than another;
    - hits come in the order of chunk ids, which means nothing but is the same on every
      run, and are cut at the source's depth. When more match than that, the ones kept
      are arbitrary: the filter is too broad to be exact, and a narrower parser, or a
      deeper source, is the fix.

    Every match is read before the cut, so a parser should build narrow filters.

    Attributes:
        index: The index to read.
        parse_filter: The app's parser: a question in, a filter on the index's fields
            out, or `None` when the question names nothing. Its errors pass through.

    Example:
        engine = Engine(
            validity=validity,
            sources={
                "dense": VectorSource(index=index, binding=binding),
                "lookup": LookupSource(index=index, parse_filter=parse_reference),
            },
            rerankers={"bge": bge},
        )
    """

    index: Index
    parse_filter: FilterParser
