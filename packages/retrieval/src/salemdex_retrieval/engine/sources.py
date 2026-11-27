"""The indexes an engine registers, each with the scans that read it.

A statement names its index (`search("chunks")`) and the scans it runs on it
(`scan("dense", depth=50)`). The engine holds, under each index's name, a
`SearchableIndex`: the index, and under each scan's name what that scan reads it
with, either the embedder for the question (`VectorScan`) or the app's parser turning
it into a filter (`LookupScan`). A scan is one way of reading candidates from an
index, as in a SQL query plan; an index read by one of its scans is a source of
candidates. Keeping these objects here, out of statements, is what lets a statement
stay plain data.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import assert_never

from salemdex_retrieval.checks import to_stripped_str
from salemdex_retrieval.embedders import VectorBinding
from salemdex_retrieval.filters import Filter
from salemdex_retrieval.index import Index

# The app's function reading a question into a filter on the schema's fields, or None
# when the question names nothing to look up
type FilterParser = Callable[[str], Filter | None]

# Every kind of scan. A closed union: each `match` over it ends in `assert_never`,
# so mypy points at every place that hasn't learned a new kind.
type RegisteredScan = VectorScan | LookupScan


@dataclass(frozen=True, kw_only=True)
class SearchableIndex:
    """An index, with the scans that read it, as an engine registers it.

    A statement's candidates are chunks of the one index it names, so every scan it
    runs is registered here, on that index. Each vector scan's vector is checked
    against the index's schema when this is built, so no question is ever embedded
    for a vector the index doesn't hold, and the scans are kept as a read-only copy,
    so none changes after that check.

    Attributes:
        index: The index the scans read.
        scans: What each scan name a statement may use on this index stands for: a
            `VectorScan` or a `LookupScan`. Names are given as statements hold them,
            stripped.

    Example:
        SearchableIndex(
            index=chunk_index,
            scans={
                "dense": VectorScan(binding=binding),
                "lookup": LookupScan(parse_filter=parse_reference),
            },
        )
    """

    index: Index
    scans: Mapping[str, RegisteredScan]

    def __post_init__(self) -> None:
        """Check the scans' names and vectors, and keep a read-only copy of the scans.

        Raises:
            TypeError: A scan's name isn't a `str`.
            ValueError: A scan's name is empty, or has whitespace at either end.
            SchemaMismatchError: A vector scan's vector isn't one of the index's
                schema.
        """
        scans: dict[str, RegisteredScan] = {}
        for scan_name, scan in self.scans.items():
            match scan:
                case VectorScan(binding=binding):
                    self.index.schema.check_declares(binding.vector)
                case LookupScan():
                    pass
                case _:
                    assert_never(scan)
            if scan_name != to_stripped_str(scan_name, name="scan's name"):
                raise ValueError("scan's name must be stripped")
            scans[scan_name] = scan
        object.__setattr__(self, "scans", MappingProxyType(scans))


@dataclass(frozen=True, kw_only=True)
class VectorScan:
    """A scan searching one of its index's vectors with the question.

    The binding's embedder turns the question into a query for its vector; the index
    returns the nearest chunks the run's filter (`where`) lets through.

    Attributes:
        binding: The vector to search, and the embedder for the question; the vector
            must be one of the schema of the index the scan is registered on.
    """

    binding: VectorBinding


@dataclass(frozen=True, kw_only=True)
class LookupScan:
    """A scan reading the chunks that a filter, parsed from the question, matches.

    For questions naming what they ask about by an identifier ("article 12 of the staff
    rules", an error code, a product number), which a vector ranks only roughly. The
    app's `parse_filter` reads the question into a filter on the index's fields; the
    framework knows nothing of what the fields mean. Each run:

    - `None` from the parser finds nothing, and reads no index: most questions name
      nothing, and the other scans then carry the search alone;
    - the filter is checked against the index's schema, then ANDed with the run's
      filter (`where`), so a lookup can only narrow what the asker may see, never widen
      it, however the parser built it;
    - every chunk matched scores 1.0: a lookup matches or it doesn't, so no hit is
      better than another;
    - hits come in the order of chunk ids, which means nothing but is the same on every
      run, and are cut at the scan's depth. When more match than that, the ones kept
      are arbitrary: the filter is too broad to be exact, and a narrower parser, or a
      deeper scan, is the fix.

    Every match is read before the cut, so a parser should build narrow filters.

    Attributes:
        parse_filter: The app's parser: a question in, a filter on the index's fields
            out, or `None` when the question names nothing. Its errors pass through.

    Example:
        SearchableIndex(
            index=chunk_index,
            scans={
                "dense": VectorScan(binding=binding),
                "lookup": LookupScan(parse_filter=parse_reference),
            },
        )
    """

    parse_filter: FilterParser
