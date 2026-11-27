"""`Take(window=…)`: returning the chunks around each hit, read once, after the gate.

Runs on an in-memory index of numbered chunks, each named `<document>.<version>.
<position>`, with a lookup that finds the chunks a question names, so every test
chooses its hits exactly.
"""

import uuid
from collections.abc import Sequence

import pytest

from salemdex_retrieval import (
    Chunk,
    Dense,
    Engine,
    Everything,
    LookupScan,
    One,
    Schema,
    SearchableIndex,
    Window,
    scan,
    search,
)
from salemdex_retrieval.embedders import DenseBinding
from salemdex_retrieval.embedders.fake import FakeDenseEmbedder
from salemdex_retrieval.embedders.protocol import embed_chunks
from salemdex_retrieval.engine import AlwaysValid, Found, Passage, Validity
from salemdex_retrieval.errors import SchemaMismatchError
from salemdex_retrieval.filters import Filter, and_, not_, or_
from salemdex_retrieval.index.in_memory import InMemoryIndex


class Docs(Schema):
    document = One(str, indexed=True)
    version = One(int, indexed=True)
    position = One(int, indexed=True)
    dense = Dense(size=8, distance="cosine")


class Other(Schema):
    position = One(int)


EMBEDDER = FakeDenseEmbedder(size=8)

# Neighbours share the hit's document and version: the server numbers chunks per
# version, so a window must never mix two versions of a document
WINDOW = Window(
    partition_by=(Docs.document, Docs.version),
    order_by=Docs.position,
    chunks_each_side=1,
)


class CountingIndex(InMemoryIndex):
    """An in-memory index recording how many chunks each `find` returned."""

    def __init__(self) -> None:
        super().__init__(schema=Docs)
        self.finds: list[int] = []

    def find(self, *, where: Filter) -> list[Chunk]:
        found = super().find(where=where)
        self.finds.append(len(found))
        return found


class DroppingValidity:
    """A validity check dropping the chunks named, recording the names it's asked
    about."""

    def __init__(self, *drop: str) -> None:
        self.drop = set(drop)
        self.calls: list[list[str]] = []

    def valid_ids(self, chunks: Sequence[Chunk]) -> set[uuid.UUID]:
        self.calls.append(sorted(chunk.text for chunk in chunks))
        return {chunk.id for chunk in chunks if chunk.text not in self.drop}


def build_chunk(document: str, version: int, position: int) -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        text=f"{document}.{version}.{position}",
        field_values=[
            Docs.document.value(document),
            Docs.version.value(version),
            Docs.position.value(position),
        ],
    )


def stored_index(chunks: Sequence[Chunk] | None = None) -> CountingIndex:
    """Return an index of `chunks`; by default, documents a and b, version 1, and a
    version 2 of a, each numbered 0 to 9."""
    if chunks is None:
        chunks = [
            build_chunk(document, version, position)
            for document, version in [("a", 1), ("b", 1), ("a", 2)]
            for position in range(10)
        ]
    index = CountingIndex()
    index.put(
        embed_chunks(
            chunks, bindings=[DenseBinding(vector=Docs.dense, embedder=EMBEDDER)]
        )
    )
    return index


def parse_hits(question: str) -> Filter | None:
    """Read "a.1.3 b.1.7" as the chunks it names: the hits of a lookup."""
    named = [name.split(".") for name in question.split()]
    if not named:
        return None
    return or_(
        *(
            and_(
                Docs.document.eq(document),
                Docs.version.eq(int(version)),
                Docs.position.eq(int(position)),
            )
            for document, version, position in named
        )
    )


def build_engine(index: InMemoryIndex, *, validity: Validity | None = None) -> Engine:
    return Engine(
        validity=validity if validity is not None else AlwaysValid(),
        indexes={
            "docs": SearchableIndex(
                index=index, scans={"lookup": LookupScan(parse_filter=parse_hits)}
            )
        },
        rerankers={},
    )


def take(
    question: str,
    *,
    window: Window | None = WINDOW,
    index: InMemoryIndex | None = None,
    validity: Validity | None = None,
    where: Filter | None = None,
    limit: int = 10,
) -> list[Passage]:
    """Return the passages a lookup of `question` takes with `window`."""
    engine = build_engine(
        index if index is not None else stored_index(), validity=validity
    )
    prepared = engine.prepare(
        search("docs").retrieve(scan("lookup", depth=10)).take(window=window)
    )
    return prepared.execute(
        question,
        limit=limit,
        where=where if where is not None else Everything(),
    )


def windows(passages: Sequence[Passage]) -> dict[str, list[str]]:
    """Return each passage's window as names, keyed by its hit's name."""
    return {
        passage.chunk.text: [chunk.text for chunk in passage.window]
        for passage in passages
    }


# What a window holds


def test_without_a_window_each_passage_holds_its_hit_alone() -> None:
    assert windows(take("a.1.3", window=None)) == {"a.1.3": ["a.1.3"]}


def test_a_window_holds_the_hit_and_its_neighbours_in_order() -> None:
    assert windows(take("a.1.3")) == {"a.1.3": ["a.1.2", "a.1.3", "a.1.4"]}


def test_a_window_holds_as_many_neighbours_each_side_as_asked() -> None:
    wide = Window(
        partition_by=(Docs.document, Docs.version),
        order_by=Docs.position,
        chunks_each_side=2,
    )

    assert windows(take("a.1.5", window=wide)) == {
        "a.1.5": ["a.1.3", "a.1.4", "a.1.5", "a.1.6", "a.1.7"]
    }


def test_a_window_at_the_start_of_a_document_is_shorter() -> None:
    assert windows(take("a.1.0")) == {"a.1.0": ["a.1.0", "a.1.1"]}


def test_a_window_keeps_to_its_hits_document_and_version() -> None:
    # b.1.2 and a.2.2 sit at the same positions, in another document or version
    assert windows(take("a.1.3"))["a.1.3"] == ["a.1.2", "a.1.3", "a.1.4"]


def test_a_window_with_no_partition_runs_across_documents() -> None:
    # The whole index is one sequence, so every chunk at a near position is in it
    across = Window(partition_by=(), order_by=Docs.position, chunks_each_side=1)

    window = windows(take("a.1.0", window=across))["a.1.0"]

    assert sorted(window) == ["a.1.0", "a.1.1", "a.2.0", "a.2.1", "b.1.0", "b.1.1"]
    assert [name.rsplit(".", 1)[1] for name in window] == ["0", "0", "0", "1", "1", "1"]


def test_the_passage_keeps_its_hit_and_trail() -> None:
    [passage] = take("a.1.3")

    assert passage.chunk.text == "a.1.3"
    assert passage.trail == (Found(scan="lookup", rank=1, score=1.0),)


# Several hits


def test_far_apart_hits_in_one_document_each_get_their_own_window() -> None:
    assert windows(take("a.1.3 a.1.8")) == {
        "a.1.3": ["a.1.2", "a.1.3", "a.1.4"],
        "a.1.8": ["a.1.7", "a.1.8", "a.1.9"],
    }


def test_hits_in_different_documents_each_get_their_own_documents_window() -> None:
    assert windows(take("a.1.3 b.1.7")) == {
        "a.1.3": ["a.1.2", "a.1.3", "a.1.4"],
        "b.1.7": ["b.1.6", "b.1.7", "b.1.8"],
    }


def test_neighbouring_hits_appear_in_each_others_windows() -> None:
    # Windows may overlap; each holds all of its own chunks, every one once
    assert windows(take("a.1.3 a.1.4")) == {
        "a.1.3": ["a.1.2", "a.1.3", "a.1.4"],
        "a.1.4": ["a.1.3", "a.1.4", "a.1.5"],
    }


def test_the_limit_counts_hits_not_the_chunks_of_their_windows() -> None:
    passages = take("a.1.3 b.1.7", limit=1)

    assert len(passages) == 1
    assert len(passages[0].window) == 3


# Gaps: the filter and validity apply to every neighbour


def test_a_neighbour_the_filter_hides_leaves_a_gap() -> None:
    where = not_(Docs.position.eq(4))

    assert windows(take("a.1.3", where=where)) == {"a.1.3": ["a.1.2", "a.1.3"]}


def test_a_neighbour_validity_drops_leaves_a_gap() -> None:
    validity = DroppingValidity("a.1.4")

    assert windows(take("a.1.3", validity=validity)) == {"a.1.3": ["a.1.2", "a.1.3"]}


def test_validity_is_asked_about_the_neighbours_but_not_the_hits_again() -> None:
    # The hits passed the final gate already; asking again could drop one from its
    # own window
    validity = DroppingValidity()

    take("a.1.3 a.1.4", validity=validity)

    assert validity.calls[-1] == ["a.1.2", "a.1.5"]


def test_a_gap_in_the_numbering_only_shortens_a_window() -> None:
    index = stored_index([build_chunk("a", 1, position) for position in [1, 2, 4, 5]])

    assert windows(take("a.1.2", index=index)) == {"a.1.2": ["a.1.1", "a.1.2"]}


def test_chunks_sharing_a_position_are_ordered_by_id() -> None:
    # Allowed, if unusual: the order is then the same on every run
    first, second = build_chunk("a", 1, 4), build_chunk("a", 1, 4)
    index = stored_index([build_chunk("a", 1, 3), first, second])
    by_id = [chunk.text for chunk in sorted([first, second], key=lambda c: c.id)]

    [passage] = take("a.1.3", index=index)

    assert [chunk.id for chunk in passage.window[1:]] == sorted([first.id, second.id])
    assert [chunk.text for chunk in passage.window[1:]] == by_id


# Reading the index


def test_the_windows_are_read_in_one_find_of_only_their_chunks() -> None:
    index = stored_index()

    take("a.1.3 a.1.8 b.1.5", index=index)

    # The lookup's own find, then one for every window: 3 hits, 3 chunks each
    assert index.finds == [3, 9]


def test_without_a_window_nothing_more_is_read() -> None:
    index = stored_index()

    take("a.1.3", window=None, index=index)

    assert index.finds == [1]


def test_with_no_hits_nothing_more_is_read() -> None:
    index = stored_index()

    passages = take("a.1.3", index=index, where=Docs.document.eq("z"))

    assert passages == []
    assert index.finds == [0]


def test_a_window_naming_a_field_its_index_doesnt_declare_is_refused_when_prepared() -> (
    None
):
    index = stored_index()
    foreign = Window(partition_by=(), order_by=Other.position, chunks_each_side=1)

    with pytest.raises(SchemaMismatchError, match="not in schema Docs"):
        build_engine(index).prepare(
            search("docs").retrieve(scan("lookup", depth=10)).take(window=foreign)
        )

    assert index.finds == []
