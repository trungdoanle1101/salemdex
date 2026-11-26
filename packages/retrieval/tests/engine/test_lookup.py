"""`LookupSource`: fetching the chunks a filter parsed from the question matches.

Runs on an in-memory index with invented text: articles of two invented rule books,
and a parser of "article N of book X" standing in for an app's own.
"""

import re
import uuid
from collections.abc import Sequence

import pytest

from salemdex_retrieval import (
    Chunk,
    Dense,
    EmbeddedChunk,
    Engine,
    Everything,
    LookupSource,
    One,
    Rrf,
    Schema,
    VectorSource,
    search,
    source,
)
from salemdex_retrieval.embedders import DenseBinding
from salemdex_retrieval.embedders.fake import FakeDenseEmbedder
from salemdex_retrieval.engine import AlwaysValid, Found, Passage, Validity
from salemdex_retrieval.errors import SchemaMismatchError, StatementError
from salemdex_retrieval.filters import Filter, and_
from salemdex_retrieval.index import InMemoryIndex

SIZE = 16


class Rules(Schema):
    book = One(str, indexed=True)
    article = One(int, indexed=True)
    group = One(str, indexed=True)
    dense = Dense(size=SIZE, distance="cosine")


class Other(Schema):
    article = One(int)
    dense = Dense(size=SIZE, distance="cosine")


EMBEDDER = FakeDenseEmbedder(size=SIZE)
LOOKUP = source("lookup", depth=10)
ARTICLE = re.compile(r"article (\d+) of book (\w+)")


def parse_article(question: str) -> Filter | None:
    """Return a filter on the book and article the question names, or None."""
    named = ARTICLE.search(question)
    if named is None:
        return None
    return and_(Rules.book.eq(named[2]), Rules.article.eq(int(named[1])))


def build_chunk(
    text: str, *, book: str, article: int, group: str = "public"
) -> EmbeddedChunk:
    """Return a chunk of `book`'s `article`, embedded by the fake embedder."""
    chunk = Chunk(
        id=uuid.uuid4(),
        text=text,
        field_values=[
            Rules.book.value(book),
            Rules.article.value(article),
            Rules.group.value(group),
        ],
    )
    [row] = EMBEDDER.embed_passages([text])
    return EmbeddedChunk(chunk=chunk, vector_values=[Rules.dense.value(row)])


LEAVE = build_chunk("annual leave is twelve days", book="staff", article=12)
LEAVE_MORE = build_chunk("leave carries over one year", book="staff", article=12)
INTEREST = build_chunk("late payment is charged interest", book="staff", article=13)
OTHER_BOOK = build_chunk("scope of these rules", book="finance", article=12)
STORED = [LEAVE, LEAVE_MORE, INTEREST, OTHER_BOOK]


class CountingIndex(InMemoryIndex):
    """An in-memory index counting its `find` calls."""

    def __init__(self) -> None:
        super().__init__(schema=Rules)
        self.find_calls = 0

    def find(self, *, where: Filter) -> list[Chunk]:
        self.find_calls += 1
        return super().find(where=where)


class DroppingValidity:
    """A validity check dropping the chunks given."""

    def __init__(self, *dropped: EmbeddedChunk) -> None:
        self.dropped = {embedded.chunk.id for embedded in dropped}

    def valid_ids(self, chunks: Sequence[Chunk]) -> set[uuid.UUID]:
        return {chunk.id for chunk in chunks} - self.dropped


def stored_index(embedded: Sequence[EmbeddedChunk] = STORED) -> CountingIndex:
    """Return an index holding `embedded`."""
    index = CountingIndex()
    index.put(embedded)
    return index


def lookup_engine(
    index: InMemoryIndex,
    *,
    parse_filter: object = parse_article,
    validity: Validity | None = None,
) -> Engine:
    """Return an engine with a lookup and a dense source on `index`."""
    return Engine(
        validity=validity if validity is not None else AlwaysValid(),
        sources={
            "lookup": LookupSource(index=index, parse_filter=parse_filter),  # type: ignore[arg-type]
            "dense": VectorSource(
                index=index,
                binding=DenseBinding(vector=Rules.dense, embedder=EMBEDDER),
            ),
        },
        rerankers={},
    )


def look_up(
    question: str,
    *,
    index: InMemoryIndex,
    depth: int = 10,
    where: Filter | None = None,
    validity: Validity | None = None,
) -> list[Passage]:
    """Return what a lookup-only statement finds for `question`."""
    engine = lookup_engine(index, validity=validity)
    prepared = engine.prepare(search().retrieve(source("lookup", depth=depth)).take())
    return prepared.execute(
        question=question,
        limit=10,
        where=where if where is not None else Everything(),
    )


def ids(items: Sequence[Passage]) -> set[uuid.UUID]:
    return {passage.chunk.id for passage in items}


# Finding


def test_a_lookup_finds_exactly_the_chunks_its_filter_matches() -> None:
    passages = look_up("what does article 12 of book staff say?", index=stored_index())

    assert ids(passages) == {LEAVE.chunk.id, LEAVE_MORE.chunk.id}


def test_every_lookup_hit_scores_one_as_an_exact_match() -> None:
    passages = look_up("article 12 of book staff", index=stored_index())

    assert [passage.score for passage in passages] == [1.0, 1.0]


def test_a_lookup_hit_is_found_by_the_lookup_in_the_order_of_chunk_ids() -> None:
    passages = look_up("article 12 of book staff", index=stored_index())

    by_id = sorted([LEAVE, LEAVE_MORE], key=lambda embedded: embedded.chunk.id)
    assert [passage.trail for passage in passages] == [
        (Found(source="lookup", rank=rank, score=1.0),)
        for rank, _ in enumerate(by_id, start=1)
    ]
    assert [passage.chunk.id for passage in passages] == [
        embedded.chunk.id for embedded in by_id
    ]


def test_more_matches_than_the_depth_keep_the_first_by_chunk_id_every_run() -> None:
    articles = [build_chunk(f"rule {n}", book="staff", article=1) for n in range(5)]
    index = stored_index(articles)
    first_two = sorted(embedded.chunk.id for embedded in articles)[:2]

    runs = [
        [p.chunk.id for p in look_up("article 1 of book staff", index=index, depth=2)]
        for _ in range(3)
    ]

    assert runs == [first_two] * 3


def test_a_question_naming_nothing_finds_nothing_and_reads_no_index() -> None:
    index = stored_index()

    passages = look_up("how many days of annual leave?", index=index)

    assert passages == []
    assert index.find_calls == 0


def test_a_question_naming_what_isnt_stored_finds_nothing() -> None:
    assert look_up("article 99 of book staff", index=stored_index()) == []


# The guarantees: the asker's filter and validity apply to a lookup too


def test_a_lookup_never_finds_a_chunk_the_askers_filter_hides() -> None:
    hidden = build_chunk("leave for managers", book="staff", article=12, group="hr")
    index = stored_index([*STORED, hidden])

    passages = look_up(
        "article 12 of book staff", index=index, where=Rules.group.eq("public")
    )

    assert hidden.chunk.id not in ids(passages)
    assert ids(passages) == {LEAVE.chunk.id, LEAVE_MORE.chunk.id}


def test_a_lookup_hit_validity_drops_is_never_shown() -> None:
    passages = look_up(
        "article 12 of book staff",
        index=stored_index(),
        validity=DroppingValidity(LEAVE),
    )

    assert ids(passages) == {LEAVE_MORE.chunk.id}


def test_a_filter_the_lookups_index_cannot_read_is_refused_before_parsing() -> None:
    asked: list[str] = []

    def recording(question: str) -> Filter | None:
        asked.append(question)
        return parse_article(question)

    engine = lookup_engine(stored_index(), parse_filter=recording)
    prepared = engine.prepare(search().retrieve(LOOKUP).take())

    with pytest.raises(SchemaMismatchError):
        prepared.execute(
            question="article 12 of book staff", limit=5, where=Other.article.eq(12)
        )
    assert asked == []


# With other sources


def test_an_exact_match_joins_the_fusion_with_the_vector_sources_hits() -> None:
    engine = lookup_engine(stored_index())
    prepared = engine.prepare(
        search()
        .retrieve(source("dense", depth=10), source("lookup", depth=10))
        .merge(Rrf())
        .take()
    )

    passages = prepared.execute(
        question="what does article 12 of book finance say?",
        limit=10,
        where=Everything(),
    )

    [exact] = [p for p in passages if p.chunk.id == OTHER_BOOK.chunk.id]
    assert [entry.source for entry in exact.trail if isinstance(entry, Found)] == [
        "dense",
        "lookup",
    ]
    assert passages[0].chunk.id == OTHER_BOOK.chunk.id


def test_a_question_naming_nothing_leaves_the_fusion_to_the_other_sources() -> None:
    engine = lookup_engine(stored_index())
    question = "how many days of annual leave?"
    dense_only = engine.prepare(search().retrieve(source("dense", depth=10)).take())
    with_lookup = engine.prepare(
        search()
        .retrieve(source("dense", depth=10), source("lookup", depth=10))
        .merge(Rrf())
        .take()
    )

    alone = dense_only.execute(question=question, limit=10, where=Everything())
    fused = with_lookup.execute(question=question, limit=10, where=Everything())

    assert [p.chunk.id for p in fused] == [p.chunk.id for p in alone]


def test_two_lookups_in_one_retrieve_are_allowed() -> None:
    index = stored_index()
    engine = Engine(
        validity=AlwaysValid(),
        sources={
            "article": LookupSource(index=index, parse_filter=parse_article),
            "book": LookupSource(
                index=index, parse_filter=lambda question: Rules.book.eq("finance")
            ),
        },
        rerankers={},
    )

    engine.prepare(
        search()
        .retrieve(source("article", depth=5), source("book", depth=5))
        .merge(Rrf())
        .take()
    )


def test_a_lookup_and_a_source_on_another_index_are_refused_together() -> None:
    engine = Engine(
        validity=AlwaysValid(),
        sources={
            "lookup": LookupSource(index=stored_index(), parse_filter=parse_article),
            "dense": VectorSource(
                index=InMemoryIndex(schema=Rules),
                binding=DenseBinding(vector=Rules.dense, embedder=EMBEDDER),
            ),
        },
        rerankers={},
    )

    with pytest.raises(StatementError, match="different indexes"):
        engine.prepare(
            search()
            .retrieve(source("lookup", depth=5), source("dense", depth=5))
            .merge(Rrf())
            .take()
        )


# The parser's mistakes


def test_a_parser_returning_something_other_than_a_filter_is_refused() -> None:
    engine = lookup_engine(stored_index(), parse_filter=lambda question: "article 12")
    prepared = engine.prepare(search().retrieve(LOOKUP).take())

    with pytest.raises(TypeError):
        prepared.execute(question="article 12", limit=5, where=Everything())


def test_a_parser_filtering_by_another_schemas_field_is_refused() -> None:
    engine = lookup_engine(
        stored_index(), parse_filter=lambda question: Other.article.eq(12)
    )
    prepared = engine.prepare(search().retrieve(LOOKUP).take())

    with pytest.raises(SchemaMismatchError):
        prepared.execute(question="article 12", limit=5, where=Everything())


def test_an_error_in_the_parser_passes_through() -> None:
    def broken(question: str) -> Filter | None:
        raise RuntimeError("parser broke")

    engine = lookup_engine(stored_index(), parse_filter=broken)
    prepared = engine.prepare(search().retrieve(LOOKUP).take())

    with pytest.raises(RuntimeError, match="parser broke"):
        prepared.execute(question="article 12", limit=5, where=Everything())
