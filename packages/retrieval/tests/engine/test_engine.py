import dataclasses
import math
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pytest

from salemdex_retrieval.chunks import Chunk, EmbeddedChunk, Hit
from salemdex_retrieval.embedders.fake import FakeDenseEmbedder
from salemdex_retrieval.embedders.protocol import DenseBinding, embed_chunks
from salemdex_retrieval.engine.engine import Engine, PreparedStatement
from salemdex_retrieval.engine.results import Found, Fused, Passage, Reranked
from salemdex_retrieval.engine.sources import VectorSource
from salemdex_retrieval.engine.stages import Retrieve, Rrf, Take, source
from salemdex_retrieval.engine.statements import Statement, search
from salemdex_retrieval.engine.validity import AlwaysValid, Validity
from salemdex_retrieval.errors import SchemaMismatchError, StatementError
from salemdex_retrieval.filters import Everything, Filter, and_, not_
from salemdex_retrieval.index.in_memory import InMemoryIndex
from salemdex_retrieval.index.protocol import Index
from salemdex_retrieval.rerankers.protocol import Reranker
from salemdex_retrieval.schema import One, Schema
from salemdex_retrieval.vectors import Dense, Vector, VectorQuery

SIZE = 16


class Chunks(Schema):
    group = One(str, indexed=True)
    dense = Dense(size=SIZE, distance="cosine")
    title = Dense(size=SIZE, distance="cosine")


class Other(Schema):
    group = One(str, indexed=True)
    dense = Dense(size=SIZE, distance="cosine")


PUBLIC = Chunks.group.eq("public")
DENSE = source("dense", depth=10)
TITLE = source("title", depth=10)


def build_chunk(text: str, *, group: str = "public") -> Chunk:
    """Return a chunk of `group` whose embedded text is `text` with a heading."""
    return Chunk(
        id=uuid.uuid4(),
        text=text,
        embedded_text=f"Điều 1. {text}",
        field_values=[Chunks.group.value(group)],
    )


A, B, C, D = (build_chunk(text) for text in ["a", "b", "c", "d"])


# Fakes that record what the engine asks of them


class SpyEmbedder(FakeDenseEmbedder):
    """A fake embedder recording what it embedded, as queries or as passages."""

    def __init__(self) -> None:
        super().__init__(size=SIZE)
        self.calls: list[tuple[str, list[str]]] = []

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(("queries", list(texts)))
        return super().embed_queries(texts)

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(("passages", list(texts)))
        return super().embed_passages(texts)


class ScriptedIndex:
    """An index of `schema` returning fixed hits per vector, whatever the query, and
    recording each search."""

    def __init__(
        self, hits: dict[Vector, list[Hit]], *, schema: type[Schema] = Chunks
    ) -> None:
        self.hits = hits
        self._schema = schema
        self.searches: list[tuple[Vector, Filter, int]] = []

    @property
    def schema(self) -> type[Schema]:
        return self._schema

    def search(self, query: VectorQuery, *, where: Filter, limit: int) -> list[Hit]:
        self.searches.append((query.vector, where, limit))
        return self.hits.get(query.vector, [])[:limit]

    def put(self, embedded_chunks: Iterable[EmbeddedChunk]) -> None:
        raise NotImplementedError

    def find(self, *, where: Filter) -> list[Chunk]:
        raise NotImplementedError

    def delete(self, *, where: Filter) -> None:
        raise NotImplementedError

    def replace(self, *, where: Filter, with_: Iterable[EmbeddedChunk]) -> None:
        raise NotImplementedError


class SpyValidity:
    """A validity check dropping chosen chunks, from a chosen call on, and recording
    the ids it's asked about."""

    def __init__(self, *drop: Chunk, from_call: int = 1) -> None:
        self.drop = {chunk.id for chunk in drop}
        self.from_call = from_call
        self.calls: list[list[uuid.UUID]] = []

    def valid_ids(self, chunks: Sequence[Chunk]) -> set[uuid.UUID]:
        self.calls.append([chunk.id for chunk in chunks])
        dropped = self.drop if len(self.calls) >= self.from_call else set()
        return {chunk.id for chunk in chunks} - dropped


class SpyReranker:
    """A reranker scoring texts from a table (0.0 otherwise), recording what it reads."""

    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.scores = scores or {}
        self.texts: list[list[str]] = []

    def score(self, question: str, texts: Sequence[str]) -> Sequence[float]:
        self.texts.append(list(texts))
        return [self.scores.get(text, 0.0) for text in texts]


def scripted(**hits_by_vector: list[Chunk]) -> ScriptedIndex:
    """Return an index giving each named vector's chunks, scored 0.9, 0.8, …"""
    return ScriptedIndex(
        {
            Chunks.vectors()[name]: [
                Hit(chunk=chunk, score=round(0.9 - 0.1 * position, 1))
                for position, chunk in enumerate(chunks)
            ]
            for name, chunks in hits_by_vector.items()
        }
    )


def vector_source(
    index: Index,
    vector: Dense = Chunks.dense,
    *,
    embedder: FakeDenseEmbedder | None = None,
) -> VectorSource:
    """Return a registered source searching `vector` in `index`."""
    return VectorSource(
        index=index,
        binding=DenseBinding(
            vector=vector, embedder=embedder or FakeDenseEmbedder(size=SIZE)
        ),
    )


def build_engine(
    index: Index,
    *,
    validity: Validity | None = None,
    rerankers: Mapping[str, Reranker] | None = None,
    embedder: FakeDenseEmbedder | None = None,
) -> Engine:
    """Return an engine registering "dense" and "title", both searching `index`."""
    return Engine(
        validity=validity or AlwaysValid(),
        sources={
            "dense": vector_source(index, Chunks.dense, embedder=embedder),
            "title": vector_source(index, Chunks.title, embedder=embedder),
        },
        rerankers=rerankers or {},
    )


def run(
    statement: Statement,
    *,
    engine: Engine,
    question: str = "q",
    limit: int = 5,
    where: Filter = PUBLIC,
) -> list[Passage]:
    """Prepare `statement` on `engine`, and run it once."""
    return engine.prepare(statement).execute(
        question=question, limit=limit, where=where
    )


def ids(passages: Iterable[Passage]) -> list[uuid.UUID]:
    """Return the chunk id of each passage, in order."""
    return [passage.chunk.id for passage in passages]


# Building an engine


@pytest.mark.parametrize("name", ["", "  ", " dense", "dense\n"])
def test_an_engine_refuses_a_source_name_no_statement_could_give(name: str) -> None:
    # Statements strip their names, so an unstripped one could never be matched
    with pytest.raises(ValueError, match="source's name"):
        Engine(
            validity=AlwaysValid(),
            sources={name: vector_source(scripted())},
            rerankers={},
        )


@pytest.mark.parametrize("name", ["", "  ", " bge", "bge\n"])
def test_an_engine_refuses_a_reranker_name_no_statement_could_give(name: str) -> None:
    with pytest.raises(ValueError, match="reranker's name"):
        Engine(validity=AlwaysValid(), sources={}, rerankers={name: SpyReranker()})


def test_an_engine_refuses_a_name_that_isnt_a_str() -> None:
    with pytest.raises(TypeError, match="source's name"):
        Engine(
            validity=AlwaysValid(),
            sources={None: vector_source(scripted())},  # type: ignore[dict-item]
            rerankers={},
        )


def test_changing_the_mappings_given_doesnt_change_what_the_engine_registered() -> None:
    first, second = scripted(dense=[A]), scripted(dense=[B])
    sources = {"dense": vector_source(first)}
    engine = Engine(validity=AlwaysValid(), sources=sources, rerankers={})

    sources["dense"] = vector_source(second)

    assert ids(run(search().retrieve(DENSE).take(), engine=engine)) == [A.id]
    assert second.searches == []


def test_an_engine_cannot_be_changed() -> None:
    engine = build_engine(scripted())

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(engine, "_validity", SpyValidity(A))  # noqa: B010


def test_an_engine_exposes_none_of_what_it_registered() -> None:
    # Indexes are reached only through a prepared statement, which adds the filter
    engine = build_engine(scripted())

    for name in ["validity", "sources", "rerankers"]:
        assert not hasattr(engine, name), name


# Preparing a statement


def test_preparing_gives_a_prepared_statement_holding_the_statement() -> None:
    statement = search().retrieve(DENSE).take()

    prepared = build_engine(scripted()).prepare(statement)

    assert isinstance(prepared, PreparedStatement)
    assert prepared.statement == statement


def test_preparing_a_statement_built_by_hand_holds_it_as_the_steps_would() -> None:
    # As config might give it: a list, not a tuple
    by_hand = Statement(stages=[Retrieve(sources=(DENSE,)), Take()])  # type: ignore[arg-type]

    prepared = build_engine(scripted()).prepare(by_hand)

    assert prepared.statement == search().retrieve(DENSE).take()


def test_a_statement_out_of_order_is_refused_when_prepared() -> None:
    statement = Statement(stages=(Take(), Retrieve(sources=(DENSE,))))

    with pytest.raises(StatementError, match="can't follow Question"):
        build_engine(scripted()).prepare(statement)


@pytest.mark.parametrize(
    "statement",
    [
        pytest.param(
            search().retrieve(source("sparse", depth=10)).take(), id="the only source"
        ),
        pytest.param(
            search().retrieve(source("sparse", depth=10), DENSE).merge(Rrf()).take(),
            id="the first of several",
        ),
        pytest.param(
            search().retrieve(DENSE, source("sparse", depth=10)).merge(Rrf()).take(),
            id="a later one",
        ),
    ],
)
def test_a_source_the_engine_didnt_register_is_refused(statement: Statement) -> None:
    with pytest.raises(StatementError, match="source 'sparse' was not registered"):
        build_engine(scripted()).prepare(statement)


def test_a_reranker_the_engine_didnt_register_is_refused() -> None:
    engine = build_engine(scripted(), rerankers={"bge": SpyReranker()})

    with pytest.raises(StatementError, match="reranker 'other' was not registered"):
        engine.prepare(search().retrieve(DENSE).rerank("other").take())


def test_a_retrieve_whose_sources_search_different_indexes_is_refused() -> None:
    # Chunk ids are unique only within one index, so fusion couldn't tell chunks apart
    engine = Engine(
        validity=AlwaysValid(),
        sources={
            "dense": vector_source(scripted()),
            "elsewhere": vector_source(scripted(), Chunks.title),
        },
        rerankers={},
    )
    statement = (
        search().retrieve(DENSE, source("elsewhere", depth=10)).merge(Rrf()).take()
    )

    with pytest.raises(StatementError, match="different indexes"):
        engine.prepare(statement)


def test_a_retrieve_searching_one_vector_under_two_names_is_refused() -> None:
    # The shallower list would be the start of the deeper one, counted twice by fusion
    index = scripted()
    engine = Engine(
        validity=AlwaysValid(),
        sources={
            "dense": vector_source(index),
            "dense_again": vector_source(index),
        },
        rerankers={},
    )
    statement = (
        search().retrieve(DENSE, source("dense_again", depth=50)).merge(Rrf()).take()
    )

    with pytest.raises(StatementError, match="search vector 'dense'"):
        engine.prepare(statement)


def test_a_refused_statement_calls_no_model_and_reads_no_index() -> None:
    embedder, reranker, index = SpyEmbedder(), SpyReranker(), scripted(dense=[A])
    engine = build_engine(index, embedder=embedder, rerankers={"bge": reranker})

    with pytest.raises(StatementError):
        engine.prepare(search().retrieve(DENSE).rerank("other").take())

    assert embedder.calls == []
    assert reranker.texts == []
    assert index.searches == []


# A prepared statement: built only by prepare, never changed


def test_a_prepared_statement_cannot_be_built_but_by_prepare() -> None:
    with pytest.raises(TypeError, match=re.escape("engine.prepare(statement)")):
        PreparedStatement()


def test_a_prepared_statement_cannot_be_changed() -> None:
    prepared = build_engine(scripted()).prepare(search().retrieve(DENSE).take())
    unchecked = Statement(
        stages=(Retrieve(sources=(source("sparse", depth=10),)), Take())
    )

    # Refused by mypy too: `statement` is a read-only property
    with pytest.raises(AttributeError, match="can't be changed"):
        prepared.statement = unchecked  # type: ignore[misc]


def test_a_prepared_statement_cannot_be_copied_with_another_statement() -> None:
    # `dataclasses.replace` would build a copy that skipped the checks
    prepared = build_engine(scripted()).prepare(search().retrieve(DENSE).take())

    with pytest.raises(TypeError):
        dataclasses.replace(prepared, statement=search().retrieve(TITLE).take())  # type: ignore[type-var]


def test_one_prepared_statement_runs_for_every_question_limit_and_filter() -> None:
    embedder, index = SpyEmbedder(), scripted(dense=[A, B, C])
    prepared = build_engine(index, embedder=embedder).prepare(
        search().retrieve(DENSE).take()
    )

    first = prepared.execute(question="first", limit=1, where=PUBLIC)
    second = prepared.execute(question="second", limit=3, where=Everything())

    assert ids(first) == [A.id]
    assert ids(second) == [A.id, B.id, C.id]
    assert [texts for _, texts in embedder.calls] == [["first"], ["second"]]
    assert [where for _, where, _ in index.searches] == [PUBLIC, Everything()]


# Running: the inputs, checked before any model is called or any index is read


@pytest.mark.parametrize(
    ("inputs", "error", "message"),
    [
        pytest.param(
            {"question": "  "}, ValueError, "question", id="an empty question"
        ),
        pytest.param(
            {"question": None}, TypeError, "question", id="a question that isn't a str"
        ),
        pytest.param({"limit": 0}, ValueError, "limit", id="a zero limit"),
        pytest.param({"limit": True}, ValueError, "limit", id="a bool limit"),
        pytest.param(
            # What comparing a field with == gives, since fields don't define __eq__
            {"where": Chunks.group == "public"},  # type: ignore[comparison-overlap]
            TypeError,
            "execute: bool is not a valid Filter",
            id="a comparison as the filter",
        ),
        pytest.param(
            {"where": None},
            TypeError,
            "execute: NoneType is not a valid Filter",
            id="no filter",
        ),
        pytest.param(
            {"where": Other.group.eq("public")},
            SchemaMismatchError,
            "not in schema Chunks",
            id="a filter of another schema",
        ),
        pytest.param(
            {"where": and_(PUBLIC, not_(Other.group.eq("hr")))},
            SchemaMismatchError,
            "not in schema Chunks",
            id="a filter nesting another schema's field after a fitting one",
        ),
    ],
)
def test_wrong_inputs_are_refused_before_any_model_or_index_is_used(
    inputs: dict[str, Any], error: type[Exception], message: str
) -> None:
    embedder, index = SpyEmbedder(), scripted(dense=[A])
    prepared = build_engine(index, embedder=embedder).prepare(
        search().retrieve(DENSE).take()
    )
    given: dict[str, Any] = {"question": "q", "limit": 5, "where": PUBLIC, **inputs}

    with pytest.raises(error, match=re.escape(message)):
        prepared.execute(**given)

    assert embedder.calls == []
    assert index.searches == []


def test_the_question_is_stripped_before_it_is_embedded() -> None:
    embedder = SpyEmbedder()

    run(
        search().retrieve(DENSE).take(),
        engine=build_engine(scripted(), embedder=embedder),
        question=" q\n",
    )

    assert embedder.calls == [("queries", ["q"])]


# The filter: inside every source's search


def test_a_hidden_chunk_never_appears_even_when_it_matches_best() -> None:
    hidden = build_chunk("thời hạn nộp hồ sơ", group="finance")
    visible = build_chunk("lương cơ bản")
    embedder = FakeDenseEmbedder(size=SIZE)
    index = InMemoryIndex(schema=Chunks)
    index.put(
        embed_chunks(
            [hidden, visible],
            bindings=[
                DenseBinding(vector=Chunks.dense, embedder=embedder),
                DenseBinding(vector=Chunks.title, embedder=embedder),
            ],
        )
    )

    passages = run(
        search().retrieve(DENSE).take(),
        engine=build_engine(index),
        question="thời hạn nộp hồ sơ",
    )

    assert ids(passages) == [visible.id]


def test_every_source_searches_with_the_filter() -> None:
    index = scripted(dense=[A], title=[B])

    run(
        search().retrieve(DENSE, TITLE).merge(Rrf()).take(),
        engine=build_engine(index),
    )

    assert [where for _, where, _ in index.searches] == [PUBLIC, PUBLIC]


def test_everything_as_the_filter_reaches_every_source_as_it_is() -> None:
    index = scripted(dense=[A], title=[B])

    run(
        search().retrieve(DENSE, TITLE).merge(Rrf()).take(),
        engine=build_engine(index),
        where=Everything(),
    )

    assert [where for _, where, _ in index.searches] == [Everything(), Everything()]


# Retrieve


def test_each_source_searches_its_own_vector_to_its_depth() -> None:
    index = scripted()

    run(
        search()
        .retrieve(source("dense", depth=50), source("title", depth=20))
        .merge(Rrf())
        .take(),
        engine=build_engine(index),
    )

    assert [(vector, limit) for vector, _, limit in index.searches] == [
        (Chunks.dense, 50),
        (Chunks.title, 20),
    ]


def test_the_question_is_embedded_as_a_query_not_as_a_passage() -> None:
    embedder = SpyEmbedder()

    run(
        search().retrieve(DENSE).take(),
        engine=build_engine(scripted(), embedder=embedder),
    )

    assert [kind for kind, _ in embedder.calls] == ["queries"]


def test_an_embedder_giving_no_row_for_the_question_is_refused() -> None:
    class Silent(FakeDenseEmbedder):
        def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
            return []

    with pytest.raises(ValueError, match="returned 0 rows for one question"):
        run(
            search().retrieve(DENSE).take(),
            engine=build_engine(scripted(), embedder=Silent(size=SIZE)),
        )


def test_one_source_gives_its_hits_in_order_each_found_by_it() -> None:
    passages = run(
        search().retrieve(DENSE).take(), engine=build_engine(scripted(dense=[A, B, C]))
    )

    assert ids(passages) == [A.id, B.id, C.id]
    assert [passage.trail for passage in passages] == [
        (Found(source="dense", rank=1, score=0.9),),
        (Found(source="dense", rank=2, score=0.8),),
        (Found(source="dense", rank=3, score=0.7),),
    ]


def test_the_trail_names_each_source_as_the_engine_registered_it() -> None:
    # The name, not the vector: two engines may register one vector differently
    index = scripted(dense=[A])
    engine = Engine(
        validity=AlwaysValid(), sources={"semantic": vector_source(index)}, rerankers={}
    )

    [passage] = run(
        search().retrieve(source("semantic", depth=10)).take(), engine=engine
    )

    assert passage.trail == (Found(source="semantic", rank=1, score=0.9),)


# Validity: where candidates enter, and again before take


def test_a_chunk_validity_drops_at_retrieve_never_appears() -> None:
    passages = run(
        search().retrieve(DENSE).take(),
        engine=build_engine(scripted(dense=[A, B, C]), validity=SpyValidity(B)),
    )

    assert ids(passages) == [A.id, C.id]


def test_found_ranks_are_the_sources_own_whatever_validity_drops() -> None:
    passages = run(
        search().retrieve(DENSE).take(),
        engine=build_engine(scripted(dense=[A, B, C]), validity=SpyValidity(B)),
    )

    assert [passage.trail[0].rank for passage in passages] == [1, 3]


def test_a_stale_chunk_is_never_reranked() -> None:
    reranker = SpyReranker()
    engine = build_engine(
        scripted(dense=[A, B]), validity=SpyValidity(B), rerankers={"spy": reranker}
    )

    run(search().retrieve(DENSE).rerank("spy").take(), engine=engine)

    assert reranker.texts == [[A.embedded_text]]


def test_a_chunk_turning_invalid_during_the_search_never_appears() -> None:
    # Valid when retrieved, deleted before the final gate
    passages = run(
        search().retrieve(DENSE).take(),
        engine=build_engine(
            scripted(dense=[A, B]), validity=SpyValidity(A, from_call=2)
        ),
    )

    assert ids(passages) == [B.id]


def test_validity_is_asked_once_per_gate_however_many_sources() -> None:
    validity = SpyValidity()
    engine = build_engine(
        scripted(dense=[A, B], title=[C]),
        validity=validity,
        rerankers={"spy": SpyReranker()},
    )

    run(
        search().retrieve(DENSE, TITLE).merge(Rrf()).rerank("spy").take(), engine=engine
    )

    assert len(validity.calls) == 2


def test_a_chunk_several_sources_found_is_asked_about_once() -> None:
    validity = SpyValidity()
    engine = build_engine(scripted(dense=[A, B], title=[B, A]), validity=validity)

    run(search().retrieve(DENSE, TITLE).merge(Rrf()).take(), engine=engine)

    assert sorted(validity.calls[0]) == sorted([A.id, B.id])


def test_validity_isnt_asked_when_nothing_was_found() -> None:
    validity = SpyValidity()

    passages = run(
        search().retrieve(DENSE).take(),
        engine=build_engine(scripted(), validity=validity),
    )

    assert passages == []
    assert validity.calls == []


# Merge


def test_merging_ranks_by_rrf_over_the_lists_that_have_each_chunk() -> None:
    passages = run(
        search().retrieve(DENSE, TITLE).merge(Rrf(k=60)).take(),
        engine=build_engine(scripted(dense=[A, B], title=[C, A])),
    )

    assert ids(passages) == [A.id, C.id, B.id]
    assert [passage.score for passage in passages] == pytest.approx(
        [1 / 61 + 1 / 62, 1 / 61, 1 / 62]
    )


def test_a_merged_trail_holds_each_source_that_found_it_then_the_fusion() -> None:
    [first, *_] = run(
        search().retrieve(DENSE, TITLE).merge(Rrf(k=60)).take(),
        engine=build_engine(scripted(dense=[A, B], title=[C, A])),
    )

    assert first.trail == (
        Found(source="dense", rank=1, score=0.9),
        Found(source="title", rank=2, score=0.8),
        # Summed as the engine does, in list order, so the floats match exactly
        Fused(rank=1, score=1 / 61 + 1 / 62),
    )


def test_fusion_ties_keep_the_order_of_first_appearance() -> None:
    passages = run(
        search().retrieve(DENSE, TITLE).merge(Rrf()).take(),
        engine=build_engine(scripted(dense=[A], title=[B])),
    )

    assert ids(passages) == [A.id, B.id]


# Rerank


def test_a_rerank_scores_every_candidate_by_its_embedded_text() -> None:
    reranker = SpyReranker()
    engine = build_engine(scripted(dense=[A, B, C]), rerankers={"spy": reranker})

    run(search().retrieve(DENSE).rerank("spy").take(), engine=engine, limit=1)

    assert reranker.texts == [[A.embedded_text, B.embedded_text, C.embedded_text]]


def test_a_rerank_orders_by_its_scores_and_adds_them_to_the_trail() -> None:
    reranker = SpyReranker({C.embedded_text: 0.95, A.embedded_text: 0.5})
    engine = build_engine(scripted(dense=[A, B, C]), rerankers={"spy": reranker})

    passages = run(search().retrieve(DENSE).rerank("spy").take(), engine=engine)

    assert ids(passages) == [C.id, A.id, B.id]
    assert [passage.trail[-1] for passage in passages] == [
        Reranked(rank=1, score=0.95),
        Reranked(rank=2, score=0.5),
        Reranked(rank=3, score=0.0),
    ]


def test_each_rerank_uses_the_reranker_its_name_was_registered_for() -> None:
    first = SpyReranker({A.embedded_text: 0.1, B.embedded_text: 0.9})
    second = SpyReranker()
    engine = build_engine(
        scripted(dense=[A, B]), rerankers={"first": first, "second": second}
    )

    run(search().retrieve(DENSE).rerank("first").rerank("second").take(), engine=engine)

    # The second reads the first's order
    assert first.texts == [[A.embedded_text, B.embedded_text]]
    assert second.texts == [[B.embedded_text, A.embedded_text]]


def test_rerank_ties_keep_their_earlier_order() -> None:
    engine = build_engine(scripted(dense=[A, B, C]), rerankers={"spy": SpyReranker()})

    passages = run(search().retrieve(DENSE).rerank("spy").take(), engine=engine)

    assert ids(passages) == [A.id, B.id, C.id]


def test_a_reranker_giving_a_score_that_isnt_finite_is_refused() -> None:
    reranker = SpyReranker({A.embedded_text: math.nan})
    engine = build_engine(scripted(dense=[A]), rerankers={"spy": reranker})

    with pytest.raises(ValueError, match="finite"):
        run(search().retrieve(DENSE).rerank("spy").take(), engine=engine)


# Take


def test_take_keeps_the_best_limit() -> None:
    passages = run(
        search().retrieve(DENSE).take(),
        engine=build_engine(scripted(dense=[A, B, C, D])),
        limit=2,
    )

    assert ids(passages) == [A.id, B.id]


def test_take_fills_its_limit_past_a_chunk_the_final_gate_drops() -> None:
    passages = run(
        search().retrieve(DENSE).take(),
        engine=build_engine(
            scripted(dense=[A, B, C]), validity=SpyValidity(A, from_call=2)
        ),
        limit=2,
    )

    assert ids(passages) == [B.id, C.id]
