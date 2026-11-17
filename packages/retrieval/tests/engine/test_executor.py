import math
import uuid
from collections.abc import Iterable, Sequence

import pytest

from salemdex_retrieval.chunks import Chunk, EmbeddedChunk, Hit
from salemdex_retrieval.embedders.fake import FakeDenseEmbedder
from salemdex_retrieval.embedders.protocol import DenseBinding, embed_chunks
from salemdex_retrieval.engine.executor import execute
from salemdex_retrieval.engine.results import Found, Fused, Reranked
from salemdex_retrieval.engine.stages import Retrieve, Rrf, Source, Take
from salemdex_retrieval.engine.statements import Search, search
from salemdex_retrieval.engine.validity import AlwaysValid
from salemdex_retrieval.errors import SchemaMismatchError, StatementError
from salemdex_retrieval.filters import Everything, Filter, and_, not_
from salemdex_retrieval.index.in_memory import InMemoryIndex
from salemdex_retrieval.schema import One, Schema
from salemdex_retrieval.vectors import Dense, Vector, VectorQuery

SIZE = 16


class Chunks(Schema):
    group = One(str, indexed=True)
    dense = Dense(size=SIZE, distance="cosine")
    title = Dense(size=SIZE, distance="cosine")


PUBLIC = Chunks.group.eq("public")


def build_chunk(text: str, *, group: str = "public") -> Chunk:
    """Return a chunk of `group` whose embedded text is `text` with a heading."""
    return Chunk(
        id=uuid.uuid4(),
        text=text,
        embedded_text=f"Điều 1. {text}",
        field_values=[Chunks.group.value(group)],
    )


A, B, C, D = (build_chunk(text) for text in ["a", "b", "c", "d"])


# Fakes that record what the executor asks of them


class SpyEmbedder(FakeDenseEmbedder):
    """A fake embedder recording whether it embedded queries or passages."""

    def __init__(self) -> None:
        super().__init__(size=SIZE)
        self.calls: list[str] = []

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append("queries")
        return super().embed_queries(texts)

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append("passages")
        return super().embed_passages(texts)


class ScriptedIndex:
    """An index returning fixed hits per vector, whatever the query, and recording
    each search."""

    def __init__(self, hits: dict[Vector, list[Hit]]) -> None:
        self.hits = hits
        self.searches: list[tuple[Vector, Filter, int]] = []

    @property
    def schema(self) -> type[Schema]:
        return Chunks

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


def source(
    index: ScriptedIndex | InMemoryIndex,
    vector: Dense = Chunks.dense,
    *,
    depth: int = 10,
    embedder: FakeDenseEmbedder | None = None,
) -> Source:
    """Return a source named after `vector`, searching it in `index`."""
    return Source(
        name=vector.name,
        index=index,
        binding=DenseBinding(
            vector=vector, embedder=embedder or FakeDenseEmbedder(size=SIZE)
        ),
        depth=depth,
    )


def ids(passages: Iterable[object]) -> list[uuid.UUID]:
    """Return the chunk id of each passage, in order."""
    return [passage.chunk.id for passage in passages]  # type: ignore[attr-defined]


# Access: inside every source's search


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

    passages = execute(
        search("thời hạn nộp hồ sơ").retrieve(source(index)).take(5),
        access=PUBLIC,
        validity=AlwaysValid(),
    )

    assert ids(passages) == [visible.id]


def test_every_source_searches_with_the_access_filter() -> None:
    index = scripted(dense=[A], title=[B])

    execute(
        search("q")
        .retrieve(source(index), source(index, Chunks.title))
        .merge(Rrf())
        .take(5),
        access=PUBLIC,
        validity=AlwaysValid(),
    )

    assert [where for _, where, _ in index.searches] == [PUBLIC, PUBLIC]


def test_everything_as_access_reaches_every_source_as_it_is() -> None:
    index = scripted(dense=[A], title=[B])

    execute(
        search("q")
        .retrieve(source(index), source(index, Chunks.title))
        .merge(Rrf())
        .take(5),
        access=Everything(),
        validity=AlwaysValid(),
    )

    assert [where for _, where, _ in index.searches] == [Everything(), Everything()]


class Other(Schema):
    group = One(str, indexed=True)


@pytest.mark.parametrize(
    "access",
    [
        pytest.param(Other.group.eq("public"), id="same name, another schema"),
        pytest.param(
            and_(PUBLIC, not_(Other.group.eq("hr"))), id="nested after a fitting part"
        ),
    ],
)
def test_access_that_does_not_fit_a_sources_schema_fails_before_any_model_is_called(
    access: Filter,
) -> None:
    embedder = SpyEmbedder()
    index = scripted(dense=[A])

    with pytest.raises(SchemaMismatchError, match="not in schema Chunks"):
        execute(
            search("q").retrieve(source(index, embedder=embedder)).take(5),
            access=access,
            validity=AlwaysValid(),
        )

    assert embedder.calls == []
    assert index.searches == []


# Retrieve


def test_each_source_searches_its_own_vector_to_its_depth() -> None:
    index = scripted()

    execute(
        search("q")
        .retrieve(source(index, depth=50), source(index, Chunks.title, depth=20))
        .merge(Rrf())
        .take(5),
        access=PUBLIC,
        validity=AlwaysValid(),
    )

    assert [(vector, limit) for vector, _, limit in index.searches] == [
        (Chunks.dense, 50),
        (Chunks.title, 20),
    ]


def test_the_question_is_embedded_as_a_query_not_as_a_passage() -> None:
    embedder = SpyEmbedder()

    execute(
        search("q").retrieve(source(scripted(), embedder=embedder)).take(5),
        access=PUBLIC,
        validity=AlwaysValid(),
    )

    assert embedder.calls == ["queries"]


def test_an_embedder_giving_no_row_for_the_question_is_refused() -> None:
    class Silent(FakeDenseEmbedder):
        def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
            return []

    with pytest.raises(ValueError, match="returned 0 rows for one question"):
        execute(
            search("q")
            .retrieve(source(scripted(), embedder=Silent(size=SIZE)))
            .take(5),
            access=PUBLIC,
            validity=AlwaysValid(),
        )


def test_one_source_gives_its_hits_in_order_each_found_by_it() -> None:
    passages = execute(
        search("q").retrieve(source(scripted(dense=[A, B, C]))).take(5),
        access=PUBLIC,
        validity=AlwaysValid(),
    )

    assert ids(passages) == [A.id, B.id, C.id]
    assert [passage.trail for passage in passages] == [
        (Found(source="dense", rank=1, score=0.9),),
        (Found(source="dense", rank=2, score=0.8),),
        (Found(source="dense", rank=3, score=0.7),),
    ]


# Validity: where candidates enter, and again before take


def test_a_chunk_validity_drops_at_retrieve_never_appears() -> None:
    passages = execute(
        search("q").retrieve(source(scripted(dense=[A, B, C]))).take(5),
        access=PUBLIC,
        validity=SpyValidity(B),
    )

    assert ids(passages) == [A.id, C.id]


def test_found_ranks_are_the_sources_own_whatever_validity_drops() -> None:
    passages = execute(
        search("q").retrieve(source(scripted(dense=[A, B, C]))).take(5),
        access=PUBLIC,
        validity=SpyValidity(B),
    )

    assert [passage.trail[0].rank for passage in passages] == [1, 3]


def test_a_stale_chunk_is_never_reranked() -> None:
    reranker = SpyReranker()

    execute(
        search("q").retrieve(source(scripted(dense=[A, B]))).rerank(reranker).take(5),
        access=PUBLIC,
        validity=SpyValidity(B),
    )

    assert reranker.texts == [[A.embedded_text]]


def test_a_chunk_turning_invalid_during_the_search_never_appears() -> None:
    # Valid when retrieved, deleted before the final gate
    passages = execute(
        search("q").retrieve(source(scripted(dense=[A, B]))).take(5),
        access=PUBLIC,
        validity=SpyValidity(A, from_call=2),
    )

    assert ids(passages) == [B.id]


def test_validity_is_asked_once_per_gate_however_many_sources() -> None:
    validity = SpyValidity()
    index = scripted(dense=[A, B], title=[C])

    execute(
        search("q")
        .retrieve(source(index), source(index, Chunks.title))
        .merge(Rrf())
        .rerank(SpyReranker())
        .take(5),
        access=PUBLIC,
        validity=validity,
    )

    assert len(validity.calls) == 2


def test_a_chunk_several_sources_found_is_asked_about_once() -> None:
    validity = SpyValidity()
    index = scripted(dense=[A, B], title=[B, A])

    execute(
        search("q")
        .retrieve(source(index), source(index, Chunks.title))
        .merge(Rrf())
        .take(5),
        access=PUBLIC,
        validity=validity,
    )

    assert sorted(validity.calls[0]) == sorted([A.id, B.id])


def test_validity_isnt_asked_when_nothing_was_found() -> None:
    validity = SpyValidity()

    passages = execute(
        search("q").retrieve(source(scripted())).take(5),
        access=PUBLIC,
        validity=validity,
    )

    assert passages == []
    assert validity.calls == []


# Merge


def test_merging_ranks_by_rrf_over_the_lists_that_have_each_chunk() -> None:
    index = scripted(dense=[A, B], title=[C, A])

    passages = execute(
        search("q")
        .retrieve(source(index), source(index, Chunks.title))
        .merge(Rrf(k=60))
        .take(5),
        access=PUBLIC,
        validity=AlwaysValid(),
    )

    assert ids(passages) == [A.id, C.id, B.id]
    assert [passage.score for passage in passages] == pytest.approx(
        [1 / 61 + 1 / 62, 1 / 61, 1 / 62]
    )


def test_a_merged_trail_holds_each_source_that_found_it_then_the_fusion() -> None:
    index = scripted(dense=[A, B], title=[C, A])

    [first, *_] = execute(
        search("q")
        .retrieve(source(index), source(index, Chunks.title))
        .merge(Rrf(k=60))
        .take(5),
        access=PUBLIC,
        validity=AlwaysValid(),
    )

    assert first.trail == (
        Found(source="dense", rank=1, score=0.9),
        Found(source="title", rank=2, score=0.8),
        # Summed as the executor does, in list order, so the floats match exactly
        Fused(rank=1, score=1 / 61 + 1 / 62),
    )


def test_fusion_ties_keep_the_order_of_first_appearance() -> None:
    index = scripted(dense=[A], title=[B])

    passages = execute(
        search("q")
        .retrieve(source(index), source(index, Chunks.title))
        .merge(Rrf())
        .take(5),
        access=PUBLIC,
        validity=AlwaysValid(),
    )

    assert ids(passages) == [A.id, B.id]


# Rerank


def test_a_rerank_scores_every_candidate_by_its_embedded_text() -> None:
    reranker = SpyReranker()

    execute(
        search("q")
        .retrieve(source(scripted(dense=[A, B, C])))
        .rerank(reranker)
        .take(1),
        access=PUBLIC,
        validity=AlwaysValid(),
    )

    assert reranker.texts == [[A.embedded_text, B.embedded_text, C.embedded_text]]


def test_a_rerank_orders_by_its_scores_and_adds_them_to_the_trail() -> None:
    reranker = SpyReranker({C.embedded_text: 0.95, A.embedded_text: 0.5})

    passages = execute(
        search("q")
        .retrieve(source(scripted(dense=[A, B, C])))
        .rerank(reranker)
        .take(5),
        access=PUBLIC,
        validity=AlwaysValid(),
    )

    assert ids(passages) == [C.id, A.id, B.id]
    assert [passage.trail[-1] for passage in passages] == [
        Reranked(rank=1, score=0.95),
        Reranked(rank=2, score=0.5),
        Reranked(rank=3, score=0.0),
    ]


def test_rerank_ties_keep_their_earlier_order() -> None:
    passages = execute(
        search("q")
        .retrieve(source(scripted(dense=[A, B, C])))
        .rerank(SpyReranker())
        .take(5),
        access=PUBLIC,
        validity=AlwaysValid(),
    )

    assert ids(passages) == [A.id, B.id, C.id]


def test_a_reranker_giving_a_score_that_isnt_finite_is_refused() -> None:
    reranker = SpyReranker({A.embedded_text: math.nan})

    with pytest.raises(ValueError, match="finite"):
        execute(
            search("q").retrieve(source(scripted(dense=[A]))).rerank(reranker).take(5),
            access=PUBLIC,
            validity=AlwaysValid(),
        )


# Take


def test_take_keeps_the_best_limit() -> None:
    passages = execute(
        search("q").retrieve(source(scripted(dense=[A, B, C, D]))).take(2),
        access=PUBLIC,
        validity=AlwaysValid(),
    )

    assert ids(passages) == [A.id, B.id]


def test_take_fills_its_limit_past_a_chunk_the_final_gate_drops() -> None:
    passages = execute(
        search("q").retrieve(source(scripted(dense=[A, B, C]))).take(2),
        access=PUBLIC,
        validity=SpyValidity(A, from_call=2),
    )

    assert ids(passages) == [B.id, C.id]


# Statements built at run time


def test_a_statement_out_of_order_fails_before_any_model_or_index_is_used() -> None:
    embedder = SpyEmbedder()
    index = scripted(dense=[A])
    statement = Search(
        question="q",
        stages=(Take(limit=5), Retrieve(sources=(source(index, embedder=embedder),))),
    )

    with pytest.raises(StatementError):
        execute(statement, access=PUBLIC, validity=AlwaysValid())
    assert embedder.calls == []
    assert index.searches == []
