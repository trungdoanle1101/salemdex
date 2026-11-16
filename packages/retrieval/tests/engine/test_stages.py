from typing import Any

import pytest

from salemdex_retrieval.embedders.fake import FakeDenseEmbedder
from salemdex_retrieval.embedders.protocol import DenseBinding
from salemdex_retrieval.engine.stages import Merge, Rerank, Retrieve, Rrf, Source, Take
from salemdex_retrieval.errors import SchemaMismatchError
from salemdex_retrieval.index.in_memory import InMemoryIndex
from salemdex_retrieval.rerankers.fake import FakeReranker
from salemdex_retrieval.schema import One, Schema
from salemdex_retrieval.vectors import Dense


class Chunks(Schema):
    position = One(int)
    dense = Dense(size=8, distance="cosine")
    title = Dense(size=8, distance="cosine")


class Entries(Schema):
    position = One(int)
    dense = Dense(size=8, distance="cosine")


CHUNKS = InMemoryIndex(schema=Chunks)
EMBEDDER = FakeDenseEmbedder(size=8)


def build_source(
    vector: Dense = Chunks.dense, *, index: InMemoryIndex = CHUNKS, depth: int = 20
) -> Source:
    """Return a source searching `vector` of `index`, keeping `depth` hits."""
    return Source(
        index=index, binding=DenseBinding(vector=vector, embedder=EMBEDDER), depth=depth
    )


BAD_COUNTS = [
    pytest.param(0, id="zero"),
    pytest.param(-1, id="negative"),
    pytest.param(True, id="a bool"),
]


# Source


def test_a_source_keeps_its_index_binding_and_depth() -> None:
    source = build_source(depth=50)

    assert source.index is CHUNKS
    assert source.binding.vector is Chunks.dense
    assert source.depth == 50


@pytest.mark.parametrize("depth", BAD_COUNTS)
def test_a_source_refuses_a_depth_that_isnt_a_positive_int(depth: Any) -> None:
    with pytest.raises(ValueError, match="depth must be a positive integer"):
        build_source(depth=depth)


def test_a_source_refuses_a_vector_its_index_doesnt_declare() -> None:
    with pytest.raises(
        SchemaMismatchError,
        match=r"vector 'dense' \(another schema's vector\) not in schema Chunks",
    ):
        build_source(Entries.dense)


# Retrieve


def test_a_retrieve_keeps_its_sources_in_order() -> None:
    dense, title = build_source(Chunks.dense), build_source(Chunks.title)

    retrieve = Retrieve(sources=(dense, title))

    assert retrieve.sources == (dense, title)


def test_a_retrieve_refuses_no_sources() -> None:
    with pytest.raises(ValueError, match="source"):
        Retrieve(sources=())


def test_a_retrieve_refuses_the_same_vector_of_the_same_index_twice() -> None:
    with pytest.raises(ValueError, match="'dense'"):
        Retrieve(sources=(build_source(), build_source()))


def test_a_retrieve_refuses_the_same_vector_twice_even_at_different_depths() -> None:
    with pytest.raises(ValueError, match="'dense'"):
        Retrieve(sources=(build_source(depth=20), build_source(depth=50)))


def test_a_retrieve_takes_two_vectors_of_one_index() -> None:
    Retrieve(sources=(build_source(Chunks.dense), build_source(Chunks.title)))


def test_a_retrieve_takes_the_same_vector_of_two_index_objects() -> None:
    # Two in-memory indexes hold different chunks, so neither repeats the other
    other = InMemoryIndex(schema=Chunks)

    Retrieve(sources=(build_source(), build_source(index=other)))


@pytest.mark.parametrize(
    "not_a_source",
    [
        pytest.param(Chunks.dense, id="a vector"),
        pytest.param(None, id="None"),
    ],
)
def test_a_retrieve_refuses_something_that_isnt_a_source(not_a_source: Any) -> None:
    with pytest.raises(TypeError, match="Source"):
        Retrieve(sources=(build_source(), not_a_source))


# Merge, Rrf


def test_a_merge_keeps_its_fusion() -> None:
    fusion = Rrf(k=60)

    assert Merge(fusion=fusion).fusion is fusion


@pytest.mark.parametrize("k", BAD_COUNTS)
def test_rrf_refuses_a_k_that_isnt_a_positive_int(k: Any) -> None:
    with pytest.raises(ValueError, match="k must be a positive integer"):
        Rrf(k=k)


# Rerank


def test_a_rerank_keeps_its_reranker_and_depth() -> None:
    reranker = FakeReranker()

    rerank = Rerank(reranker=reranker, depth=30)

    assert rerank.reranker is reranker
    assert rerank.depth == 30


@pytest.mark.parametrize("depth", BAD_COUNTS)
def test_a_rerank_refuses_a_depth_that_isnt_a_positive_int(depth: Any) -> None:
    with pytest.raises(ValueError, match="depth must be a positive integer"):
        Rerank(reranker=FakeReranker(), depth=depth)


# Take


def test_a_take_keeps_its_limit() -> None:
    assert Take(limit=10).limit == 10


@pytest.mark.parametrize("limit", BAD_COUNTS)
def test_a_take_refuses_a_limit_that_isnt_a_positive_int(limit: Any) -> None:
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        Take(limit=limit)
