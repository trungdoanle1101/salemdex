from typing import Any

import pytest

from salemdex_retrieval.embedders.fake import FakeDenseEmbedder
from salemdex_retrieval.embedders.protocol import DenseBinding
from salemdex_retrieval.engine.sources import VectorSource
from salemdex_retrieval.engine.stages import (
    Merge,
    Rerank,
    Retrieve,
    Rrf,
    Source,
    Take,
    source,
)
from salemdex_retrieval.index.in_memory import InMemoryIndex
from salemdex_retrieval.schema import Schema
from salemdex_retrieval.vectors import Dense

BAD_COUNTS = [
    pytest.param(0, id="zero"),
    pytest.param(-1, id="negative"),
    pytest.param(True, id="a bool"),
]

NAMES_WITHOUT_TEXT = ["", "   ", "\n\t"]


# Source


def test_a_source_keeps_its_name_and_depth() -> None:
    dense = Source(name="dense", depth=50)

    assert dense.name == "dense"
    assert dense.depth == 50


def test_the_source_function_builds_the_same_source() -> None:
    assert source("dense", depth=50) == Source(name="dense", depth=50)


def test_two_sources_with_the_same_name_and_depth_are_equal() -> None:
    # Plain data: a statement built twice, or read back from a log, compares equal
    assert source("dense", depth=50) == source("dense", depth=50)
    assert source("dense", depth=50) != source("dense", depth=20)


@pytest.mark.parametrize("depth", BAD_COUNTS)
def test_a_source_refuses_a_depth_that_isnt_a_positive_int(depth: Any) -> None:
    with pytest.raises(ValueError, match="depth must be a positive integer"):
        source("dense", depth=depth)


@pytest.mark.parametrize("name", NAMES_WITHOUT_TEXT)
def test_a_source_refuses_a_name_without_text(name: str) -> None:
    with pytest.raises(ValueError, match="name"):
        source(name, depth=20)


def test_a_source_strips_whitespace_from_either_end_of_its_name() -> None:
    assert source(" dense\n", depth=20).name == "dense"


def test_a_source_refuses_a_name_that_isnt_a_str() -> None:
    with pytest.raises(TypeError, match="name"):
        source(None, depth=20)  # type: ignore[arg-type]


# Retrieve


def test_a_retrieve_keeps_its_sources_in_order() -> None:
    dense, title = source("dense", depth=50), source("title", depth=20)

    retrieve = Retrieve(sources=(dense, title))

    assert retrieve.sources == (dense, title)


def test_a_retrieve_refuses_no_sources() -> None:
    with pytest.raises(ValueError, match="source"):
        Retrieve(sources=())


def test_a_retrieve_refuses_two_sources_with_the_same_name() -> None:
    # The trail names each source, so two of one name couldn't be told apart
    with pytest.raises(ValueError, match="share a name: 'dense'"):
        Retrieve(sources=(source("dense", depth=50), source("dense", depth=50)))


def test_a_retrieve_refuses_one_name_twice_even_at_different_depths() -> None:
    with pytest.raises(ValueError, match="share a name: 'dense'"):
        Retrieve(sources=(source("dense", depth=20), source("dense", depth=50)))


def test_a_retrieve_refuses_names_that_differ_only_by_surrounding_whitespace() -> None:
    with pytest.raises(ValueError, match="share a name: 'dense'"):
        Retrieve(sources=(source("dense", depth=20), source("dense ", depth=20)))


def test_a_retrieve_refuses_a_registered_source_in_place_of_a_name() -> None:
    # A statement holds names only; the engine holds what they stand for
    class Chunks(Schema):
        dense = Dense(size=8, distance="cosine")

    registered = VectorSource(
        index=InMemoryIndex(schema=Chunks),
        binding=DenseBinding(vector=Chunks.dense, embedder=FakeDenseEmbedder(size=8)),
    )

    with pytest.raises(TypeError, match=r"sources\[1\] is VectorSource"):
        Retrieve(sources=(source("dense", depth=20), registered))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "not_a_source",
    [
        pytest.param("dense", id="a bare name"),
        pytest.param(None, id="None"),
    ],
)
def test_a_retrieve_refuses_something_that_isnt_a_source(not_a_source: Any) -> None:
    with pytest.raises(TypeError, match="Source"):
        Retrieve(sources=(source("dense", depth=20), not_a_source))


# Merge, Rrf


def test_a_merge_keeps_its_fusion() -> None:
    fusion = Rrf(k=60)

    assert Merge(fusion=fusion).fusion is fusion


@pytest.mark.parametrize("k", BAD_COUNTS)
def test_rrf_refuses_a_k_that_isnt_a_positive_int(k: Any) -> None:
    with pytest.raises(ValueError, match="k must be a positive integer"):
        Rrf(k=k)


# Rerank


def test_a_rerank_keeps_its_rerankers_name() -> None:
    assert Rerank(reranker="cross_encoder").reranker == "cross_encoder"


def test_a_rerank_strips_whitespace_from_either_end_of_its_rerankers_name() -> None:
    assert Rerank(reranker=" cross_encoder\n").reranker == "cross_encoder"


@pytest.mark.parametrize("name", NAMES_WITHOUT_TEXT)
def test_a_rerank_refuses_a_rerankers_name_without_text(name: str) -> None:
    with pytest.raises(ValueError, match="reranker"):
        Rerank(reranker=name)


def test_a_rerank_refuses_a_rerankers_name_that_isnt_a_str() -> None:
    with pytest.raises(TypeError, match="reranker"):
        Rerank(reranker=None)  # type: ignore[arg-type]


# Take


def test_every_take_is_equal_since_it_holds_no_limit() -> None:
    # The limit is given each time the statement runs, not in the statement
    assert Take() == Take()
