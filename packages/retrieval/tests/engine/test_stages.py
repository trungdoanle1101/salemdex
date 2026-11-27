from typing import Any

import pytest

from salemdex_retrieval.embedders.fake import FakeDenseEmbedder
from salemdex_retrieval.embedders.protocol import DenseBinding
from salemdex_retrieval.engine.sources import VectorScan
from salemdex_retrieval.engine.stages import (
    Merge,
    Rerank,
    Retrieve,
    Rrf,
    Scan,
    Take,
    Window,
    scan,
)
from salemdex_retrieval.schema import Many, One, Schema
from salemdex_retrieval.vectors import Dense

BAD_COUNTS = [
    pytest.param(0, id="zero"),
    pytest.param(-1, id="negative"),
    pytest.param(True, id="a bool"),
]

NAMES_WITHOUT_TEXT = ["", "   ", "\n\t"]


# The fields a window names
class Chunks(Schema):
    document = One(str, indexed=True)
    version = One(int)
    position = One(int)
    title = One(str)
    headings = Many(str)


# Scan


def test_a_scan_keeps_its_name_and_depth() -> None:
    dense = Scan(name="dense", depth=50)

    assert dense.name == "dense"
    assert dense.depth == 50


def test_the_scan_function_builds_the_same_scan() -> None:
    assert scan("dense", depth=50) == Scan(name="dense", depth=50)


def test_two_scans_with_the_same_name_and_depth_are_equal() -> None:
    # Plain data: a statement built twice, or read back from a log, compares equal
    assert scan("dense", depth=50) == scan("dense", depth=50)
    assert scan("dense", depth=50) != scan("dense", depth=20)


@pytest.mark.parametrize("depth", BAD_COUNTS)
def test_a_scan_refuses_a_depth_that_isnt_a_positive_int(depth: Any) -> None:
    with pytest.raises(ValueError, match="depth must be a positive integer"):
        scan("dense", depth=depth)


@pytest.mark.parametrize("name", NAMES_WITHOUT_TEXT)
def test_a_scan_refuses_a_name_without_text(name: str) -> None:
    with pytest.raises(ValueError, match="name"):
        scan(name, depth=20)


def test_a_scan_strips_whitespace_from_either_end_of_its_name() -> None:
    assert scan(" dense\n", depth=20).name == "dense"


def test_a_scan_refuses_a_name_that_isnt_a_str() -> None:
    with pytest.raises(TypeError, match="name"):
        scan(None, depth=20)  # type: ignore[arg-type]


# Retrieve


def test_a_retrieve_keeps_its_scans_in_order() -> None:
    dense, title = scan("dense", depth=50), scan("title", depth=20)

    retrieve = Retrieve(scans=(dense, title))

    assert retrieve.scans == (dense, title)


def test_a_retrieve_refuses_no_scans() -> None:
    with pytest.raises(ValueError, match="at least one scan"):
        Retrieve(scans=())


def test_a_retrieve_refuses_two_scans_with_the_same_name() -> None:
    # The trail names each scan, so two of one name couldn't be told apart
    with pytest.raises(ValueError, match="share a name: 'dense'"):
        Retrieve(scans=(scan("dense", depth=50), scan("dense", depth=50)))


def test_a_retrieve_refuses_one_name_twice_even_at_different_depths() -> None:
    with pytest.raises(ValueError, match="share a name: 'dense'"):
        Retrieve(scans=(scan("dense", depth=20), scan("dense", depth=50)))


def test_a_retrieve_refuses_names_that_differ_only_by_surrounding_whitespace() -> None:
    with pytest.raises(ValueError, match="share a name: 'dense'"):
        Retrieve(scans=(scan("dense", depth=20), scan("dense ", depth=20)))


def test_a_retrieve_refuses_a_registered_scan_in_place_of_a_name() -> None:
    # A statement holds names only; the engine holds what they stand for
    class Chunks(Schema):
        dense = Dense(size=8, distance="cosine")

    registered = VectorScan(
        binding=DenseBinding(vector=Chunks.dense, embedder=FakeDenseEmbedder(size=8)),
    )

    with pytest.raises(TypeError, match=r"scans\[1\] is VectorScan"):
        Retrieve(scans=(scan("dense", depth=20), registered))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "not_a_scan",
    [
        pytest.param("dense", id="a bare name"),
        pytest.param(None, id="None"),
    ],
)
def test_a_retrieve_refuses_something_that_isnt_a_scan(not_a_scan: Any) -> None:
    with pytest.raises(TypeError, match="expected a Scan"):
        Retrieve(scans=(scan("dense", depth=20), not_a_scan))


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


def test_a_take_has_no_window_unless_given_one() -> None:
    assert Take().window is None


def test_a_take_keeps_its_window() -> None:
    window = Window(
        partition_by=(Chunks.document,), order_by=Chunks.position, chunks_each_side=1
    )

    assert Take(window=window).window is window


def test_a_take_refuses_a_window_that_isnt_a_window() -> None:
    # Found here, not when the statement is prepared or run
    with pytest.raises(TypeError, match="window"):
        Take(window=(Chunks.document,))  # type: ignore[arg-type]


# Window


def test_a_window_keeps_its_settings() -> None:
    window = Window(
        partition_by=(Chunks.document, Chunks.version),
        order_by=Chunks.position,
        chunks_each_side=2,
    )

    assert window.partition_by == (Chunks.document, Chunks.version)
    assert window.order_by is Chunks.position
    assert window.chunks_each_side == 2


def test_a_window_may_have_no_partition() -> None:
    # The whole index is then one sequence
    window = Window(partition_by=(), order_by=Chunks.position, chunks_each_side=1)

    assert window.partition_by == ()


def test_two_windows_with_the_same_settings_are_equal() -> None:
    # Plain data, like every stage: a statement read back from a log compares equal
    def build(each_side: int) -> Window:
        return Window(
            partition_by=(Chunks.document,),
            order_by=Chunks.position,
            chunks_each_side=each_side,
        )

    assert build(1) == build(1)
    assert hash(build(1)) == hash(build(1))
    assert build(1) != build(2)


def test_a_window_refuses_partition_fields_given_as_a_list() -> None:
    # A list would leave a frozen stage holding something mutable and unhashable
    with pytest.raises(TypeError, match="partition_by must be a tuple"):
        Window(
            partition_by=[Chunks.document],  # type: ignore[arg-type]
            order_by=Chunks.position,
            chunks_each_side=1,
        )


def test_a_window_refuses_a_partition_field_with_many_values() -> None:
    # A chunk shares a partition with its hit by equal values, one per field
    with pytest.raises(TypeError, match="found a Many instead"):
        Window(
            partition_by=(Chunks.headings,),  # type: ignore[arg-type]
            order_by=Chunks.position,
            chunks_each_side=1,
        )


@pytest.mark.parametrize(
    ("order_by", "found"),
    [
        pytest.param(Chunks.title, r"One\[str\]", id="a field of str"),
        pytest.param(Chunks.headings, r"Many\[str\]", id="a field of many values"),
        pytest.param("position", "str", id="a field's name"),
    ],
)
def test_a_window_refuses_an_order_that_isnt_a_field_of_one_int(
    order_by: Any, found: str
) -> None:
    # Neighbours are found by distance in positions, which needs integers
    with pytest.raises(TypeError, match=rf"One\[int\], got {found}"):
        Window(partition_by=(), order_by=order_by, chunks_each_side=1)


@pytest.mark.parametrize("each_side", BAD_COUNTS)
def test_a_window_refuses_a_count_each_side_that_isnt_a_positive_int(
    each_side: Any,
) -> None:
    with pytest.raises(ValueError, match="chunks_each_side must be a positive integer"):
        Window(partition_by=(), order_by=Chunks.position, chunks_each_side=each_side)
