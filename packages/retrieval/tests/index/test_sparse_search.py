"""Searching a sparse vector, run against each backend in turn.

What Qdrant does is the reference: scores are the dot product over the ids the query
and the chunk share, a chunk sharing none isn't a hit, and with `modifier="idf"` each
query weight is first multiplied by its id's IDF over every stored chunk.

Until `QdrantIndex` stores sparse vectors, its backends are expected to fail here
(strictly, so the marks go when it does).
"""

import math
import uuid
from collections.abc import Callable, Mapping, Sequence

import pytest

from salemdex_retrieval.chunks import Chunk, EmbeddedChunk
from salemdex_retrieval.filters import Everything
from salemdex_retrieval.index import Index
from salemdex_retrieval.schema import One, Schema
from salemdex_retrieval.vectors import Sparse

# Invented weights only


class Terms(Schema):
    position = One(int)
    learned = Sparse(modifier="none")
    counted = Sparse(modifier="idf")


type Rows = Sequence[Mapping[int, float]]


@pytest.fixture
def index_of(
    new_index: Callable[[type[Schema]], Index], request: pytest.FixtureRequest
) -> Callable[[Rows], Index]:
    """Return a function opening an index of Terms, one chunk per row, at its position.

    Each chunk has the row as its value for both vectors, so a test picks the modifier
    by the vector it searches.
    """
    if str(request.node.callspec.params["new_index"]).startswith("qdrant"):
        request.applymarker(
            pytest.mark.xfail(
                strict=True, reason="QdrantIndex doesn't store sparse vectors yet"
            )
        )

    def open_with(rows: Rows) -> Index:
        index = new_index(Terms)
        index.put(
            EmbeddedChunk(
                chunk=Chunk(
                    id=uuid.UUID(int=position + 1),
                    text="Invented text",
                    field_values=[Terms.position.value(position)],
                ),
                vector_values=[Terms.learned.value(row), Terms.counted.value(row)],
            )
            for position, row in enumerate(rows)
        )
        return index

    return open_with


def scores(
    index: Index, query: Mapping[int, float], *, vector: Sparse
) -> dict[int, float]:
    """Return each hit's score by its chunk's position, best first, from every chunk."""
    hits = index.search(vector.query(query), where=Everything(), limit=10)
    return {hit.chunk[Terms.position]: hit.score for hit in hits}


def idf(document_frequency: int, *, chunk_count: int) -> float:
    """Return the IDF Qdrant gives an id held by `document_frequency` of the chunks."""
    return math.log(
        (chunk_count - document_frequency + 0.5) / (document_frequency + 0.5) + 1
    )


# Without a modifier: the weights as they are


def test_a_sparse_search_scores_by_the_dot_product_of_the_shared_ids(
    index_of: Callable[[Rows], Index],
) -> None:
    index = index_of([{1: 2.0, 2: 1.0}, {1: 1.0, 3: 4.0}])

    found = scores(index, {1: 1.0, 3: 0.5}, vector=Terms.learned)

    # 1×1 + 0.5×4 for the second; 1×2 for the first, whose id 2 the query lacks
    assert list(found) == [1, 0]
    assert list(found.values()) == pytest.approx([3.0, 2.0])


def test_a_chunk_sharing_no_id_with_the_query_is_not_a_hit(
    index_of: Callable[[Rows], Index],
) -> None:
    index = index_of([{1: 1.0}, {2: 1.0}, {}])

    found = scores(index, {1: 1.0}, vector=Terms.learned)

    assert list(found) == [0]


def test_a_chunk_sharing_an_id_whose_weight_is_zero_is_a_hit_scoring_zero(
    index_of: Callable[[Rows], Index],
) -> None:
    index = index_of([{1: 0.0}])

    assert scores(index, {1: 1.0}, vector=Terms.learned) == {0: 0.0}


# With modifier="idf": each query weight times its id's IDF


def test_idf_weighs_each_query_id_by_how_few_chunks_hold_it(
    index_of: Callable[[Rows], Index],
) -> None:
    # id 1 is in every chunk, id 2 only in the first
    index = index_of([{1: 1.0, 2: 1.0}, {1: 1.0}, {1: 1.0}, {1: 1.0}])

    found = scores(index, {1: 1.0, 2: 1.0}, vector=Terms.counted)

    common, rare = idf(4, chunk_count=4), idf(1, chunk_count=4)
    assert common < rare
    assert found[0] == pytest.approx(common + rare)
    assert found[1] == pytest.approx(common)


def test_idf_multiplies_the_query_weight_and_the_stored_one(
    index_of: Callable[[Rows], Index],
) -> None:
    index = index_of([{1: 3.0}, {2: 1.0}])

    found = scores(index, {1: 2.0}, vector=Terms.counted)

    assert list(found) == [0]
    assert found[0] == pytest.approx(2.0 * idf(1, chunk_count=2) * 3.0)


def test_idf_counts_every_stored_chunk_not_only_those_where_matches(
    index_of: Callable[[Rows], Index],
) -> None:
    index = index_of([{1: 1.0}, {1: 1.0}, {2: 1.0}])
    query = Terms.counted.query({1: 1.0})

    everywhere = index.search(query, where=Everything(), limit=10)
    only_first = index.search(query, where=Terms.position.in_([0]), limit=10)

    assert [hit.chunk[Terms.position] for hit in only_first] == [0]
    assert only_first[0].score == pytest.approx(idf(2, chunk_count=3))
    assert only_first[0].score == pytest.approx(everywhere[0].score)


def test_a_chunk_with_no_weights_still_counts_towards_the_number_of_chunks(
    index_of: Callable[[Rows], Index],
) -> None:
    index = index_of([{1: 1.0}, {}])

    found = scores(index, {1: 1.0}, vector=Terms.counted)

    assert list(found) == [0]
    assert found[0] == pytest.approx(idf(1, chunk_count=2))
