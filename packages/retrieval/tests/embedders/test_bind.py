"""`bind`: the binding a vector needs, chosen by its kind, its pairing checked by mypy.

Each `type: ignore` here must stay needed: strict mypy reports one that isn't, so the
tests fail to type-check if the overloads stop refusing a wrong pairing.
"""

from typing import Any

import pytest

from salemdex_retrieval.embedders import (
    BM25Embedder,
    DenseBinding,
    SparseBinding,
    bind,
)
from salemdex_retrieval.embedders.fake import FakeDenseEmbedder
from salemdex_retrieval.schema import One, Schema
from salemdex_retrieval.vectors import Dense, Sparse, Vector


class Docs(Schema):
    title = One(str)
    dense = Dense(size=8, distance="cosine")
    terms = Sparse(modifier="idf")


DENSE_EMBEDDER = FakeDenseEmbedder(size=8)
SPARSE_EMBEDDER = BM25Embedder(average_length=10.0)


def test_a_dense_vector_is_bound_in_a_dense_binding() -> None:
    binding = bind(Docs.dense, embedder=DENSE_EMBEDDER)

    assert binding == DenseBinding(vector=Docs.dense, embedder=DENSE_EMBEDDER)
    assert type(binding) is DenseBinding


def test_a_sparse_vector_is_bound_in_a_sparse_binding() -> None:
    binding = bind(Docs.terms, embedder=SPARSE_EMBEDDER)

    assert binding == SparseBinding(vector=Docs.terms, embedder=SPARSE_EMBEDDER)
    assert type(binding) is SparseBinding


def test_the_binding_holds_the_vector_and_embedder_given() -> None:
    binding = bind(Docs.dense, embedder=DENSE_EMBEDDER)

    assert binding.vector is Docs.dense
    assert binding.embedder is DENSE_EMBEDDER


@pytest.mark.parametrize(
    ("not_a_vector", "found"),
    [
        pytest.param(Docs.title, "One", id="a field"),
        pytest.param("dense", "str", id="a vector's name"),
        pytest.param(None, "NoneType", id="None"),
    ],
)
def test_bind_refuses_what_isnt_a_vector_saying_what_it_takes(
    not_a_vector: Any, found: str
) -> None:
    with pytest.raises(
        TypeError, match=f"bind takes a Dense or Sparse vector, got {found}$"
    ):
        bind(not_a_vector, embedder=DENSE_EMBEDDER)


def test_a_vector_paired_with_the_wrong_kind_of_embedder_is_refused_by_mypy() -> None:
    # Not at run time: a protocol can't be checked without calling the model, and a
    # wrong embedder fails on its first rows instead (`Dense.value`, `Sparse.value`)
    dense_with_sparse = bind(Docs.dense, embedder=SPARSE_EMBEDDER)  # type: ignore[call-overload]
    sparse_with_dense = bind(Docs.terms, embedder=DENSE_EMBEDDER)  # type: ignore[call-overload]

    assert type(dense_with_sparse) is DenseBinding
    assert type(sparse_with_dense) is SparseBinding


def test_every_kind_of_vector_can_be_bound() -> None:
    # `Vector` isn't a closed union, so mypy won't point at `bind` when a kind is
    # added: this does. A new kind needs its binding, an overload and a case in `bind`
    kinds = set(_subclasses(Vector))

    assert kinds == {Dense, Sparse}, "teach bind the new kinds of vector"


def _subclasses(cls: type) -> list[type]:
    """Return every subclass of `cls`, however deep."""
    direct: list[type] = cls.__subclasses__()
    return [*direct, *(deeper for sub in direct for deeper in _subclasses(sub))]
