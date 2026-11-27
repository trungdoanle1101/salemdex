from types import MappingProxyType

import pytest

from salemdex_retrieval.embedders.fake import FakeDenseEmbedder
from salemdex_retrieval.embedders.protocol import DenseBinding
from salemdex_retrieval.engine.sources import (
    LookupScan,
    RegisteredScan,
    SearchableIndex,
    VectorScan,
)
from salemdex_retrieval.errors import SchemaMismatchError
from salemdex_retrieval.index.in_memory import InMemoryIndex
from salemdex_retrieval.schema import One, Schema
from salemdex_retrieval.vectors import Dense


class Chunks(Schema):
    position = One(int)
    dense = Dense(size=8, distance="cosine")


class Entries(Schema):
    position = One(int)
    dense = Dense(size=8, distance="cosine")


CHUNKS = InMemoryIndex(schema=Chunks)
EMBEDDER = FakeDenseEmbedder(size=8)
DENSE = VectorScan(binding=DenseBinding(vector=Chunks.dense, embedder=EMBEDDER))
LOOKUP = LookupScan(parse_filter=lambda question: None)


def test_a_searchable_index_keeps_its_index_and_scans() -> None:
    searchable = SearchableIndex(index=CHUNKS, scans={"dense": DENSE, "lookup": LOOKUP})

    assert searchable.index is CHUNKS
    assert dict(searchable.scans) == {"dense": DENSE, "lookup": LOOKUP}


def test_a_searchable_index_refuses_a_vector_its_index_doesnt_declare() -> None:
    # Same name, another schema's vector: refused when the engine is built, so no
    # question is ever embedded for it
    other = VectorScan(binding=DenseBinding(vector=Entries.dense, embedder=EMBEDDER))

    with pytest.raises(
        SchemaMismatchError,
        match=r"vector 'dense' \(another schema's vector\) not in schema Chunks",
    ):
        SearchableIndex(index=CHUNKS, scans={"dense": other})


@pytest.mark.parametrize("name", ["", "  ", " dense", "dense\n"])
def test_a_searchable_index_refuses_a_scan_name_no_statement_could_give(
    name: str,
) -> None:
    # Statements strip their names, so an unstripped one could never be matched, and
    # stripping it here could merge two scans into one
    with pytest.raises(ValueError, match="scan's name"):
        SearchableIndex(index=CHUNKS, scans={name: DENSE})


def test_a_searchable_index_refuses_a_scan_name_that_isnt_a_str() -> None:
    with pytest.raises(TypeError, match="scan's name"):
        SearchableIndex(index=CHUNKS, scans={None: DENSE})  # type: ignore[dict-item]


def test_changing_the_mapping_given_doesnt_change_the_scans_kept() -> None:
    scans: dict[str, RegisteredScan] = {"dense": DENSE}
    searchable = SearchableIndex(index=CHUNKS, scans=scans)

    scans["lookup"] = LOOKUP

    assert isinstance(searchable.scans, MappingProxyType)
    assert list(searchable.scans) == ["dense"]
