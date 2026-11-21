import pytest

from salemdex_retrieval.embedders.fake import FakeDenseEmbedder
from salemdex_retrieval.embedders.protocol import DenseBinding
from salemdex_retrieval.engine.sources import VectorSource
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


def test_a_vector_source_keeps_its_index_and_binding() -> None:
    binding = DenseBinding(vector=Chunks.dense, embedder=EMBEDDER)

    registered = VectorSource(index=CHUNKS, binding=binding)

    assert registered.index is CHUNKS
    assert registered.binding is binding


def test_a_vector_source_refuses_a_vector_its_index_doesnt_declare() -> None:
    # Same name, another schema's vector: refused when the engine is built, so no
    # question is ever embedded for it
    with pytest.raises(
        SchemaMismatchError,
        match=r"vector 'dense' \(another schema's vector\) not in schema Chunks",
    ):
        VectorSource(
            index=CHUNKS,
            binding=DenseBinding(vector=Entries.dense, embedder=EMBEDDER),
        )
