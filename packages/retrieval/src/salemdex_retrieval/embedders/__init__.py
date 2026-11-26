"""Embedders: the models turning chunks and questions into vector values.

`BM25Embedder` needs no extra, so it's exported here with its helpers. Adapters needing
an optional extra (Hugging Face) aren't imported here; import them from their own
module.
"""

from salemdex_retrieval.embedders.bm25 import BM25Embedder, measure_average_length
from salemdex_retrieval.embedders.protocol import (
    DenseBinding,
    DenseEmbedder,
    SparseBinding,
    SparseEmbedder,
    VectorBinding,
    embed_chunks,
)
from salemdex_retrieval.embedders.words import split_words

__all__ = [
    "BM25Embedder",
    "DenseBinding",
    "DenseEmbedder",
    "SparseBinding",
    "SparseEmbedder",
    "VectorBinding",
    "embed_chunks",
    "measure_average_length",
    "split_words",
]
