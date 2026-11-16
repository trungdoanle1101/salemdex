"""Embedders: the models turning chunks and questions into vector values.

Adapters needing an optional extra (Hugging Face) aren't imported here; import them
from their own module.
"""

from salemdex_retrieval.embedders.protocol import (
    DenseBinding,
    DenseEmbedder,
    VectorBinding,
    embed_chunks,
)

__all__ = ["DenseBinding", "DenseEmbedder", "VectorBinding", "embed_chunks"]
