"""Rerankers: the models scoring candidates against the question.

Adapters needing an optional extra (Hugging Face) aren't imported here; import them
from their own module. The fake for tests is in `rerankers.fake`.
"""

from salemdex_retrieval.rerankers.protocol import Reranker

__all__ = ["Reranker"]
