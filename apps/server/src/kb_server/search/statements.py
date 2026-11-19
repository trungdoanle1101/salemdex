from dataclasses import dataclass
from typing import Protocol

from salemdex_retrieval import Source, search
from salemdex_retrieval.engine import Search
from salemdex_retrieval.rerankers import Reranker


class BuildStatement(Protocol):
    def __call__(self, question: str, /, *, limit: int) -> Search: ...


@dataclass(frozen=True, kw_only=True)
class DenseThenRerank:
    source: Source
    reranker: Reranker

    def __call__(self, question: str, /, *, limit: int) -> Search:
        return search(question).retrieve(self.source).rerank(self.reranker).take(limit)
