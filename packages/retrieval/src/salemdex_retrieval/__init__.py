"""Salemdex retrieval: storing chunks in an index and searching them safely.

The app registers its sources and rerankers on one engine, prepares each statement
once, then runs it for every question, with a filter applied inside every source's
search and validity checked before passages are returned:

    engine = Engine(
        validity=validity,  # the app's adapter, or engine.AlwaysValid()
        sources={"dense": VectorSource(index=index, binding=binding)},
        rerankers={},
    )
    prepared = engine.prepare(search().retrieve(source("dense", depth=20)).take())
    passages = prepared.execute(question=question, limit=10, where=Everything())

Each public name has one import path:

- here: what callers build or call to use the package, the vocabulary at many call
  sites: schemas, chunks, filters, statements, the `Engine` and its `VectorSource`;
- its own module or subpackage: what callers receive, annotate, catch or implement,
  where the path tells the reader what it is: `errors.SchemaMismatchError`,
  `filters.Filter`, `engine.PreparedStatement`, `embedders.DenseEmbedder`,
  `index.Index`;
- a module of its own for what needs an extra (`index.qdrant`) or is only for tests
  (`embedders.fake`, `rerankers.fake`), so the import line says so.

No `execute` function is exported anywhere: `PreparedStatement.execute` is the one way
to run a statement, so none runs unchecked, and none without a filter.
"""

from salemdex_retrieval.chunks import Chunk, EmbeddedChunk
from salemdex_retrieval.engine.engine import Engine
from salemdex_retrieval.engine.sources import VectorSource
from salemdex_retrieval.engine.stages import Rrf, source
from salemdex_retrieval.engine.statements import search
from salemdex_retrieval.filters import Everything, and_, not_, or_
from salemdex_retrieval.schema import Many, One, Schema
from salemdex_retrieval.vectors import Dense

__all__ = [
    "Chunk",
    "Dense",
    "EmbeddedChunk",
    "Engine",
    "Everything",
    "Many",
    "One",
    "Rrf",
    "Schema",
    "VectorSource",
    "and_",
    "not_",
    "or_",
    "search",
    "source",
]
