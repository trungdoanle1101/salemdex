"""Salemdex retrieval: storing chunks in an index and searching them safely.

Every search runs through a session, which adds the asker's access filter to each
source and checks validity before passages are returned:

    engine = Engine(validity=validity)  # the app's adapter, or engine.AlwaysValid()
    statement = search(question).retrieve(Source(…)).take(10)
    passages = engine.session(access=access).execute(statement)

Each public name has one import path:

- here: what callers build or call to use the package, the vocabulary at many call
  sites: schemas, chunks, filters, statements and the `Engine`;
- its own module or subpackage: what callers receive, annotate, catch or implement,
  where the path tells the reader what it is: `errors.SchemaMismatchError`,
  `filters.Filter`, `engine.Session`, `embedders.DenseEmbedder`, `index.Index`;
- a module of its own for what needs an extra (`index.qdrant`) or is only for tests
  (`embedders.fake`, `rerankers.fake`), so the import line says so.

`execute` isn't exported anywhere: a session is the one way to run a statement, so
no search can run without an access filter.
"""

from salemdex_retrieval.chunks import Chunk, EmbeddedChunk
from salemdex_retrieval.engine.session import Engine
from salemdex_retrieval.engine.stages import Rrf, Source
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
    "Source",
    "and_",
    "not_",
    "or_",
    "search",
]
