"""The engine: running search statements, and what they give back.

`Engine`, `VectorSource`, `LookupSource`, `search`, `source` and the stages callers build are exported
from the package's top level. Here are the names callers receive, annotate or
implement: the `Statement` a statement ends as, the `PreparedStatement` that
`Engine.prepare` returns, the `Passage` results with their trail entries, and the
`Validity` port with `AlwaysValid`.

No `execute` function is exported: `PreparedStatement.execute` is the one way to run
a statement, so none runs without being prepared.
"""

from salemdex_retrieval.engine.engine import PreparedStatement
from salemdex_retrieval.engine.results import (
    Found,
    Fused,
    Passage,
    Reranked,
    TrailEntry,
)
from salemdex_retrieval.engine.statements import Statement
from salemdex_retrieval.engine.validity import AlwaysValid, Validity

__all__ = [
    "AlwaysValid",
    "Found",
    "Fused",
    "Passage",
    "PreparedStatement",
    "Reranked",
    "Statement",
    "TrailEntry",
    "Validity",
]
