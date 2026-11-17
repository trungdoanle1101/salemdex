"""The engine: running search statements, and what they give back.

`Engine`, `search` and the stages callers build are exported from the package's top
level. Here are the names callers receive, annotate or implement: the `Session` that
`Engine.session` returns, the `Search` a statement ends as, the `Passage` results with
their trail entries, and the `Validity` port with `AlwaysValid`.

`execute` isn't exported: `Session.execute` is the one way to run a statement.
"""

from salemdex_retrieval.engine.results import (
    Found,
    Fused,
    Passage,
    Reranked,
    TrailEntry,
)
from salemdex_retrieval.engine.session import Session
from salemdex_retrieval.engine.statements import Search
from salemdex_retrieval.engine.validity import AlwaysValid, Validity

__all__ = [
    "AlwaysValid",
    "Found",
    "Fused",
    "Passage",
    "Reranked",
    "Search",
    "Session",
    "TrailEntry",
    "Validity",
]
