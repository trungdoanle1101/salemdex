"""The statement search runs, and the names it gives what the engine registers.

The wiring registers the sources and rerankers under these names and prepares the
statement at startup; the tests do the same with the fake models, so they run the
statement the API runs, not a copy of it.
"""

from kb_server.domain.chunks import KbChunks
from salemdex_retrieval import search, source
from salemdex_retrieval.engine import Statement

# Named after what they do, not which model does it, so changing the model
# (`KB_EMBEDDER_SPEC`, `KB_RERANKER_SPEC`) leaves the statement as it is
DENSE = KbChunks.dense.name
CROSS_ENCODER = "cross_encoder"


def dense_then_rerank(*, depth: int) -> Statement:
    """Return the statement finding passages by their dense vector, then reranking them.

    Args:
        depth: How many candidates the dense search returns, all of which the
            reranker reads (`KB_RERANK_DEPTH`).

    Returns:
        The statement, to be prepared on an engine registering `DENSE` and
        `CROSS_ENCODER`.
    """
    return search().retrieve(source(DENSE, depth=depth)).rerank(CROSS_ENCODER).take()
