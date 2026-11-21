"""The sources an engine registers: what a statement's source names stand for.

A statement names its sources (`source("dense", depth=50)`); the engine holds, under
each name, what that source searches and with what: an index, and the embedder for
the question. Keeping these objects here, out of statements, is what lets a statement
stay plain data.
"""

from dataclasses import dataclass

from salemdex_retrieval.embedders import VectorBinding
from salemdex_retrieval.index import Index


@dataclass(frozen=True, kw_only=True)
class VectorSource:
    """A way of finding candidates: a vector of an index, searched with a question.

    The binding's embedder turns the question into a query for its vector; the index
    returns the nearest chunks the run's filter (`where`) lets through.

    Attributes:
        index: The index to search.
        binding: The vector to search, and the embedder for the question; the vector
            must be one of the index's schema.
    """

    index: Index
    binding: VectorBinding

    def __post_init__(self) -> None:
        """Check that the index's schema declares the vector.

        Raises:
            SchemaMismatchError: The vector isn't one of the index's schema, checked
                here, when the engine is built, so that no question is embedded for it.
        """
        self.index.schema.check_declares(self.binding.vector)
