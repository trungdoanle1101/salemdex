"""The validity port: which chunks may be served now, decided by the app.

Whether a chunk may be served can depend on facts outside the index, which change
without the chunk being rewritten: in the server, whether its document is still
active and whether its version is the one currently served. The index can't filter
on those, so the executor asks the app after each search, through this port, at
every gate: where candidates enter, and again before `Take`.

The package defines only the shape (ports and adapters): the app's adapter reads its
own database, and nothing here imports one.
"""

import uuid
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from typing import Protocol

from salemdex_retrieval.chunks import Chunk


class Validity(Protocol):
    """A check of which chunks may be served now, whoever is asking.

    The engine holds it for its whole life: it doesn't depend on the asker, whose
    limits are the session's access filter.
    """

    def valid_ids(self, chunks: Sequence[Chunk]) -> AbstractSet[uuid.UUID]:
        """Return the ids of the chunks that may be served now.

        Called once per gate, with every chunk the gate checks, so an adapter can
        answer with one query. Only returned ids are kept: an adapter can narrow the
        results but never widen them, and one that returns nothing serves nothing.

        Args:
            chunks: The chunks to check; the adapter reads its own schema's fields
                (e.g. the document and version) from them.

        Returns:
            The ids to keep; ids of chunks not given are ignored.
        """
        ...


class AlwaysValid:
    """A validity check that keeps every chunk, for apps with nothing to check.

    Passing it says so explicitly, where leaving the slot empty would hide it.
    """

    def valid_ids(self, chunks: Sequence[Chunk]) -> set[uuid.UUID]:
        """Return every chunk's id.

        Args:
            chunks: The chunks to check.

        Returns:
            The id of each chunk.
        """
        return {chunk.id for chunk in chunks}

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
