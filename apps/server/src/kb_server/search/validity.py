import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.orm import Session, sessionmaker

from kb_server.domain.chunks import Guarded
from kb_server.storage.repositories.documents import DocumentRepository
from salemdex_retrieval import Chunk


@dataclass(frozen=True)
class ServedVersionsOnly:
    session_factory: sessionmaker[Session]

    def valid_ids(self, chunks: Sequence[Chunk]) -> set[uuid.UUID]:
        if not chunks:
            return set()
        with self.session_factory.begin() as session:
            documents = DocumentRepository(session)
            served = documents.served_versions(
                {chunk[Guarded.document_id] for chunk in chunks}
            )
            return {
                chunk.id
                for chunk in chunks
                if chunk[Guarded.document_version]
                == served.get(chunk[Guarded.document_id])
            }
