import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, get_args

from kb_server.domain.files import StoredFile
from kb_server.domain.groups import Group
from kb_server.domain.jobs import IndexingJob

DocumentStatus = Literal["draft", "active", "deleting"]


DOCUMENT_STATUSES = get_args(DocumentStatus)


@dataclass(frozen=True, kw_only=True)
class Document:
    id: uuid.UUID
    collection_id: uuid.UUID
    title: str
    status: DocumentStatus
    current_version: int
    indexed_version: int | None
    created_by: uuid.UUID
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, kw_only=True)
class DocumentVersion:
    document_id: uuid.UUID
    version: int
    file_id: uuid.UUID
    original_filename: str
    provenance: Mapping[str, Any] | None
    uploaded_by: uuid.UUID
    created_at: datetime


@dataclass(frozen=True, kw_only=True)
class DocumentDetails:
    document: Document
    version: DocumentVersion  # current version
    file: StoredFile  # size, type, and hash
    groups: list[Group]
    job: IndexingJob | None  # latest job for the current version


@dataclass(frozen=True, kw_only=True)
class DocumentPosition:
    """A document's place in the newest-first list; the next page starts after it.

    Attributes:
        created_at: When the document was created; the list's order.
        id: The document's id, which breaks ties: one transaction's documents share
            `created_at`.
    """

    created_at: datetime
    id: uuid.UUID


@dataclass(frozen=True, kw_only=True)
class DocumentPage:
    """One page of a list of documents, newest first.

    Attributes:
        items: The documents on this page.
        next_position: Where the next page starts after; None on the last page.
    """

    items: list[Document]
    next_position: DocumentPosition | None
