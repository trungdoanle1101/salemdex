import base64
import uuid
from datetime import datetime
from http import HTTPStatus
from typing import Annotated

from fastapi import Form, Query, Response, UploadFile
from fastapi.routing import APIRouter
from pydantic import BaseModel, StringConstraints

from kb_server.api.deps import CurrentPrincipal, DocumentServiceDep
from kb_server.api.groups import GroupOut
from kb_server.domain.documents import (
    Document,
    DocumentDetails,
    DocumentPosition,
    DocumentStatus,
)
from kb_server.domain.errors import InvalidRequest
from kb_server.domain.jobs import JobStatus

router = APIRouter(prefix="/documents", tags=["documents"])

Title = Annotated[str, StringConstraints(strip_whitespace=True, max_length=500)]


class FileOut(BaseModel):
    filename: str
    size: int
    media_type: str
    sha256: str


class IndexingJobOut(BaseModel):
    job_id: uuid.UUID
    status: JobStatus
    error: str | None


class DocumentOut(BaseModel):
    id: uuid.UUID
    collection_id: uuid.UUID
    title: str
    status: DocumentStatus
    current_version: int
    indexed_version: int | None
    created_at: datetime
    updated_at: datetime
    groups: list[GroupOut]
    file: FileOut
    job: IndexingJobOut | None

    @classmethod
    def from_details(cls, details: DocumentDetails) -> "DocumentOut":
        return DocumentOut(
            id=details.document.id,
            collection_id=details.document.collection_id,
            title=details.document.title,
            status=details.document.status,
            current_version=details.document.current_version,
            indexed_version=details.document.indexed_version,
            created_at=details.document.created_at,
            updated_at=details.document.updated_at,
            groups=[GroupOut.model_validate(group) for group in details.groups],
            file=FileOut(
                filename=details.version.original_filename,
                size=details.file.size,
                media_type=details.file.media_type,
                sha256=details.file.sha256,
            ),
            job=IndexingJobOut(
                job_id=details.job.id,
                status=details.job.status,
                error=details.job.error,
            )
            if details.job is not None
            else None,
        )


class DocumentSummaryOut(BaseModel):
    id: uuid.UUID
    title: str
    collection_id: uuid.UUID
    current_version: int
    indexed_version: int | None

    @classmethod
    def from_document(cls, document: Document) -> "DocumentSummaryOut":
        return cls(
            id=document.id,
            title=document.title,
            collection_id=document.collection_id,
            current_version=document.current_version,
            indexed_version=document.indexed_version,
        )


class DocumentPageOut(BaseModel):
    items: list[DocumentSummaryOut]
    next_cursor: str | None


@router.post("", status_code=HTTPStatus.CREATED)
def upload_document(
    principal: CurrentPrincipal,
    document_service: DocumentServiceDep,
    file: UploadFile,
    collection_id: Annotated[uuid.UUID, Form()],
    title: Annotated[Title | None, Form()] = None,
    group_ids: Annotated[list[uuid.UUID] | None, Form()] = None,
    private: Annotated[bool, Form()] = False,
) -> DocumentOut:
    details = document_service.upload_document(
        principal,
        collection_id=collection_id,
        content=file.file,
        filename=file.filename or "upload",
        title=title,
        group_ids=group_ids,
        private=private,
    )
    return DocumentOut.from_details(details)


@router.get("/{document_id}")
def get_document(
    document_id: uuid.UUID,
    principal: CurrentPrincipal,
    document_service: DocumentServiceDep,
) -> DocumentOut:
    details = document_service.get_document(principal, document_id)
    return DocumentOut.from_details(details)


@router.get("")
def list_documents(
    principal: CurrentPrincipal,
    document_service: DocumentServiceDep,
    collection_id: uuid.UUID | None = None,
    cursor: Annotated[str, Query(max_length=200)] | None = None,
    limit: Annotated[int, Query(gt=0, le=100)] = 50,
) -> DocumentPageOut:
    page = document_service.list_documents(
        principal,
        collection_id=collection_id,
        start_after=decode_cursor(cursor),
        limit=limit,
    )
    return DocumentPageOut(
        items=[DocumentSummaryOut.from_document(doc) for doc in page.items],
        next_cursor=encode_cursor(page.next_position),
    )


@router.delete(
    "/{document_id}", status_code=HTTPStatus.ACCEPTED, response_class=Response
)
def delete_document(
    document_id: uuid.UUID,
    principal: CurrentPrincipal,
    document_service: DocumentServiceDep,
) -> None:
    document_service.delete_document(principal, document_id)


def encode_cursor(position: DocumentPosition | None) -> str | None:
    """Return `position` as an opaque cursor for clients to send back.

    Args:
        position: Where a page ended; None on the last page.

    Returns:
        base64url of the creation time and id, or None when `position` is None.
    """
    if position is None:
        return None
    # isoformat keeps the microseconds, so the cursor gives back the exact time stored
    text = f"{position.created_at.isoformat()}{_CURSOR_SEPARATOR}{position.id}"
    return base64.urlsafe_b64encode(text.encode()).decode("ascii")


def decode_cursor(cursor: str | None) -> DocumentPosition | None:
    """Return the position a cursor from `encode_cursor` holds.

    Args:
        cursor: A cursor from a previous page; None for the first page.

    Returns:
        The position to start after, or None when `cursor` is None.

    Raises:
        InvalidRequest: `cursor` wasn't made by `encode_cursor`.
    """
    if cursor is None:
        return None
    try:
        # validate=True refuses stray characters instead of silently dropping them
        text = base64.b64decode(cursor, altchars=b"-_", validate=True).decode()
        created_at_text, id_text = text.split(_CURSOR_SEPARATOR)
        created_at = datetime.fromisoformat(created_at_text)
        document_id = uuid.UUID(id_text)
    except ValueError:  # binascii.Error and UnicodeDecodeError are ValueErrors too
        raise InvalidRequest("cursor is not valid") from None
    # A time without an offset would be compared in the database's own time zone
    if created_at.tzinfo is None:
        raise InvalidRequest("cursor is not valid")
    return DocumentPosition(created_at=created_at, id=document_id)


_CURSOR_SEPARATOR = "|"
