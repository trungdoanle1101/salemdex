import uuid
from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from typing import Any, cast

from sqlalchemy import (
    ColumnElement,
    CursorResult,
    and_,
    delete,
    insert,
    or_,
    select,
    update,
)
from sqlalchemy.orm import Session

from kb_server.domain.credentials import ReadAccess
from kb_server.domain.documents import (
    Document,
    DocumentPage,
    DocumentPosition,
    DocumentStatus,
    DocumentVersion,
)
from kb_server.storage.models import (
    DocumentGroupRow,
    DocumentRow,
    DocumentVersionRow,
    StoredFileRow,
)


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(
        self,
        *,
        collection_id: uuid.UUID,
        title: str,
        status: DocumentStatus,
        group_ids: Iterable[uuid.UUID],
        file_id: uuid.UUID,
        original_filename: str,
        created_by: uuid.UUID,
    ) -> tuple[Document, DocumentVersion]:
        """A new document with version 1 and its read groups, written together."""
        document_id = uuid.uuid4()
        document_row = DocumentRow(
            id=document_id,
            collection_id=collection_id,
            title=title,
            status=status,
            current_version=1,
            created_by=created_by,
        )
        version_row = DocumentVersionRow(
            document_id=document_id,
            version=1,
            file_id=file_id,
            original_filename=original_filename,
            uploaded_by=created_by,
        )

        self._session.add_all([document_row, version_row])
        self._session.flush()

        group_rows = [
            {"document_id": document_id, "group_id": group_id}
            for group_id in set(group_ids)
        ]

        if group_rows:
            self._session.execute(insert(DocumentGroupRow), group_rows)

        return to_document(document_row), to_document_version(version_row)

    def get(self, document_id: uuid.UUID) -> Document | None:
        row = self._session.get(DocumentRow, document_id)
        return to_document(row) if row is not None else None

    def list_page(
        self,
        *,
        collection_id: uuid.UUID | None,
        start_after: DocumentPosition | None,
        access: ReadAccess | None,
        limit: int,
    ) -> DocumentPage:
        """Return a page of documents, newest first, leaving out those being deleted.

        Args:
            collection_id: Only this collection's documents; None for every one.
            start_after: Where the previous page ended; None for the first page.
            access: What the asker may read; None for everything.
            limit: The most documents on the page.

        Returns:
            The page, and where the next one starts if more documents follow.
        """
        statement = (
            select(DocumentRow)
            .where(DocumentRow.status != "deleting")
            .order_by(DocumentRow.created_at.desc(), DocumentRow.id.desc())
        )

        if collection_id is not None:
            statement = statement.where(DocumentRow.collection_id == collection_id)

        if start_after is not None:
            statement = statement.where(
                or_(
                    DocumentRow.created_at < start_after.created_at,
                    and_(
                        DocumentRow.created_at == start_after.created_at,
                        DocumentRow.id < start_after.id,
                    ),
                )
            )

        if access is not None:
            shares_group = select(DocumentGroupRow.document_id).where(
                DocumentGroupRow.group_id.in_(access.group_ids)
            )
            statement = statement.where(
                or_(
                    DocumentRow.id.in_(shares_group),
                    DocumentRow.collection_id.in_(access.managed_collection_ids),
                )
            )

        rows = self._session.scalars(statement.limit(limit + 1)).all()

        has_more = len(rows) > limit
        rows = rows[:limit]
        next_position = (
            DocumentPosition(created_at=rows[-1].created_at, id=rows[-1].id)
            if has_more
            else None
        )

        return DocumentPage(
            items=[to_document(row) for row in rows], next_position=next_position
        )

    def mark_deleting(self, document_id: uuid.UUID) -> bool:
        """Mark the document to be deleted.

        Returns true if the document exists and not already marked for deletion.
        """
        result = cast(
            CursorResult[Any],
            self._session.execute(
                update(DocumentRow)
                .where(DocumentRow.id == document_id, DocumentRow.status != "deleting")
                .values(status="deleting")
            ),
        )
        return result.rowcount == 1

    def delete(self, document_id: uuid.UUID) -> bool:
        """Remove a document marked deleting, with its versions, groups and jobs (cascade). False if there's no such document."""
        result = cast(
            CursorResult[Any],
            self._session.execute(
                delete(DocumentRow).where(
                    DocumentRow.id == document_id, DocumentRow.status == "deleting"
                )
            ),
        )
        return result.rowcount == 1

    def get_version(
        self, document_id: uuid.UUID, version: int
    ) -> DocumentVersion | None:
        row = self._session.get(DocumentVersionRow, (document_id, version))
        return to_document_version(row) if row is not None else None

    def list_group_ids(self, document_id: uuid.UUID) -> set[uuid.UUID]:
        ids = self._session.scalars(
            select(DocumentGroupRow.group_id).where(
                DocumentGroupRow.document_id == document_id
            )
        ).all()
        return set(ids)

    def list_file_ids(self, document_id: uuid.UUID) -> set[uuid.UUID]:
        ids = self._session.scalars(
            select(DocumentVersionRow.file_id).where(
                DocumentVersionRow.document_id == document_id
            )
        ).all()
        return set(ids)

    def find_current_with_file(
        self, collection_id: uuid.UUID, sha256: str
    ) -> Document | None:
        """A document in the collection whose current version is this file."""
        row = self._session.scalar(
            select(DocumentRow)
            .join(
                DocumentVersionRow,
                and_(
                    DocumentRow.id == DocumentVersionRow.document_id,
                    DocumentRow.current_version == DocumentVersionRow.version,
                ),
            )
            .join(StoredFileRow, DocumentVersionRow.file_id == StoredFileRow.id)
            .where(
                DocumentRow.collection_id == collection_id,
                StoredFileRow.sha256 == sha256,
                DocumentRow.status != "deleting",
            )
        )
        return to_document(row) if row is not None else None

    def set_provenance(
        self, document_id: uuid.UUID, version: int, provenance: Mapping[str, Any]
    ) -> bool:
        """Replace the version's provenance. False if there is no such version."""
        result = self._session.execute(
            update(DocumentVersionRow)
            .where(
                DocumentVersionRow.document_id == document_id,
                DocumentVersionRow.version == version,
            )
            .values(provenance=dict(provenance))
        )

        return cast(CursorResult[Any], result).rowcount == 1

    def set_indexed_version(self, document_id: uuid.UUID, version: int) -> bool:
        """Record `version` as the document's version that search shows.

        Args:
            document_id: The document.
            version: The version now fully in the index.

        Returns:
            False if there is no such document, e.g. it was deleted meanwhile.
        """
        result = self._session.execute(
            update(DocumentRow)
            .where(DocumentRow.id == document_id)
            .values(indexed_version=version)
        )
        return cast(CursorResult[Any], result).rowcount == 1

    def served_versions(
        self, document_ids: AbstractSet[uuid.UUID]
    ) -> dict[uuid.UUID, int]:
        """Return the indexed version of each of these documents that search may show.

        One query, whatever the number of ids: search's validity check calls it at
        every gate.

        Args:
            document_ids: The documents of the chunks being checked.

        Returns:
            `indexed_version` by document id, for active documents that have one.
            Drafts, documents being deleted, documents never indexed and unknown ids
            are left out: a missing id means none of that document is shown.
        """
        if not document_ids:
            return {}

        rows = self._session.execute(
            select(DocumentRow.id, DocumentRow.indexed_version).where(
                *_served(document_ids)
            )
        )
        return {row.id: row.indexed_version for row in rows}

    def served_titles(
        self, document_ids: AbstractSet[uuid.UUID]
    ) -> dict[uuid.UUID, str]:
        """Return the title of each of these documents that search may show.

        One query, whatever the number of ids: search reads every passage's title at
        once, from the database, so a rename never touches the index.

        Args:
            document_ids: The documents of the passages being shown.

        Returns:
            The title by document id, for the same documents as `served_versions`:
            active ones that have an indexed version.
        """
        if not document_ids:
            return {}

        rows = self._session.execute(
            select(DocumentRow.id, DocumentRow.title).where(*_served(document_ids))
        )
        return {row.id: row.title for row in rows}


def to_document(row: DocumentRow) -> Document:
    return Document(
        id=row.id,
        collection_id=row.collection_id,
        title=row.title,
        status=row.status,
        current_version=row.current_version,
        indexed_version=row.indexed_version,
        created_by=row.created_by,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def to_document_version(row: DocumentVersionRow) -> DocumentVersion:
    return DocumentVersion(
        document_id=row.document_id,
        version=row.version,
        file_id=row.file_id,
        original_filename=row.original_filename,
        provenance=row.provenance,
        uploaded_by=row.uploaded_by,
        created_at=row.created_at,
    )


def _served(
    document_ids: AbstractSet[uuid.UUID],
) -> tuple[ColumnElement[bool], ...]:
    """Return the conditions for the documents among these that search may show."""
    # One definition for every query of what search serves, so they can't drift apart
    return (
        DocumentRow.status == "active",
        DocumentRow.indexed_version.is_not(None),
        DocumentRow.id.in_(document_ids),
    )
