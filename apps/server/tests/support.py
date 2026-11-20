"""Helpers for the tests. Fixtures live in conftest.py."""

import hashlib
import uuid
from datetime import datetime, timedelta
from typing import BinaryIO

from sqlalchemy import update
from sqlalchemy.orm import Session

from kb_server.domain.chunks import KbChunks
from kb_server.domain.credentials import Principal
from kb_server.domain.documents import Document
from kb_server.services.auth import AuthService
from kb_server.storage.files import FileStore
from kb_server.storage.models import StoredFileRow, UserRow
from salemdex_retrieval import Chunk
from salemdex_retrieval.embedders import DenseBinding
from salemdex_retrieval.embedders.fake import FakeDenseEmbedder

PASSWORD = "correct horse battery"
MAX_UPLOAD_MB = 1  # small, so the size limit is cheap to test

# What search tests ask; every chunk `publish` indexes holds its words
QUESTION = "nghỉ phép năm"
SEARCH_BINDING = DenseBinding(
    vector=KbChunks.dense, embedder=FakeDenseEmbedder(size=KbChunks.dense.size)
)


class FakeClock:
    """A clock that only moves when told to."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, by: timedelta) -> None:
        self.now += by


def sign_in(auth: AuthService, email: str) -> tuple[str, Principal]:
    """Log in as email (whose password is PASSWORD): the session token and its principal."""
    issued = auth.login(email, PASSWORD)
    assert issued is not None, f"could not log in as {email}"
    principal = auth.resolve(issued.token)
    assert principal is not None
    return issued.token, principal


def deactivate(session: Session, user_id: uuid.UUID) -> None:
    session.execute(
        update(UserRow).where(UserRow.id == user_id).values(is_active=False)
    )


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class RefusingStore:
    """A FileStore that fails like a full or read-only disk: on put, or on delete.

    Wraps a real store for everything else, and records the keys it was asked to put.
    """

    def __init__(
        self,
        store: FileStore,
        *,
        on_put: OSError | None = None,
        on_delete: OSError | None = None,
    ) -> None:
        self._store = store
        self._on_put = on_put
        self._on_delete = on_delete
        self.put_keys: list[str] = []

    def put(self, key: str, data: BinaryIO, *, size: int, media_type: str) -> None:
        self.put_keys.append(key)
        if self._on_put is not None:
            raise self._on_put
        self._store.put(key, data, size=size, media_type=media_type)

    def open(self, key: str) -> BinaryIO:
        return self._store.open(key)

    def exists(self, key: str) -> bool:
        return self._store.exists(key)

    def delete(self, key: str) -> None:
        if self._on_delete is not None:
            raise self._on_delete
        self._store.delete(key)

    def delete_folder(self, folder: str) -> int:
        if self._on_delete is not None:
            raise self._on_delete
        return self._store.delete_folder(folder)


def add_stored_file(session: Session) -> uuid.UUID:
    """A stored_files row of its own (the file store isn't part of these tests)."""
    sha256 = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    row = StoredFileRow(
        sha256=sha256, size=1, media_type="text/plain", storage_key=f"k/{sha256}"
    )
    session.add(row)
    session.flush()
    return row.id


def answering_chunk(
    document: Document,
    *,
    version: int,
    group_ids: tuple[uuid.UUID, ...],
    headings: tuple[str, ...],
) -> Chunk:
    """A chunk of `document` at `version` holding QUESTION's words."""
    text = f"Người lao động có 12 ngày {QUESTION}."
    return Chunk(
        id=uuid.uuid5(document.id, f"{version}:0"),
        text=text,
        embedded_text=text,
        field_values=[
            KbChunks.access_group_ids.value(sorted(group_ids)),
            KbChunks.collection_id.value(document.collection_id),
            KbChunks.document_id.value(document.id),
            KbChunks.document_version.value(version),
            KbChunks.position.value(0),
            KbChunks.headings.value(headings),
        ],
    )
