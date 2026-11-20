import hashlib
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import event, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from kb_server.domain.collections import Collection
from kb_server.domain.credentials import ReadAccess
from kb_server.domain.documents import (
    DOCUMENT_STATUSES,
    Document,
    DocumentPosition,
    DocumentStatus,
)
from kb_server.domain.groups import Group
from kb_server.domain.users import User
from kb_server.storage.models import (
    DocumentGroupRow,
    DocumentRow,
    DocumentVersionRow,
    IndexingJobRow,
    StoredFileRow,
)
from kb_server.storage.repositories.collections import CollectionRepository
from kb_server.storage.repositories.documents import DocumentRepository
from kb_server.storage.repositories.groups import GroupRepository
from kb_server.storage.repositories.jobs import JobRepository


def stored_file(session: Session, content: bytes) -> uuid.UUID:
    """A stored_files row (added directly: the file store isn't part of these tests)."""
    sha256 = hashlib.sha256(content).hexdigest()
    row = StoredFileRow(
        sha256=sha256,
        size=len(content),
        media_type="text/plain",
        storage_key=f"k/{sha256}",
    )
    session.add(row)
    session.flush()
    return row.id


@pytest.fixture
def uploader(make_user: Callable[..., User]) -> User:
    return make_user("manager@example.com", role="manager")


@pytest.fixture
def hr_docs(collections: CollectionRepository) -> Collection:
    return collections.add(slug="hr-docs", title="HR", description="")


@pytest.fixture
def it_docs(collections: CollectionRepository) -> Collection:
    return collections.add(slug="it-docs", title="IT", description="")


@pytest.fixture
def hr(groups: GroupRepository) -> Group:
    return groups.add("hr")


@pytest.fixture
def it(groups: GroupRepository) -> Group:
    return groups.add("it")


@pytest.fixture
def add_document(
    session: Session,
    documents: DocumentRepository,
    uploader: User,
    hr_docs: Collection,
) -> Callable[..., Document]:
    """add_document(content=b"…", group_ids=[…], collection=…) -> a new document."""

    def add(
        *,
        content: bytes = b"Dieu 1",
        group_ids: tuple[uuid.UUID, ...] = (),
        collection: Collection | None = None,
        **overrides: Any,
    ) -> Document:
        document, _ = documents.add(
            **{
                "collection_id": (collection or hr_docs).id,
                "title": "Quy chế nhân sự",
                "status": "active",
                "group_ids": group_ids,
                "file_id": stored_file(session, content),
                "original_filename": "quy-che.docx",
                "created_by": uploader.id,
                **overrides,
            }
        )
        return document

    return add


# adding


def test_a_new_document_comes_with_version_1(
    session: Session, documents: DocumentRepository, uploader: User, hr_docs: Collection
) -> None:
    file_id = stored_file(session, b"Dieu 1")

    document, version = documents.add(
        collection_id=hr_docs.id,
        title="Quy chế nhân sự",
        status="active",
        group_ids=[],
        file_id=file_id,
        original_filename="quy-che.docx",
        created_by=uploader.id,
    )

    assert document.collection_id == hr_docs.id
    assert document.title == "Quy chế nhân sự"
    assert document.status == "active"
    assert document.current_version == 1
    assert document.created_by == uploader.id
    assert document.created_at.tzinfo is not None
    assert document.updated_at.tzinfo is not None

    assert version.document_id == document.id
    assert version.version == 1
    assert version.file_id == file_id
    assert version.original_filename == "quy-che.docx"
    assert version.uploaded_by == uploader.id
    assert version.provenance is None  # filled in by the worker
    assert version.created_at.tzinfo is not None


def test_a_new_document_reads_back_the_same(
    documents: DocumentRepository, add_document: Callable[..., Document]
) -> None:
    document = add_document()

    assert documents.get(document.id) == document
    version = documents.get_version(document.id, 1)
    assert version is not None
    assert version.document_id == document.id


def test_a_draft_document_can_be_added(add_document: Callable[..., Document]) -> None:
    assert add_document(status="draft").status == "draft"


# compare_metadata ignores an Enum's values, so only a write shows that the CHECK allows them
@pytest.mark.parametrize("status", DOCUMENT_STATUSES)
def test_the_database_allows_every_document_status(
    add_document: Callable[..., Document], status: DocumentStatus
) -> None:
    assert add_document(status=status).status == status


def test_the_read_groups_are_saved_once_each(
    documents: DocumentRepository,
    add_document: Callable[..., Document],
    hr: Group,
    it: Group,
) -> None:
    document = add_document(group_ids=(hr.id, it.id, hr.id))

    assert documents.list_group_ids(document.id) == {hr.id, it.id}


def test_a_document_can_have_no_groups(
    documents: DocumentRepository, add_document: Callable[..., Document]
) -> None:
    document = add_document()

    assert documents.list_group_ids(document.id) == set()


def test_each_document_gets_its_own_id(add_document: Callable[..., Document]) -> None:
    first = add_document(content=b"one")
    second = add_document(content=b"two")

    assert first.id != second.id


def test_two_documents_can_share_a_stored_file(
    session: Session,
    documents: DocumentRepository,
    uploader: User,
    hr_docs: Collection,
    it_docs: Collection,
) -> None:
    file_id = stored_file(session, b"the same regulation")

    first, first_version = documents.add(
        collection_id=hr_docs.id,
        title="A",
        status="active",
        group_ids=[],
        file_id=file_id,
        original_filename="a.docx",
        created_by=uploader.id,
    )
    second, second_version = documents.add(
        collection_id=it_docs.id,
        title="B",
        status="active",
        group_ids=[],
        file_id=file_id,
        original_filename="b.docx",
        created_by=uploader.id,
    )

    assert first.id != second.id
    assert first_version.file_id == second_version.file_id == file_id


# reading


def test_unknown_documents_and_versions_are_none(
    documents: DocumentRepository, add_document: Callable[..., Document]
) -> None:
    document = add_document()

    assert documents.get(uuid.uuid4()) is None
    assert documents.get_version(document.id, 2) is None
    assert documents.get_version(uuid.uuid4(), 1) is None
    assert documents.list_group_ids(uuid.uuid4()) == set()


# finding a duplicate file


def test_find_current_with_file_finds_the_document(
    documents: DocumentRepository,
    add_document: Callable[..., Document],
    hr_docs: Collection,
) -> None:
    document = add_document(content=b"Dieu 1")

    found = documents.find_current_with_file(
        hr_docs.id, hashlib.sha256(b"Dieu 1").hexdigest()
    )

    assert found == document


def test_find_current_with_file_looks_only_in_the_collection(
    documents: DocumentRepository,
    add_document: Callable[..., Document],
    it_docs: Collection,
) -> None:
    add_document(content=b"Dieu 1")

    assert (
        documents.find_current_with_file(
            it_docs.id, hashlib.sha256(b"Dieu 1").hexdigest()
        )
        is None
    )


def test_find_current_with_file_ignores_other_files(
    documents: DocumentRepository,
    add_document: Callable[..., Document],
    hr_docs: Collection,
) -> None:
    add_document(content=b"Dieu 1")

    assert (
        documents.find_current_with_file(
            hr_docs.id, hashlib.sha256(b"other").hexdigest()
        )
        is None
    )


def test_find_current_with_file_skips_a_document_being_deleted(
    documents: DocumentRepository,
    add_document: Callable[..., Document],
    hr_docs: Collection,
) -> None:
    add_document(content=b"Dieu 1", status="deleting")

    assert (
        documents.find_current_with_file(
            hr_docs.id, hashlib.sha256(b"Dieu 1").hexdigest()
        )
        is None
    )


def test_find_current_with_file_ignores_older_versions(
    session: Session,
    documents: DocumentRepository,
    add_document: Callable[..., Document],
    uploader: User,
    hr_docs: Collection,
) -> None:
    # replacing files comes later: add version 2 by hand and make it current
    document = add_document(content=b"version 1")
    session.add(
        DocumentVersionRow(
            document_id=document.id,
            version=2,
            file_id=stored_file(session, b"version 2"),
            original_filename="quy-che-v2.docx",
            uploaded_by=uploader.id,
        )
    )
    session.flush()
    session.execute(
        update(DocumentRow)
        .where(DocumentRow.id == document.id)
        .values(current_version=2)
    )

    old = hashlib.sha256(b"version 1").hexdigest()
    new = hashlib.sha256(b"version 2").hexdigest()
    assert documents.find_current_with_file(hr_docs.id, old) is None
    found = documents.find_current_with_file(hr_docs.id, new)
    assert found is not None
    assert found.id == document.id


# listing


def add_documents(
    session: Session,
    add_document: Callable[..., Document],
    created_at: list[datetime],
    **fields: Any,
) -> list[Document]:
    """One document per time, created then: the list's order comes from these."""
    added = []
    for when in created_at:
        document = add_document(content=uuid.uuid4().bytes, **fields)
        session.execute(
            update(DocumentRow)
            .where(DocumentRow.id == document.id)
            .values(created_at=when)
        )
        added.append(document)
    return added


def list_ids(
    documents: DocumentRepository,
    *,
    access: ReadAccess | None = None,
    collection_id: uuid.UUID | None = None,
) -> list[uuid.UUID]:
    """The ids of every document on one large page."""
    page = documents.list_page(
        collection_id=collection_id, start_after=None, access=access, limit=100
    )
    return [document.id for document in page.items]


def walk_pages(documents: DocumentRepository, *, limit: int) -> list[list[uuid.UUID]]:
    """The ids on every page, following each page's next position."""
    pages: list[list[uuid.UUID]] = []
    start_after: DocumentPosition | None = None
    while True:
        page = documents.list_page(
            collection_id=None, start_after=start_after, access=None, limit=limit
        )
        pages.append([document.id for document in page.items])
        if page.next_position is None:
            return pages
        start_after = page.next_position


T0 = datetime(2026, 1, 1, tzinfo=UTC)


def days(n: int) -> datetime:
    return T0 + timedelta(days=n)


def test_the_list_is_newest_first(
    session: Session,
    documents: DocumentRepository,
    add_document: Callable[..., Document],
) -> None:
    old, new, middle = add_documents(session, add_document, [days(0), days(2), days(1)])

    assert list_ids(documents) == [new.id, middle.id, old.id]


def test_documents_created_at_the_same_time_are_ordered_by_id(
    session: Session,
    documents: DocumentRepository,
    add_document: Callable[..., Document],
) -> None:
    added = add_documents(session, add_document, [T0, T0, T0])

    assert list_ids(documents) == sorted((d.id for d in added), reverse=True)


def test_paging_shows_every_document_once_in_order(
    session: Session,
    documents: DocumentRepository,
    add_document: Callable[..., Document],
) -> None:
    # ties on created_at land on page boundaries
    add_documents(session, add_document, [days(0), days(0), days(1), days(1), days(2)])

    pages = walk_pages(documents, limit=2)

    assert [len(page) for page in pages] == [2, 2, 1]
    assert [id_ for page in pages for id_ in page] == list_ids(documents)


def test_a_full_last_page_has_no_next_position(
    session: Session,
    documents: DocumentRepository,
    add_document: Callable[..., Document],
) -> None:
    add_documents(session, add_document, [days(0), days(1), days(2), days(3)])

    assert [len(page) for page in walk_pages(documents, limit=2)] == [2, 2]


def test_a_document_added_between_pages_doesnt_shift_the_next_page(
    session: Session,
    documents: DocumentRepository,
    add_document: Callable[..., Document],
) -> None:
    d0, d1, d2, d3 = add_documents(
        session, add_document, [days(0), days(1), days(2), days(3)]
    )
    first = documents.list_page(
        collection_id=None, start_after=None, access=None, limit=2
    )
    add_documents(session, add_document, [days(4)])

    second = documents.list_page(
        collection_id=None, start_after=first.next_position, access=None, limit=2
    )

    assert [d.id for d in first.items] == [d3.id, d2.id]
    assert [d.id for d in second.items] == [d1.id, d0.id]


def test_the_list_can_be_narrowed_to_one_collection(
    documents: DocumentRepository,
    add_document: Callable[..., Document],
    hr_docs: Collection,
    it_docs: Collection,
) -> None:
    in_hr = add_document(content=b"hr")
    add_document(content=b"it", collection=it_docs)

    assert list_ids(documents, collection_id=hr_docs.id) == [in_hr.id]


def test_documents_being_deleted_are_left_out(
    session: Session,
    documents: DocumentRepository,
    add_document: Callable[..., Document],
) -> None:
    kept = add_document(content=b"kept")
    deleting = add_document(content=b"deleting")
    documents.mark_deleting(deleting.id)

    assert list_ids(documents) == [kept.id]


# who the list shows


def test_the_list_shows_documents_sharing_one_of_the_askers_groups(
    documents: DocumentRepository,
    add_document: Callable[..., Document],
    hr: Group,
    it: Group,
) -> None:
    for_hr = add_document(content=b"hr", group_ids=(hr.id,))
    for_both = add_document(content=b"both", group_ids=(hr.id, it.id))
    add_document(content=b"it", group_ids=(it.id,))
    add_document(content=b"private")
    access = ReadAccess(
        group_ids=frozenset({hr.id}), managed_collection_ids=frozenset()
    )

    assert set(list_ids(documents, access=access)) == {for_hr.id, for_both.id}


def test_the_list_shows_every_document_of_a_managed_collection_private_ones_included(
    documents: DocumentRepository,
    add_document: Callable[..., Document],
    hr_docs: Collection,
    it_docs: Collection,
    it: Group,
) -> None:
    private = add_document(content=b"private")
    shared = add_document(content=b"shared", group_ids=(it.id,))
    add_document(content=b"elsewhere", group_ids=(it.id,), collection=it_docs)
    access = ReadAccess(
        group_ids=frozenset(), managed_collection_ids=frozenset({hr_docs.id})
    )

    assert set(list_ids(documents, access=access)) == {private.id, shared.id}


def test_an_asker_with_no_groups_and_no_collections_sees_nothing(
    documents: DocumentRepository, add_document: Callable[..., Document], hr: Group
) -> None:
    add_document(content=b"hr", group_ids=(hr.id,))
    access = ReadAccess(group_ids=frozenset(), managed_collection_ids=frozenset())

    assert list_ids(documents, access=access) == []


def test_no_access_limit_shows_everything_private_documents_included(
    documents: DocumentRepository, add_document: Callable[..., Document], hr: Group
) -> None:
    shared = add_document(content=b"hr", group_ids=(hr.id,))
    private = add_document(content=b"private")

    assert set(list_ids(documents, access=None)) == {shared.id, private.id}


# what the database refuses (the services check these first; this is the backstop)


def test_an_unknown_group_is_refused(
    session: Session, add_document: Callable[..., Document]
) -> None:
    with pytest.raises(IntegrityError), session.begin_nested():
        add_document(group_ids=(uuid.uuid4(),))


def test_a_group_used_by_a_document_is_in_use_and_cannot_be_deleted(
    session: Session,
    groups: GroupRepository,
    add_document: Callable[..., Document],
    hr: Group,
) -> None:
    add_document(group_ids=(hr.id,))

    assert groups.is_in_use(hr.id)
    with pytest.raises(IntegrityError), session.begin_nested():
        groups.delete(hr.id)


def test_a_collection_with_documents_cannot_be_deleted(
    session: Session,
    collections: CollectionRepository,
    add_document: Callable[..., Document],
    hr_docs: Collection,
) -> None:
    add_document()

    with pytest.raises(IntegrityError), session.begin_nested():
        collections.delete(hr_docs.id)


def test_a_stored_file_in_use_cannot_be_deleted(
    session: Session, add_document: Callable[..., Document]
) -> None:
    document = add_document()
    version = session.get(DocumentVersionRow, (document.id, 1))
    assert version is not None

    with pytest.raises(IntegrityError), session.begin_nested():
        session.delete(session.get(StoredFileRow, version.file_id))
        session.flush()


# deleting


def rows_for(session: Session, row_type: Any, document_id: uuid.UUID) -> int:
    return (
        session.scalar(
            select(func.count())
            .select_from(row_type)
            .where(row_type.document_id == document_id)
        )
        or 0
    )


def test_list_file_ids_gives_the_file_of_every_version(
    session: Session,
    documents: DocumentRepository,
    add_document: Callable[..., Document],
    uploader: User,
) -> None:
    document = add_document(content=b"version 1")
    first = documents.get_version(document.id, 1)
    assert first is not None
    second_file = stored_file(session, b"version 2")
    session.add(
        DocumentVersionRow(
            document_id=document.id,
            version=2,
            file_id=second_file,
            original_filename="quy-che.docx",
            uploaded_by=uploader.id,
        )
    )
    session.flush()

    assert documents.list_file_ids(document.id) == {first.file_id, second_file}


def test_list_file_ids_of_an_unknown_document_is_empty(
    documents: DocumentRepository,
) -> None:
    assert documents.list_file_ids(uuid.uuid4()) == set()


def test_mark_deleting_marks_an_active_document(
    documents: DocumentRepository, add_document: Callable[..., Document]
) -> None:
    document = add_document()

    assert documents.mark_deleting(document.id) is True
    found = documents.get(document.id)
    assert found is not None and found.status == "deleting"


def test_mark_deleting_a_document_already_being_deleted_reports_false(
    documents: DocumentRepository, add_document: Callable[..., Document]
) -> None:
    document = add_document(status="deleting")

    assert documents.mark_deleting(document.id) is False


def test_mark_deleting_an_unknown_document_reports_false(
    documents: DocumentRepository,
) -> None:
    assert documents.mark_deleting(uuid.uuid4()) is False


def test_delete_removes_the_document_its_versions_groups_and_jobs(
    session: Session,
    documents: DocumentRepository,
    jobs: JobRepository,
    add_document: Callable[..., Document],
    hr: Group,
) -> None:
    document = add_document(status="deleting", group_ids=(hr.id,))
    jobs.add(kind="delete", document_id=document.id, version=None, created_by=None)

    assert documents.delete(document.id) is True

    assert documents.get(document.id) is None
    for row_type in (DocumentVersionRow, DocumentGroupRow, IndexingJobRow):
        assert rows_for(session, row_type, document.id) == 0


def test_delete_keeps_the_stored_file_for_the_worker_to_decide(
    session: Session,
    documents: DocumentRepository,
    add_document: Callable[..., Document],
) -> None:
    document = add_document(status="deleting")

    documents.delete(document.id)

    assert session.scalar(select(func.count()).select_from(StoredFileRow)) == 1


def test_delete_refuses_a_document_not_marked_for_deletion(
    documents: DocumentRepository, add_document: Callable[..., Document]
) -> None:
    document = add_document()

    assert documents.delete(document.id) is False
    assert documents.get(document.id) == document


def test_delete_an_unknown_document_reports_false(
    documents: DocumentRepository,
) -> None:
    assert documents.delete(uuid.uuid4()) is False


# the version search shows


def test_a_new_document_has_no_indexed_version(
    add_document: Callable[..., Document],
) -> None:
    assert add_document().indexed_version is None


def test_set_indexed_version_records_the_version_search_shows(
    documents: DocumentRepository, add_document: Callable[..., Document]
) -> None:
    document = add_document()

    assert documents.set_indexed_version(document.id, 1) is True

    found = documents.get(document.id)
    assert found is not None and found.indexed_version == 1


def test_set_indexed_version_of_an_unknown_document_reports_false(
    documents: DocumentRepository,
) -> None:
    assert documents.set_indexed_version(uuid.uuid4(), 1) is False


def test_served_versions_gives_each_active_documents_indexed_version(
    documents: DocumentRepository, add_document: Callable[..., Document]
) -> None:
    first = add_document(content=b"Dieu 1")
    second = add_document(content=b"Dieu 2")
    documents.set_indexed_version(first.id, 1)
    documents.set_indexed_version(second.id, 3)

    assert documents.served_versions({first.id, second.id}) == {
        first.id: 1,
        second.id: 3,
    }


@pytest.mark.parametrize("status", ["draft", "deleting"])
def test_served_versions_leaves_out_documents_that_arent_active(
    documents: DocumentRepository,
    add_document: Callable[..., Document],
    status: DocumentStatus,
) -> None:
    # a deleting document is a tombstone: search drops it before its delete job runs
    document = add_document(status=status)
    documents.set_indexed_version(document.id, 1)

    assert documents.served_versions({document.id}) == {}


def test_served_versions_leaves_out_documents_never_indexed_and_unknown_ids(
    documents: DocumentRepository, add_document: Callable[..., Document]
) -> None:
    never_indexed = add_document()

    assert documents.served_versions({never_indexed.id, uuid.uuid4()}) == {}


def test_served_versions_asks_once_however_many_documents(
    session: Session,
    documents: DocumentRepository,
    add_document: Callable[..., Document],
) -> None:
    # validity calls it at every gate of every search
    added = [add_document(content=f"Dieu {n}".encode()) for n in range(5)]
    for document in added:
        documents.set_indexed_version(document.id, 1)
    session.flush()

    with recorded_statements(session) as statements:
        served = documents.served_versions({document.id for document in added})

    assert len(served) == 5
    assert len(statements) == 1


def test_served_versions_of_no_documents_asks_nothing(
    session: Session, documents: DocumentRepository
) -> None:
    with recorded_statements(session) as statements:
        assert documents.served_versions(set()) == {}

    assert statements == []


def test_served_titles_gives_each_active_indexed_documents_title(
    documents: DocumentRepository, add_document: Callable[..., Document]
) -> None:
    first = add_document(content=b"Dieu 1", title="Quy chế nhân sự")
    second = add_document(content=b"Dieu 2", title="Quy chế tài chính")
    documents.set_indexed_version(first.id, 1)
    documents.set_indexed_version(second.id, 2)

    assert documents.served_titles({first.id, second.id}) == {
        first.id: "Quy chế nhân sự",
        second.id: "Quy chế tài chính",
    }


@pytest.mark.parametrize("status", ["draft", "deleting"])
def test_served_titles_leaves_out_the_documents_served_versions_leaves_out(
    documents: DocumentRepository,
    add_document: Callable[..., Document],
    status: DocumentStatus,
) -> None:
    not_active = add_document(content=b"Dieu 1", status=status)
    documents.set_indexed_version(not_active.id, 1)
    never_indexed = add_document(content=b"Dieu 2")
    ids = {not_active.id, never_indexed.id, uuid.uuid4()}

    assert documents.served_titles(ids) == {}
    assert documents.served_versions(ids) == {}


def test_served_titles_of_no_documents_asks_nothing(
    session: Session, documents: DocumentRepository
) -> None:
    with recorded_statements(session) as statements:
        assert documents.served_titles(set()) == {}

    assert statements == []


@contextmanager
def recorded_statements(session: Session) -> Iterator[list[str]]:
    """The SQL statements the session sends while the block runs."""
    statements: list[str] = []
    connection = session.connection()

    def record(*args: Any) -> None:
        # before_cursor_execute(conn, cursor, statement, parameters, context, executemany)
        statements.append(args[2])

    event.listen(connection, "before_cursor_execute", record)
    try:
        yield statements
    finally:
        event.remove(connection, "before_cursor_execute", record)
