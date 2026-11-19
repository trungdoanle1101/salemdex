"""The worker: claiming a job, running its handler, recording the outcome, recovering
stuck jobs and running until stopped.

Real Postgres rows (uploaded through DocumentService, so each document comes with its
prepare job), the real ProfileChooser, and the handler tests' recording fake embedder and
in-memory index. The worker's sessions join the test's transaction, so its commits are
savepoints; the test session reads them after expire_all().
"""

import io
import logging
import threading
import uuid
from collections.abc import Callable, Iterable, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any, BinaryIO

import pytest
from sqlalchemy import delete, insert, select, update
from sqlalchemy.orm import Session, sessionmaker

from kb_server.domain.chunks import KbChunks
from kb_server.domain.collections import Collection
from kb_server.domain.credentials import Principal
from kb_server.domain.documents import DocumentDetails, DocumentVersion
from kb_server.domain.groups import Group
from kb_server.domain.jobs import IndexingJob, JobKind
from kb_server.domain.users import User
from kb_server.indexing.handlers import JobTools
from kb_server.indexing.worker import Worker
from kb_server.services.documents import DocumentService
from kb_server.storage.files import FileStore, prepared_folder
from kb_server.storage.models import DocumentRow, DocumentVersionRow, IndexingJobRow
from kb_server.storage.repositories.collections import CollectionRepository
from kb_server.storage.repositories.documents import DocumentRepository
from kb_server.storage.repositories.files import StoredFileRepository
from kb_server.storage.repositories.groups import GroupRepository
from kb_server.storage.repositories.jobs import JobRepository
from salemdex_retrieval import Chunk as IndexChunk
from salemdex_retrieval import EmbeddedChunk, Everything
from salemdex_retrieval.chunks import Hit
from salemdex_retrieval.filters import Filter
from salemdex_retrieval.index import Index, InMemoryIndex
from salemdex_retrieval.schema import Schema
from salemdex_retrieval.vectors import VectorQuery
from tests.indexing.test_handlers import (
    EMBEDDER,
    INDEX_LABEL,
    REGULATION,
    VN_REGULATION,
    RecordingEmbedder,
    make_chooser,
    make_index_tools,
)
from tests.support import FakeClock

MakeWorker = Callable[..., Worker]
Upload = Callable[..., DocumentDetails]

POLL_INTERVAL = timedelta(minutes=1)
JOB_TIMEOUT = timedelta(minutes=30)
LEASE_EXPIRED = "lease expired: the work stopped during this job"


# setup


@pytest.fixture
def admin(make_user: Callable[..., User]) -> Principal:
    return Principal(
        user=make_user("admin@example.com", role="admin"),
        credential_id=uuid.uuid4(),
        credential_kind="session",
        acting_for_others=False,
    )


@pytest.fixture
def hr(groups: GroupRepository) -> Group:
    return groups.add("hr")


@pytest.fixture
def hr_docs(collections: CollectionRepository, hr: Group) -> Collection:
    collection = collections.add(slug="hr-docs", title="HR documents", description="")
    collections.set_groups(collection.id, [hr.id])
    return collection


@pytest.fixture
def upload(
    document_service: DocumentService, admin: Principal, hr_docs: Collection
) -> Upload:
    """upload(content=…, filename=…): an active document in hr-docs, with its prepare job."""

    def run(
        *, content: bytes = REGULATION.encode(), filename: str = "quy-dinh.txt"
    ) -> DocumentDetails:
        return document_service.upload_document(
            admin,
            collection_id=hr_docs.id,
            content=io.BytesIO(content),
            filename=filename,
            title=None,
            group_ids=None,
            private=False,
        )

    return run


@pytest.fixture
def embedder() -> RecordingEmbedder:
    return RecordingEmbedder()


@pytest.fixture
def index() -> InMemoryIndex:
    return InMemoryIndex(schema=KbChunks)


@pytest.fixture
def make_worker(
    session_factory: sessionmaker[Session],
    file_store: FileStore,
    index: InMemoryIndex,
    embedder: RecordingEmbedder,
    clock: FakeClock,
) -> MakeWorker:
    """make_worker(embedder=…, max_attempts=…, store=…, index=…): a worker over the test's rows and index."""

    def make(
        *,
        embedder_: RecordingEmbedder | None = None,
        max_attempts: int = 3,
        store: FileStore | None = None,
        index_: Index | None = None,
    ) -> Worker:
        tools = JobTools(
            profile_chooser=make_chooser(),
            index_tools=make_index_tools(embedder_ or embedder, index_ or index),
            file_store=store or file_store,
        )
        return Worker(
            session_factory=session_factory,
            tools=tools,
            max_attempts=max_attempts,
            poll_interval=POLL_INTERVAL,
            job_timeout=JOB_TIMEOUT,
            now=clock,
        )

    return make


@pytest.fixture
def worker(make_worker: MakeWorker) -> Worker:
    return make_worker()


class UnreachableEmbedder(RecordingEmbedder):
    """Fails every embedding with the error's message given, like a model server that's down."""

    def __init__(self, message: str = "model server unreachable") -> None:
        super().__init__()
        self._message = message

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        raise ConnectionError(self._message)


class WatchedIndex:
    """An index over another, recording which versions it holds after each replace.

    Its deletes fail while `deletes_fail` is set, like an index that's down. Only
    `replace` and `delete` matter to the index job and its cleanup; the rest passes
    through.
    """

    def __init__(self, inner: InMemoryIndex) -> None:
        self._inner = inner
        self.deletes_fail = False
        self.versions_after_replace: list[set[int]] = []

    @property
    def schema(self) -> type[Schema]:
        return self._inner.schema

    def put(self, embedded_chunks: Iterable[EmbeddedChunk]) -> None:
        self._inner.put(embedded_chunks)

    def find(self, *, where: Filter) -> list[IndexChunk]:
        return self._inner.find(where=where)

    def delete(self, *, where: Filter) -> None:
        if self.deletes_fail:
            # a message quoting the document, as a backend error might
            raise ConnectionError("Điều 1 couldn't be deleted")
        self._inner.delete(where=where)

    def replace(self, *, where: Filter, with_: Iterable[EmbeddedChunk]) -> None:
        self._inner.replace(where=where, with_=with_)
        held = self._inner.find(where=Everything())
        self.versions_after_replace.append(
            {chunk[KbChunks.document_version] for chunk in held}
        )

    def search(self, query: VectorQuery, *, where: Filter, limit: int) -> list[Hit]:
        return self._inner.search(query, where=where, limit=limit)


class StopOnFirstWait(threading.Event):
    """A stop event that sets itself the first time the worker waits, and records the wait."""

    def __init__(self) -> None:
        super().__init__()
        self.waits_seconds: list[float | None] = []

    def wait(self, timeout: float | None = None) -> bool:
        self.waits_seconds.append(timeout)
        self.set()
        return True


class SignalsWaiting(threading.Event):
    """A real stop event that also tells the test when the worker has started waiting."""

    def __init__(self) -> None:
        super().__init__()
        self.waiting = threading.Event()

    def wait(self, timeout: float | None = None) -> bool:
        self.waiting.set()
        return super().wait(timeout)


# an empty queue


def test_run_once_returns_false_when_no_job_is_waiting(worker: Worker) -> None:
    assert worker.run_once() is False


# the prepare job


def test_a_prepare_job_records_its_preparation_and_queues_the_index_job(
    worker: Worker, upload: Upload, session: Session, clock: FakeClock
) -> None:
    details = upload()

    assert worker.run_once() is True

    prepare = job(session, _first_job_id(details))
    assert (prepare.status, prepare.attempts) == ("done", 1)
    assert prepare.started_at == clock.now
    assert prepare.finished_at == clock.now
    assert [(j.kind, j.version, j.status) for j in jobs_of(session, details)] == [
        ("prepare", 1, "done"),
        ("index", 1, "queued"),
    ]
    preparation = provenance(session, details)["preparation"]
    assert preparation["profile"] == VN_REGULATION
    assert preparation["profiles_without_structure"] == []
    assert "embedder" not in preparation["settings"]


def test_the_index_job_is_queued_by_whoever_queued_the_prepare_job(
    worker: Worker, upload: Upload, session: Session, admin: Principal
) -> None:
    details = upload()

    worker.run_once()

    assert {j.created_by for j in jobs_of(session, details)} == {admin.user.id}


def test_a_draft_document_is_prepared_but_not_indexed(
    worker: Worker, upload: Upload, session: Session
) -> None:
    details = upload()
    session.execute(
        update(DocumentRow)
        .where(DocumentRow.id == details.document.id)
        .values(status="draft")
    )

    worker.run_once()

    assert [(j.kind, j.status) for j in jobs_of(session, details)] == [
        ("prepare", "done")
    ]
    assert "preparation" in provenance(session, details)


# the index job


def test_an_index_job_indexes_the_chunks_and_adds_its_provenance(
    worker: Worker, upload: Upload, session: Session, index: InMemoryIndex
) -> None:
    details = upload()

    worker.run_once()
    worker.run_once()

    assert [(j.kind, j.status) for j in jobs_of(session, details)] == [
        ("prepare", "done"),
        ("index", "done"),
    ]
    recorded = provenance(session, details)
    assert recorded["index"] == {
        "embedder": EMBEDDER,
        "index": INDEX_LABEL,
    }
    assert recorded["preparation"]["profile"] == VN_REGULATION
    assert indexed(index, details)


def test_the_index_gets_the_documents_groups_and_version(
    worker: Worker, upload: Upload, index: InMemoryIndex, hr: Group
) -> None:
    details = upload()

    worker.run_once()
    worker.run_once()

    chunk = indexed(index, details)[0]
    assert chunk[KbChunks.access_group_ids] == (hr.id,)
    assert chunk[KbChunks.collection_id] == details.document.collection_id
    assert chunk[KbChunks.document_version] == 1


def test_an_index_job_makes_its_version_the_one_search_shows(
    worker: Worker, upload: Upload, session: Session
) -> None:
    details = upload()

    run_until_idle(worker)

    assert indexed_version(session, details) == 1
    assert DocumentRepository(session).served_versions({details.document.id}) == {
        details.document.id: 1
    }


def test_a_prepared_but_not_yet_indexed_document_is_shown_nowhere(
    worker: Worker, upload: Upload, session: Session
) -> None:
    details = upload()

    worker.run_once()  # the prepare job only

    assert indexed_version(session, details) is None


def test_a_new_version_keeps_the_old_one_shown_until_it_is_recorded(
    make_worker: MakeWorker, upload: Upload, session: Session, index: InMemoryIndex
) -> None:
    watched = WatchedIndex(index)
    worker = make_worker(index_=watched)
    details = upload()
    run_until_idle(worker)
    add_current_version(session, details, version=2)
    queue(session, details, kind="prepare", version=2)

    run_until_idle(worker)

    # while v2 was written, v1 was still there for search to show
    assert watched.versions_after_replace == [{1}, {1, 2}]
    assert indexed_version(session, details) == 2
    assert {chunk[KbChunks.document_version] for chunk in indexed(index, details)} == {
        2
    }


def test_a_failed_cleanup_leaves_the_job_done_and_logs_only_the_error_type(
    make_worker: MakeWorker,
    upload: Upload,
    session: Session,
    index: InMemoryIndex,
    caplog: pytest.LogCaptureFixture,
) -> None:
    watched = WatchedIndex(index)
    worker = make_worker(index_=watched)
    details = upload()
    run_until_idle(worker)
    add_current_version(session, details, version=2)
    queue(session, details, kind="prepare", version=2)
    watched.deletes_fail = True
    caplog.set_level(logging.WARNING, logger="kb_server.indexing.worker")

    run_until_idle(worker)

    index_jobs = [j for j in jobs_of(session, details) if j.kind == "index"]
    assert [(j.version, j.status, j.attempts) for j in index_jobs] == [
        (1, "done", 1),
        (2, "done", 1),
    ]
    assert indexed_version(session, details) == 2
    # left behind, but hidden: validity shows only the indexed version
    assert {chunk[KbChunks.document_version] for chunk in indexed(index, details)} == {
        1,
        2,
    }
    assert "ConnectionError" in caplog.text
    assert str(details.document.id) in caplog.text
    assert "Điều" not in caplog.text


def test_the_next_index_job_removes_what_a_failed_cleanup_left(
    make_worker: MakeWorker, upload: Upload, session: Session, index: InMemoryIndex
) -> None:
    watched = WatchedIndex(index)
    worker = make_worker(index_=watched)
    details = upload()
    run_until_idle(worker)
    add_current_version(session, details, version=2)
    queue(session, details, kind="prepare", version=2)
    watched.deletes_fail = True
    run_until_idle(worker)
    watched.deletes_fail = False

    queue(session, details, kind="index", version=2)
    run_until_idle(worker)

    assert {chunk[KbChunks.document_version] for chunk in indexed(index, details)} == {
        2
    }


def test_preparing_a_version_again_drops_its_index_provenance_and_queues_indexing(
    worker: Worker, upload: Upload, session: Session
) -> None:
    details = upload()
    worker.run_once()
    worker.run_once()
    queue(session, details, kind="prepare", version=1)

    worker.run_once()

    assert "index" not in provenance(session, details)
    assert sorted((j.kind, j.status) for j in jobs_of(session, details)) == [
        ("index", "done"),
        ("index", "queued"),
        ("prepare", "done"),
        ("prepare", "done"),
    ]


# failures


def test_an_error_is_kept_and_the_job_requeued_while_attempts_are_left(
    make_worker: MakeWorker, upload: Upload, session: Session
) -> None:
    details = upload()
    worker = make_worker(embedder_=UnreachableEmbedder(), max_attempts=2)
    worker.run_once()

    assert worker.run_once() is True

    index_job = jobs_of(session, details)[-1]
    assert (index_job.kind, index_job.status) == ("index", "queued")
    assert index_job.attempts == 1
    assert index_job.error == "ConnectionError: model server unreachable"
    assert index_job.started_at is None


def test_a_job_fails_for_good_on_its_last_attempt(
    make_worker: MakeWorker, upload: Upload, session: Session, clock: FakeClock
) -> None:
    details = upload()
    worker = make_worker(embedder_=UnreachableEmbedder(), max_attempts=2)
    worker.run_once()

    worker.run_once()
    clock.advance(timedelta(minutes=1))
    worker.run_once()

    index_job = jobs_of(session, details)[-1]
    assert (index_job.status, index_job.attempts) == ("failed", 2)
    assert index_job.finished_at == clock.now
    assert worker.run_once() is False


def test_a_permanent_error_fails_the_job_at_once(
    worker: Worker, upload: Upload, session: Session, file_store_root: Path
) -> None:
    details = upload()
    (file_store_root / details.file.storage_key).unlink()

    worker.run_once()

    prepare = jobs_of(session, details)[0]
    assert (prepare.status, prepare.attempts) == ("failed", 1)
    assert prepare.error is not None
    assert prepare.error.startswith("PermanentJobError: ")
    assert len(jobs_of(session, details)) == 1


def test_a_failure_logs_the_error_type_but_not_its_message(
    make_worker: MakeWorker, upload: Upload, caplog: pytest.LogCaptureFixture
) -> None:
    upload()
    # an error message that quotes the document, as a framework error might
    worker = make_worker(embedder_=UnreachableEmbedder("Điều 1. Phạm vi điều chỉnh"))
    worker.run_once()

    with caplog.at_level(logging.INFO, logger="kb_server.indexing.worker"):
        worker.run_once()

    assert "ConnectionError" in caplog.text
    assert "Phạm vi" not in caplog.text


def test_a_job_whose_rows_cant_be_loaded_is_failed_and_the_next_job_runs(
    worker: Worker, upload: Upload, session: Session
) -> None:
    broken = upload()
    healthy = upload(content="Điều 1. Một văn bản khác\n".encode(), filename="khac.txt")
    queued_at(session, _first_job_id(broken), minutes=-1)
    # current_version has no foreign key, so it can point at a version row that's missing
    session.execute(
        update(DocumentRow)
        .where(DocumentRow.id == broken.document.id)
        .values(current_version=9)
    )

    assert worker.run_once() is True

    failed = job(session, _first_job_id(broken))
    assert failed.status == "failed"
    assert failed.error is not None
    assert "v9" in failed.error
    assert job(session, _first_job_id(healthy)).status == "done"


# superseded jobs


def test_a_superseded_job_is_skipped_and_the_next_job_runs_in_the_same_call(
    worker: Worker, upload: Upload, session: Session
) -> None:
    details = upload()
    add_current_version(session, details, version=2)
    newer = queue(session, details, kind="prepare", version=2)
    queued_at(session, _first_job_id(details), minutes=-1)

    assert worker.run_once() is True

    assert job(session, _first_job_id(details)).status == "done"
    assert job(session, newer.id).status == "done"
    assert [(j.kind, j.version) for j in jobs_of(session, details)][-1] == ("index", 2)
    assert provenance(session, details, version=1) is None


def test_a_superseded_index_job_embeds_nothing(
    worker: Worker, upload: Upload, session: Session, embedder: RecordingEmbedder
) -> None:
    details = upload()
    worker.run_once()
    add_current_version(session, details, version=2)

    worker.run_once()

    assert jobs_of(session, details)[-1].status == "done"
    assert embedder.passages == []


def test_a_job_for_the_whole_document_is_never_superseded(
    worker: Worker, upload: Upload, session: Session
) -> None:
    details = upload()
    session.execute(delete(IndexingJobRow))
    add_current_version(session, details, version=2)
    sync = queue(session, details, kind="sync_attributes", version=None)

    worker.run_once()

    # it reached its handler, which doesn't support it yet
    recorded = job(session, sync.id)
    assert recorded.status == "failed"
    assert recorded.error == "PermanentJobError: sync_attributes not yet supported"


def test_every_job_of_a_document_being_deleted_is_skipped(
    worker: Worker, upload: Upload, session: Session, embedder: RecordingEmbedder
) -> None:
    details = upload()
    queue(session, details, kind="index", version=1)
    queue(session, details, kind="sync_attributes", version=None)
    DocumentRepository(session).mark_deleting(details.document.id)

    worker.run_once()

    assert [(j.kind, j.status) for j in jobs_of(session, details)] == [
        ("sync_attributes", "done"),
        ("prepare", "done"),
        ("index", "done"),
    ]
    assert provenance(session, details) is None
    assert embedder.passages == []


def test_the_delete_job_of_a_document_being_deleted_is_not_skipped(
    worker: Worker, upload: Upload, session: Session
) -> None:
    details = upload()
    session.execute(delete(IndexingJobRow))
    DocumentRepository(session).mark_deleting(details.document.id)
    deletion = queue(session, details, kind="delete", version=None)

    worker.run_once()

    session.expire_all()
    assert JobRepository(session).get(deletion.id) is None
    assert DocumentRepository(session).get(details.document.id) is None


# the delete job


def test_a_delete_job_removes_the_document_everywhere(
    worker: Worker,
    upload: Upload,
    session: Session,
    index: InMemoryIndex,
    file_store: FileStore,
    file_store_root: Path,
) -> None:
    details = upload()
    run_until_idle(worker)
    assert indexed(index, details) != []
    request_deletion(session, details)

    assert worker.run_once() is True

    session.expire_all()
    assert DocumentRepository(session).get(details.document.id) is None
    assert jobs_of(session, details) == []
    assert StoredFileRepository(session).get(details.file.id) is None
    assert indexed(index, details) == []
    assert not (file_store_root / prepared_folder(details.document.id)).exists()
    assert not file_store.exists(details.file.storage_key)


def test_an_original_another_document_uses_is_kept(
    worker: Worker, upload: Upload, session: Session, file_store: FileStore
) -> None:
    details = upload()
    DocumentRepository(session).add(
        collection_id=details.document.collection_id,
        title="Bản sao",
        status="active",
        group_ids=[],
        file_id=details.file.id,
        original_filename="ban-sao.txt",
        created_by=details.document.created_by,
    )
    request_deletion(session, details)

    run_until_idle(worker)

    session.expire_all()
    assert DocumentRepository(session).get(details.document.id) is None
    assert StoredFileRepository(session).get(details.file.id) is not None
    assert file_store.exists(details.file.storage_key)


def test_an_original_the_store_cant_delete_is_left_and_logged_by_id(
    make_worker: MakeWorker,
    upload: Upload,
    session: Session,
    file_store: FileStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    worker = make_worker(store=FailingFileDeletes(file_store))
    details = upload()
    request_deletion(session, details)
    caplog.set_level(logging.WARNING, logger="kb_server.indexing.worker")

    run_until_idle(worker)

    session.expire_all()
    assert DocumentRepository(session).get(details.document.id) is None
    assert file_store.exists(details.file.storage_key)  # an orphan, for gc
    assert str(details.file.id) in caplog.text
    assert details.file.sha256 not in caplog.text


class FailingFileDeletes:
    """A store that can't delete single files, like a read-only bucket; folders still go."""

    def __init__(self, store: FileStore) -> None:
        self._store = store

    def put(self, key: str, data: BinaryIO, *, size: int, media_type: str) -> None:
        self._store.put(key, data, size=size, media_type=media_type)

    def open(self, key: str) -> BinaryIO:
        return self._store.open(key)

    def exists(self, key: str) -> bool:
        return self._store.exists(key)

    def delete(self, key: str) -> None:
        raise PermissionError("read-only")

    def delete_folder(self, folder: str) -> int:
        return self._store.delete_folder(folder)


# recovering stuck jobs


def test_a_job_left_running_past_the_timeout_is_requeued(
    worker: Worker, upload: Upload, session: Session, clock: FakeClock
) -> None:
    details = upload()
    abandon(session, clock)
    clock.advance(JOB_TIMEOUT + timedelta(minutes=1))

    assert worker.recover_stuck_jobs() == 1

    requeued = job(session, _first_job_id(details))
    assert (requeued.status, requeued.attempts) == ("queued", 1)
    assert requeued.error == LEASE_EXPIRED
    assert requeued.started_at is None


def test_a_requeued_stuck_job_runs_again(
    worker: Worker, upload: Upload, session: Session, clock: FakeClock
) -> None:
    details = upload()
    abandon(session, clock)
    clock.advance(JOB_TIMEOUT + timedelta(minutes=1))
    worker.recover_stuck_jobs()

    assert worker.run_once() is True

    prepare = job(session, _first_job_id(details))
    assert (prepare.status, prepare.attempts) == ("done", 2)


def test_a_stuck_job_on_its_last_attempt_is_failed(
    make_worker: MakeWorker, upload: Upload, session: Session, clock: FakeClock
) -> None:
    details = upload()
    worker = make_worker(max_attempts=1)
    abandon(session, clock)
    clock.advance(JOB_TIMEOUT + timedelta(minutes=1))

    assert worker.recover_stuck_jobs() == 1

    failed = job(session, _first_job_id(details))
    assert (failed.status, failed.attempts) == ("failed", 1)
    assert failed.error == LEASE_EXPIRED
    assert failed.finished_at == clock.now


def test_a_job_running_for_less_than_the_timeout_is_left_alone(
    worker: Worker, upload: Upload, session: Session, clock: FakeClock
) -> None:
    details = upload()
    abandon(session, clock)
    clock.advance(JOB_TIMEOUT - timedelta(minutes=1))

    assert worker.recover_stuck_jobs() == 0

    assert job(session, _first_job_id(details)).status == "running"


def test_the_timeout_counts_from_the_claim_not_from_the_queueing(
    worker: Worker, upload: Upload, session: Session, clock: FakeClock
) -> None:
    details = upload()
    # queued a day before the clock's now (created_at comes from the database's clock),
    # claimed just now
    session.execute(
        update(IndexingJobRow)
        .where(IndexingJobRow.id == _first_job_id(details))
        .values(created_at=clock.now - timedelta(days=1))
    )
    abandon(session, clock)

    assert worker.recover_stuck_jobs() == 0

    assert job(session, _first_job_id(details)).status == "running"


def test_queued_and_finished_jobs_are_left_alone(
    worker: Worker, upload: Upload, session: Session, clock: FakeClock
) -> None:
    details = upload()
    worker.run_once()
    clock.advance(JOB_TIMEOUT + timedelta(minutes=1))

    assert worker.recover_stuck_jobs() == 0

    assert [(j.kind, j.status) for j in jobs_of(session, details)] == [
        ("prepare", "done"),
        ("index", "queued"),
    ]


def test_a_stuck_job_is_logged_with_its_id(
    worker: Worker,
    upload: Upload,
    session: Session,
    clock: FakeClock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    details = upload()
    abandon(session, clock)
    clock.advance(JOB_TIMEOUT + timedelta(minutes=1))

    with caplog.at_level(logging.WARNING, logger="kb_server.indexing.worker"):
        worker.recover_stuck_jobs()

    assert str(_first_job_id(details)) in caplog.text
    assert "%s" not in caplog.text


# running until stopped


def test_run_forever_runs_every_waiting_job_before_it_waits(
    worker: Worker, upload: Upload, session: Session
) -> None:
    first = upload()
    second = upload(content="Điều 1. Một văn bản khác\n".encode(), filename="khac.txt")
    stop = StopOnFirstWait()

    assert ran_to_the_end(worker, stop=stop), "run_forever never waited on stop"

    for details in (first, second):
        assert [(j.kind, j.status) for j in jobs_of(session, details)] == [
            ("prepare", "done"),
            ("index", "done"),
        ]
    assert stop.waits_seconds == [POLL_INTERVAL.total_seconds()]


def test_run_forever_runs_no_job_once_stop_is_set(
    worker: Worker, upload: Upload, session: Session
) -> None:
    details = upload()
    stop = threading.Event()
    stop.set()

    worker.run_forever(stop=stop)

    assert job(session, _first_job_id(details)).status == "queued"


def test_an_idle_worker_stops_as_soon_as_stop_is_set(worker: Worker) -> None:
    stop = SignalsWaiting()
    # poll_interval is a minute: a worker that slept instead of waiting on stop would
    # still be running when the join below gives up
    thread = start_in_background(worker, stop=stop)
    assert stop.waiting.wait(timeout=5), "the worker never waited on stop"

    stop.set()
    thread.join(timeout=5)

    assert not thread.is_alive()


def start_in_background(worker: Worker, *, stop: threading.Event) -> threading.Thread:
    """Start run_forever in a daemon thread, so a worker that never stops can't hang the run."""
    thread = threading.Thread(
        target=worker.run_forever, kwargs={"stop": stop}, daemon=True
    )
    thread.start()
    return thread


def ran_to_the_end(worker: Worker, *, stop: threading.Event) -> bool:
    """Run run_forever until it returns. False if it's still running after 10 seconds."""
    thread = start_in_background(worker, stop=stop)
    thread.join(timeout=10)
    return not thread.is_alive()


# helpers


def run_until_idle(worker: Worker) -> None:
    while worker.run_once():
        pass


def request_deletion(session: Session, details: DocumentDetails) -> None:
    """What DELETE /documents/{id} does: mark the document and queue its delete job."""
    assert DocumentRepository(session).mark_deleting(details.document.id)
    queue(session, details, kind="delete", version=None)


def job(session: Session, job_id: uuid.UUID) -> IndexingJob:
    """The job as the worker left it."""
    session.expire_all()
    found = JobRepository(session).get(job_id)
    assert found is not None
    return found


def jobs_of(session: Session, details: DocumentDetails) -> list[IndexingJob]:
    """The document's jobs as the worker left them, by version, then prepare before index.

    Jobs of the same kind and version come in no particular order.
    """
    session.expire_all()
    ids = session.scalars(
        select(IndexingJobRow.id).where(
            IndexingJobRow.document_id == details.document.id
        )
    ).all()
    found = [job(session, job_id) for job_id in ids]
    order: dict[JobKind, int] = {
        "prepare": 0,
        "index": 1,
        "sync_attributes": 2,
        "delete": 3,
    }
    return sorted(found, key=lambda j: (j.version or 0, order[j.kind]))


def provenance(session: Session, details: DocumentDetails, *, version: int = 1) -> Any:
    session.expire_all()
    found = DocumentRepository(session).get_version(details.document.id, version)
    assert found is not None
    return found.provenance


def queue(
    session: Session,
    details: DocumentDetails,
    *,
    kind: JobKind,
    version: int | None,
) -> IndexingJob:
    return JobRepository(session).add(
        kind=kind, document_id=details.document.id, version=version, created_by=None
    )


def abandon(session: Session, clock: FakeClock) -> IndexingJob:
    """Claim the oldest queued job at the clock's time and leave it, as a crashed worker would."""
    claimed = JobRepository(session).claim_next(now=clock.now)
    assert claimed is not None
    return claimed


def queued_at(session: Session, job_id: uuid.UUID, *, minutes: int) -> None:
    """Move a job's created_at. Jobs added in one test share the transaction's now()."""
    session.execute(
        update(IndexingJobRow)
        .where(IndexingJobRow.id == job_id)
        .values(created_at=IndexingJobRow.created_at + timedelta(minutes=minutes))
    )


def add_current_version(
    session: Session, details: DocumentDetails, *, version: int
) -> DocumentVersion:
    """A newer version of the same file becomes current, as a re-upload would make it."""
    session.execute(
        insert(DocumentVersionRow).values(
            document_id=details.document.id,
            version=version,
            file_id=details.file.id,
            original_filename=details.version.original_filename,
            uploaded_by=details.document.created_by,
        )
    )
    session.execute(
        update(DocumentRow)
        .where(DocumentRow.id == details.document.id)
        .values(current_version=version)
    )
    found = DocumentRepository(session).get_version(details.document.id, version)
    assert found is not None
    return found


def indexed_version(session: Session, details: DocumentDetails) -> int | None:
    """The version search shows, as the worker left it."""
    session.expire_all()
    found = DocumentRepository(session).get(details.document.id)
    assert found is not None
    return found.indexed_version


def indexed(index: InMemoryIndex, details: DocumentDetails) -> list[IndexChunk]:
    """The document's chunks in the index, in document order."""
    chunks = index.find(where=KbChunks.document_id.eq(details.document.id))
    return sorted(chunks, key=lambda chunk: chunk[KbChunks.position])


def _first_job_id(details: DocumentDetails) -> uuid.UUID:
    """The prepare job the upload queued."""
    assert details.job is not None
    return details.job.id
