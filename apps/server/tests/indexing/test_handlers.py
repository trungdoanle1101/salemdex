"""Choosing how to prepare a document, and the prepare and index jobs' handlers.

No database: a ClaimedJob is built by hand, the store is a LocalFileStore in a temporary
folder, chunks are sized in words and embedded by the fake embedder, and the index lives
in memory, so no model or Qdrant is needed.
"""

import dataclasses
import hashlib
import io
import json
import logging
import tempfile
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

import docx
import pytest

from document_processor.chunking import Chunk, OutlineChunker, WordTokenizer
from document_processor.core.models import LogicalDocument
from document_processor.logical.profiles import PROFILES
from kb_server.domain.chunks import KbChunks
from kb_server.domain.documents import Document, DocumentStatus, DocumentVersion
from kb_server.domain.files import DOCX, TEXT, StoredFile
from kb_server.domain.jobs import IndexingJob
from kb_server.indexing.handlers import (
    ClaimedJob,
    IndexTools,
    PermanentJobError,
    ProfileChooser,
    remove_other_versions,
    run_delete_job,
    run_index_job,
    run_prepare_job,
)
from kb_server.indexing.prepared import PreparedParts, load_prepared, save_prepared
from kb_server.storage.files import (
    FileStore,
    LocalFileStore,
    original_key,
    prepared_folder,
    prepared_version_folder,
)
from salemdex_retrieval import Chunk as IndexChunk
from salemdex_retrieval.embedders import DenseBinding
from salemdex_retrieval.embedders.fake import FakeDenseEmbedder
from salemdex_retrieval.index import Index, InMemoryIndex

NOW = datetime(2026, 11, 9, 9, 0, tzinfo=UTC)

REGULATION = """QUY ĐỊNH QUẢN LÝ DỮ LIỆU
Chương I
QUY ĐỊNH CHUNG
Điều 1. Phạm vi điều chỉnh
1. Quy định này áp dụng cho dữ liệu khách hàng tại ngân hàng.
2. Dữ liệu vùng Staging được lưu trữ tối đa 07 ngày.
Điều 2. Thời hạn lưu trữ
Dữ liệu lưu trữ dài hạn được lưu trên Tape.
Nơi nhận:
- Như trên
"""

PLAIN_TEXT = """Ghi chú cuộc họp về kế hoạch sao lưu.
Nhóm vận hành sẽ kiểm tra lịch sao lưu hằng tuần.
Mọi thay đổi cần được báo trước cho bộ phận an ninh thông tin.
"""


def label(name: str) -> str:
    profile = PROFILES.get(name)
    return f"{profile.name}@{profile.version}"


VN_REGULATION = label("vn_regulation")
PLAIN = label("plain")
# the embedder make_chooser records and the index tools embed with, unless told otherwise
EMBEDDER = "fake"
INDEX_LABEL = "memory:test"


def make_chooser(
    profiles: tuple[str, ...] = ("vn_regulation",),
    fallback: str = "plain",
) -> ProfileChooser:
    return ProfileChooser.from_profiles(
        profiles=profiles,
        fallback_profile=fallback,
        chunker=OutlineChunker(tokenizer=WordTokenizer(), max_tokens=50),
    )


def docx_of(text: str) -> bytes:
    document = docx.Document()
    for line in text.splitlines():
        document.add_paragraph(line)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def damaged_docx() -> bytes:
    # passes the upload check (a zip with word/document.xml) but isn't a Word file
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<x/>")
        archive.writestr("word/document.xml", "<x/>")
    return buffer.getvalue()


def text_bytes(text: str) -> bytes:
    return text.encode("utf-8")


class RecordingEmbedder(FakeDenseEmbedder):
    """The fake embedder, sized for `KbChunks.dense`, recording every passage it embeds.

    Tests use the record to check that nothing was embedded.
    """

    def __init__(self) -> None:
        super().__init__(size=KbChunks.dense.size)
        self.passages: list[str] = []

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        self.passages.extend(texts)
        return super().embed_passages(texts)


def make_index_tools(embedder: RecordingEmbedder, index: Index) -> IndexTools:
    return IndexTools(
        binding=DenseBinding(vector=KbChunks.dense, embedder=embedder),
        embedder_identity=EMBEDDER,
        index=index,
        index_label=INDEX_LABEL,
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    root.mkdir()
    return root


@pytest.fixture
def store(root: Path) -> FileStore:
    return LocalFileStore(root)


@pytest.fixture
def chooser() -> ProfileChooser:
    return make_chooser()


@pytest.fixture
def embedder() -> RecordingEmbedder:
    return RecordingEmbedder()


@pytest.fixture
def index() -> InMemoryIndex:
    return InMemoryIndex(schema=KbChunks)


@pytest.fixture
def tools(embedder: RecordingEmbedder, index: InMemoryIndex) -> IndexTools:
    return make_index_tools(embedder, index)


@pytest.fixture
def temp_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Where the handler's temporary folders go, so tests can check they're cleaned up."""
    folder = tmp_path / "temp"
    folder.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(folder))
    return folder


def stage(
    store: FileStore,
    content: bytes,
    *,
    media_type: str = TEXT,
    status: DocumentStatus = "active",
    document_id: uuid.UUID | None = None,
    version: int = 1,
    stored: bool = True,
    group_ids: frozenset[uuid.UUID] = frozenset(),
) -> ClaimedJob:
    """A claimed prepare job, with its original in the store (unless stored=False)."""
    sha256 = hashlib.sha256(content).hexdigest()
    key = original_key(sha256)
    if stored:
        store.put(key, io.BytesIO(content), size=len(content), media_type=media_type)
    document_id = document_id or uuid.uuid4()
    uploader = uuid.uuid4()
    file = StoredFile(
        id=uuid.uuid4(),
        sha256=sha256,
        size=len(content),
        media_type=media_type,
        storage_key=key,
        created_at=NOW,
    )
    return ClaimedJob(
        job=IndexingJob(
            id=uuid.uuid4(),
            kind="prepare",
            document_id=document_id,
            version=version,
            status="running",
            attempts=1,
            error=None,
            created_by=uploader,
            created_at=NOW,
            started_at=NOW,
            finished_at=None,
        ),
        document=Document(
            id=document_id,
            collection_id=uuid.uuid4(),
            title="Quy định",
            status=status,
            current_version=version,
            indexed_version=None,
            created_by=uploader,
            created_at=NOW,
            updated_at=NOW,
        ),
        version=DocumentVersion(
            document_id=document_id,
            version=version,
            file_id=file.id,
            original_filename="quy-dinh.txt",
            provenance=None,
            uploaded_by=uploader,
            created_at=NOW,
        ),
        file=file,
        group_ids=group_ids,
    )


def saved_files(root: Path, job: ClaimedJob) -> list[str]:
    folder = root / prepared_version_folder(job.document.id, job.version.version)
    if not folder.exists():
        return []
    return sorted(path.name for path in folder.iterdir())


def load(
    store: FileStore, job: ClaimedJob
) -> tuple[LogicalDocument, tuple[Chunk, ...]]:
    parts = load_prepared(store, job.document.id, job.version.version)
    assert parts is not None
    return parts.logical, parts.chunks


def write(tmp_path: Path, name: str, content: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def stage_prepared(
    store: FileStore,
    content: bytes,
    *,
    chooser: ProfileChooser,
    document_id: uuid.UUID | None = None,
    version: int = 1,
    group_ids: frozenset[uuid.UUID] = frozenset(),
) -> ClaimedJob:
    """A claimed index job for a version the prepare job prepared, with its provenance."""
    staged = stage(
        store, content, document_id=document_id, version=version, group_ids=group_ids
    )
    result = run_prepare_job(staged, chooser=chooser, store=store)
    return dataclasses.replace(
        with_provenance(staged, {"preparation": result.provenance()}),
        job=dataclasses.replace(staged.job, kind="index"),
    )


def with_provenance(
    job: ClaimedJob, provenance: Mapping[str, Any] | None
) -> ClaimedJob:
    return dataclasses.replace(
        job, version=dataclasses.replace(job.version, provenance=provenance)
    )


def indexed(index: InMemoryIndex, job: ClaimedJob) -> list[IndexChunk]:
    """The job's document's chunks in the index, in document order."""
    chunks = index.find(where=KbChunks.document_id.eq(job.document.id))
    return sorted(chunks, key=lambda chunk: chunk[KbChunks.position])


def versions_of(index: InMemoryIndex, job: ClaimedJob) -> dict[int, list[str]]:
    """The texts of the job's document's chunks in the index, by version."""
    texts: dict[int, list[str]] = {}
    for chunk in indexed(index, job):
        texts.setdefault(chunk[KbChunks.document_version], []).append(chunk.text)
    return texts


class FailingPuts:
    """A store that fails every write, like S3 being unreachable."""

    def __init__(self, store: FileStore) -> None:
        self._store = store

    def put(self, key: str, data: BinaryIO, *, size: int, media_type: str) -> None:
        raise ConnectionError("store unreachable")

    def open(self, key: str) -> BinaryIO:
        return self._store.open(key)

    def exists(self, key: str) -> bool:
        return self._store.exists(key)

    def delete(self, key: str) -> None:
        self._store.delete(key)

    def delete_folder(self, folder: str) -> int:
        return self._store.delete_folder(folder)


class FailingOpens:
    """A store that fails every read, like S3 being unreachable."""

    def __init__(self, store: FileStore) -> None:
        self._store = store

    def put(self, key: str, data: BinaryIO, *, size: int, media_type: str) -> None:
        self._store.put(key, data, size=size, media_type=media_type)

    def open(self, key: str) -> BinaryIO:
        raise ConnectionError("store unreachable")

    def exists(self, key: str) -> bool:
        return self._store.exists(key)

    def delete(self, key: str) -> None:
        self._store.delete(key)

    def delete_folder(self, folder: str) -> int:
        return self._store.delete_folder(folder)


# building a ProfileChooser


def test_the_settings_name_the_candidates_and_the_fallback_with_versions(
    chooser: ProfileChooser,
) -> None:
    described = chooser.settings.describe()

    assert described["profiles"] == [VN_REGULATION]
    assert described["fallback_profile"] == PLAIN


def test_the_chunker_is_described_by_its_type_and_options(
    chooser: ProfileChooser,
) -> None:
    chunker = chooser.settings.describe()["chunker"]

    assert chunker["kind"] == "OutlineChunker"
    assert chunker["max_tokens"] == 50
    # what the sizes are counted in, so chunks in words and in tokens differ
    assert chunker["tokenizer"] == "words"
    json.dumps(chooser.settings.describe())  # plain JSON, for provenance


def test_there_is_one_preparer_per_candidate() -> None:
    chooser = make_chooser(profiles=("plain", "vn_regulation"))

    assert len(chooser.preparers) == 2
    assert chooser.settings.profiles == (PLAIN, VN_REGULATION)


@pytest.mark.parametrize(
    ("profiles", "fallback"),
    [(("no_such_profile",), "plain"), (("vn_regulation",), "no_such_profile")],
    ids=["candidate", "fallback"],
)
def test_an_unknown_profile_is_refused_when_building(
    profiles: tuple[str, ...], fallback: str
) -> None:
    # caught when the worker starts, not on the first job
    with pytest.raises(KeyError):
        make_chooser(profiles=profiles, fallback=fallback)


def test_a_chunker_that_cant_be_described_is_refused() -> None:
    class NotADataclass:
        max_tokens = 50

        def chunk(self, document: LogicalDocument) -> tuple[Chunk, ...]:
            return ()

    with pytest.raises(TypeError, match="NotADataclass"):
        ProfileChooser.from_profiles(
            profiles=("vn_regulation",),
            fallback_profile="plain",
            chunker=NotADataclass(),
        )


def test_a_tokenizer_without_a_recorded_name_is_refused() -> None:
    # its sizes would otherwise be recorded as if counted in some known unit
    class CharacterTokenizer:
        def count_tokens(self, text: str) -> int:
            return len(text)

        @property
        def model_max_tokens(self) -> int | None:
            return None

    with pytest.raises(TypeError, match="CharacterTokenizer.*_TOKENIZER_NAMES"):
        ProfileChooser.from_profiles(
            profiles=("vn_regulation",),
            fallback_profile="plain",
            chunker=OutlineChunker(tokenizer=CharacterTokenizer(), max_tokens=50),
        )


# choosing a profile


def test_a_regulation_is_prepared_with_the_first_candidate(
    tmp_path: Path, chooser: ProfileChooser
) -> None:
    chosen = chooser.prepare(write(tmp_path, "a.txt", text_bytes(REGULATION)))

    assert chosen.profile == VN_REGULATION
    assert chosen.profiles_without_structure == ()
    assert chosen.prepared.logical.profile_name == "vn_regulation"


def test_text_without_structure_falls_back(
    tmp_path: Path, chooser: ProfileChooser
) -> None:
    chosen = chooser.prepare(write(tmp_path, "a.txt", text_bytes(PLAIN_TEXT)))

    assert chosen.profile == PLAIN
    assert chosen.profiles_without_structure == (VN_REGULATION,)
    assert chosen.prepared.chunks  # still chunked, by size


def test_candidates_that_find_nothing_are_skipped_in_order(tmp_path: Path) -> None:
    chooser = make_chooser(profiles=("plain", "vn_regulation"))

    chosen = chooser.prepare(write(tmp_path, "a.txt", text_bytes(REGULATION)))

    assert chosen.profile == VN_REGULATION
    assert chosen.profiles_without_structure == (PLAIN,)


def test_the_first_candidate_that_finds_structure_wins(tmp_path: Path) -> None:
    # nothing after it is tried: the order is the preference
    chooser = make_chooser(profiles=("vn_regulation", "plain"))

    chosen = chooser.prepare(write(tmp_path, "a.txt", text_bytes(REGULATION)))

    assert chosen.profile == VN_REGULATION
    assert chosen.profiles_without_structure == ()


def test_with_no_candidates_the_fallback_is_used(tmp_path: Path) -> None:
    chooser = make_chooser(profiles=())

    chosen = chooser.prepare(write(tmp_path, "a.txt", text_bytes(REGULATION)))

    assert chosen.profile == PLAIN
    assert chosen.profiles_without_structure == ()


def test_a_word_file_is_prepared_too(tmp_path: Path, chooser: ProfileChooser) -> None:
    chosen = chooser.prepare(write(tmp_path, "a.docx", docx_of(REGULATION)))

    assert chosen.profile == VN_REGULATION


# the prepare handler


def test_preparing_saves_the_parts_and_the_review_page(
    root: Path, store: FileStore, chooser: ProfileChooser
) -> None:
    job = stage(store, text_bytes(REGULATION))

    run_prepare_job(job, chooser=chooser, store=store)

    assert saved_files(root, job) == ["chunks.jsonl", "logical.json", "review.html"]
    logical, chunks = load(store, job)
    assert logical.profile_name == "vn_regulation"
    assert "Dữ liệu vùng Staging được lưu trữ tối đa 07 ngày" in "".join(
        chunk.text for chunk in chunks
    )


def test_the_review_page_is_html(
    root: Path, store: FileStore, chooser: ProfileChooser
) -> None:
    job = stage(store, text_bytes(REGULATION))

    run_prepare_job(job, chooser=chooser, store=store)

    with store.open(
        f"{prepared_version_folder(job.document.id, 1)}/review.html"
    ) as page:
        html = page.read().decode("utf-8")
    assert "<html" in html.lower()
    assert "Điều 1" in html


def test_the_result_says_how_it_was_prepared(
    store: FileStore, chooser: ProfileChooser
) -> None:
    job = stage(store, text_bytes(PLAIN_TEXT))

    result = run_prepare_job(job, chooser=chooser, store=store)

    assert result.profile == PLAIN
    assert result.profiles_without_structure == (VN_REGULATION,)
    assert result.settings == chooser.settings.describe()


def test_the_provenance_is_the_choice_and_the_settings(
    store: FileStore, chooser: ProfileChooser
) -> None:
    job = stage(store, text_bytes(PLAIN_TEXT))

    provenance = run_prepare_job(job, chooser=chooser, store=store).provenance()

    assert provenance == {
        "profile": PLAIN,
        "profiles_without_structure": [VN_REGULATION],
        "settings": chooser.settings.describe(),
    }
    assert json.loads(json.dumps(provenance)) == provenance  # plain JSON


@pytest.mark.parametrize(
    ("status", "should_index"), [("active", True), ("draft", False)]
)
def test_only_an_active_document_is_indexed_next(
    store: FileStore,
    chooser: ProfileChooser,
    status: DocumentStatus,
    should_index: bool,
) -> None:
    job = stage(store, text_bytes(REGULATION), status=status)

    result = run_prepare_job(job, chooser=chooser, store=store)

    assert result.should_index is should_index


def test_a_word_file_is_handled(
    root: Path, store: FileStore, chooser: ProfileChooser
) -> None:
    job = stage(store, docx_of(REGULATION), media_type=DOCX)

    result = run_prepare_job(job, chooser=chooser, store=store)

    assert result.profile == VN_REGULATION
    assert saved_files(root, job) == ["chunks.jsonl", "logical.json", "review.html"]


def test_preparing_again_replaces_the_old_output(
    store: FileStore, chooser: ProfileChooser
) -> None:
    first = stage(store, text_bytes(PLAIN_TEXT))
    run_prepare_job(first, chooser=chooser, store=store)
    again = stage(store, text_bytes(REGULATION), document_id=first.document.id)

    run_prepare_job(again, chooser=chooser, store=store)

    logical, chunks = load(store, first)
    assert logical.profile_name == "vn_regulation"
    assert "kế hoạch sao lưu" not in "".join(chunk.text for chunk in chunks)


def test_other_versions_are_left_alone(
    store: FileStore, chooser: ProfileChooser
) -> None:
    first = stage(store, text_bytes(PLAIN_TEXT))
    run_prepare_job(first, chooser=chooser, store=store)
    second = stage(
        store, text_bytes(REGULATION), document_id=first.document.id, version=2
    )

    run_prepare_job(second, chooser=chooser, store=store)

    assert load(store, first)[0].profile_name == "plain"
    assert load(store, second)[0].profile_name == "vn_regulation"


def test_the_original_is_left_as_it_was(
    store: FileStore, chooser: ProfileChooser
) -> None:
    content = text_bytes(REGULATION)
    job = stage(store, content)

    run_prepare_job(job, chooser=chooser, store=store)

    with store.open(job.file.storage_key) as original:
        assert original.read() == content


def test_no_temporary_files_are_left(
    temp_root: Path, store: FileStore, chooser: ProfileChooser
) -> None:
    job = stage(store, text_bytes(REGULATION))

    run_prepare_job(job, chooser=chooser, store=store)

    assert list(temp_root.iterdir()) == []


# failures that retrying can't fix


def test_a_missing_original_fails_for_good(
    root: Path, store: FileStore, chooser: ProfileChooser
) -> None:
    job = stage(store, text_bytes(REGULATION), stored=False)

    with pytest.raises(PermanentJobError, match="missing") as raised:
        run_prepare_job(job, chooser=chooser, store=store)

    assert job.file.sha256 in str(raised.value)
    assert raised.value.__suppress_context__  # one clear error in the traceback
    assert saved_files(root, job) == []


def test_an_unsupported_file_type_fails_for_good(
    root: Path, store: FileStore, chooser: ProfileChooser
) -> None:
    job = stage(store, b"%PDF-1.7\n", media_type="application/pdf")

    with pytest.raises(PermanentJobError, match="application/pdf"):
        run_prepare_job(job, chooser=chooser, store=store)

    assert saved_files(root, job) == []


def test_a_damaged_word_file_fails_for_good_and_says_why(
    root: Path, store: FileStore, chooser: ProfileChooser
) -> None:
    job = stage(store, damaged_docx(), media_type=DOCX)

    with pytest.raises(PermanentJobError) as raised:
        run_prepare_job(job, chooser=chooser, store=store)

    cause = raised.value.__cause__
    assert cause is not None  # the full error stays available to the worker's log
    assert type(cause).__name__ in str(raised.value)
    assert len(str(raised.value)) <= 500
    assert saved_files(root, job) == []


def test_failing_for_good_leaves_no_temporary_files(
    temp_root: Path, store: FileStore, chooser: ProfileChooser
) -> None:
    job = stage(store, damaged_docx(), media_type=DOCX)

    with pytest.raises(PermanentJobError):
        run_prepare_job(job, chooser=chooser, store=store)

    assert list(temp_root.iterdir()) == []


# failures worth retrying pass through unchanged


def test_an_os_error_while_preparing_is_not_permanent(
    store: FileStore, chooser: ProfileChooser, monkeypatch: pytest.MonkeyPatch
) -> None:
    def disk_full(self: ProfileChooser, path: Path) -> Any:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ProfileChooser, "prepare", disk_full)
    job = stage(store, text_bytes(REGULATION))

    with pytest.raises(OSError, match="No space left") as raised:
        run_prepare_job(job, chooser=chooser, store=store)

    assert not isinstance(raised.value, PermanentJobError)


def test_a_store_that_cant_be_written_is_not_permanent(
    store: FileStore, chooser: ProfileChooser
) -> None:
    job = stage(store, text_bytes(REGULATION))

    with pytest.raises(ConnectionError):
        run_prepare_job(job, chooser=chooser, store=FailingPuts(store))


# logging


def test_warnings_are_logged_as_counts_without_document_text(
    store: FileStore, chooser: ProfileChooser, caplog: pytest.LogCaptureFixture
) -> None:
    job = stage(store, text_bytes(PLAIN_TEXT))

    with caplog.at_level(logging.INFO, logger="kb_server.indexing.handlers"):
        run_prepare_job(job, chooser=chooser, store=store)

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert str(job.document.id) in messages[0]
    assert PLAIN in messages[0]
    assert "no_headings=1" in messages[0]
    assert "sao lưu" not in messages[0]  # never the document's text


def test_a_clean_preparation_logs_nothing(
    store: FileStore, chooser: ProfileChooser, caplog: pytest.LogCaptureFixture
) -> None:
    job = stage(store, text_bytes(REGULATION))

    with caplog.at_level(logging.INFO, logger="kb_server.indexing.handlers"):
        run_prepare_job(job, chooser=chooser, store=store)

    assert caplog.records == []


# the index job


def test_the_prepared_chunks_are_indexed_in_document_order(
    store: FileStore,
    chooser: ProfileChooser,
    tools: IndexTools,
    index: InMemoryIndex,
) -> None:
    job = stage_prepared(store, text_bytes(REGULATION), chooser=chooser)

    run_index_job(job, tools=tools, store=store)

    _, chunks = load(store, job)
    assert [chunk.text for chunk in indexed(index, job)] == [
        chunk.text for chunk in chunks
    ]


def test_each_chunk_carries_its_groups_collection_and_version(
    store: FileStore,
    chooser: ProfileChooser,
    tools: IndexTools,
    index: InMemoryIndex,
) -> None:
    # ints hash to themselves, so these three come out of a frozenset out of order
    groups = [uuid.UUID(int=2), uuid.UUID(int=1 << 64), uuid.UUID(int=3 << 64)]
    group_ids = frozenset(groups)
    assert list(group_ids) != groups
    job = stage_prepared(
        store,
        text_bytes(REGULATION),
        chooser=chooser,
        version=3,
        group_ids=group_ids,
    )

    run_index_job(job, tools=tools, store=store)

    for chunk in indexed(index, job):
        # sorted, so the same groups always give the same payload
        assert chunk[KbChunks.access_group_ids] == tuple(groups)
        assert chunk[KbChunks.collection_id] == job.document.collection_id
        assert chunk[KbChunks.document_id] == job.document.id
        assert chunk[KbChunks.document_version] == 3


def test_each_chunk_carries_its_position_and_headings(
    store: FileStore,
    chooser: ProfileChooser,
    tools: IndexTools,
    index: InMemoryIndex,
) -> None:
    job = stage_prepared(store, text_bytes(REGULATION), chooser=chooser)

    run_index_job(job, tools=tools, store=store)

    _, prepared = load(store, job)
    chunks = indexed(index, job)
    assert [chunk[KbChunks.position] for chunk in chunks] == list(range(len(prepared)))
    assert [chunk[KbChunks.headings] for chunk in chunks] == [
        chunk.headings for chunk in prepared
    ]
    assert any(chunk[KbChunks.headings] for chunk in chunks)


def test_each_chunk_embeds_its_text_with_its_headings(
    store: FileStore,
    chooser: ProfileChooser,
    tools: IndexTools,
    index: InMemoryIndex,
    embedder: RecordingEmbedder,
) -> None:
    job = stage_prepared(store, text_bytes(REGULATION), chooser=chooser)

    run_index_job(job, tools=tools, store=store)

    _, prepared = load(store, job)
    contextualized = [chunk.contextualize() for chunk in prepared]
    assert [chunk.embedded_text for chunk in indexed(index, job)] == contextualized
    assert embedder.passages == contextualized


def test_chunk_ids_come_from_the_document_version_and_position(
    store: FileStore,
    chooser: ProfileChooser,
    tools: IndexTools,
    index: InMemoryIndex,
) -> None:
    # so indexing again overwrites, and two versions' chunks never collide
    job = stage_prepared(store, text_bytes(REGULATION), chooser=chooser, version=2)

    run_index_job(job, tools=tools, store=store)

    chunks = indexed(index, job)
    assert [chunk.id for chunk in chunks] == [
        uuid.uuid5(job.document.id, f"2:{position}") for position in range(len(chunks))
    ]


def test_a_private_document_is_indexed_without_groups(
    store: FileStore,
    chooser: ProfileChooser,
    tools: IndexTools,
    index: InMemoryIndex,
) -> None:
    job = stage_prepared(store, text_bytes(REGULATION), chooser=chooser)

    run_index_job(job, tools=tools, store=store)

    chunks = indexed(index, job)
    assert chunks
    assert all(chunk[KbChunks.access_group_ids] == () for chunk in chunks)


def test_indexing_a_new_version_leaves_the_old_one_in_place(
    store: FileStore,
    chooser: ProfileChooser,
    tools: IndexTools,
    index: InMemoryIndex,
) -> None:
    # search keeps showing v1 until the worker records v2 as indexed
    first = stage_prepared(store, text_bytes(PLAIN_TEXT), chooser=chooser)
    run_index_job(first, tools=tools, store=store)
    before = versions_of(index, first)
    second = stage_prepared(
        store,
        text_bytes(REGULATION),
        chooser=chooser,
        document_id=first.document.id,
        version=2,
    )

    run_index_job(second, tools=tools, store=store)

    chunks = indexed(index, second)
    assert {chunk[KbChunks.document_version] for chunk in chunks} == {1, 2}
    assert versions_of(index, first)[1] == before[1]


def test_indexing_a_version_again_with_fewer_chunks_leaves_none_of_the_old_ones(
    store: FileStore,
    chooser: ProfileChooser,
    tools: IndexTools,
    index: InMemoryIndex,
) -> None:
    # e.g. prepared again with another profile: validity would serve a leftover,
    # since its version is still the indexed one
    job = stage_prepared(store, text_bytes(REGULATION), chooser=chooser)
    run_index_job(job, tools=tools, store=store)
    logical, chunks = load(store, job)
    assert len(chunks) > 2
    save_prepared(
        store,
        job.document.id,
        job.version.version,
        PreparedParts(logical=logical, chunks=chunks[:2]),
    )

    run_index_job(job, tools=tools, store=store)

    assert [chunk.text for chunk in indexed(index, job)] == [
        chunk.text for chunk in chunks[:2]
    ]


# removing the other versions, after the worker records the new one


def test_removing_other_versions_keeps_only_the_jobs_version(
    store: FileStore,
    chooser: ProfileChooser,
    tools: IndexTools,
    index: InMemoryIndex,
) -> None:
    document_id = uuid.uuid4()
    first, second, third = (
        stage_prepared(
            store,
            text_bytes(REGULATION),
            chooser=chooser,
            document_id=document_id,
            version=version,
        )
        for version in (1, 2, 3)
    )
    for job in (first, second, third):
        run_index_job(job, tools=tools, store=store)

    remove_other_versions(second, index=index)

    assert set(versions_of(index, second)) == {2}


def test_removing_other_versions_leaves_other_documents_alone(
    store: FileStore,
    chooser: ProfileChooser,
    tools: IndexTools,
    index: InMemoryIndex,
) -> None:
    other = stage_prepared(store, text_bytes(PLAIN_TEXT), chooser=chooser)
    run_index_job(other, tools=tools, store=store)
    before = indexed(index, other)
    job = stage_prepared(store, text_bytes(REGULATION), chooser=chooser, version=2)
    run_index_job(job, tools=tools, store=store)

    remove_other_versions(job, index=index)

    assert indexed(index, other) == before


def test_removing_other_versions_again_is_harmless(
    store: FileStore,
    chooser: ProfileChooser,
    tools: IndexTools,
    index: InMemoryIndex,
) -> None:
    job = stage_prepared(store, text_bytes(REGULATION), chooser=chooser)
    run_index_job(job, tools=tools, store=store)
    before = indexed(index, job)

    remove_other_versions(job, index=index)
    remove_other_versions(job, index=index)

    assert indexed(index, job) == before


def test_indexing_again_leaves_the_same_chunks(
    store: FileStore,
    chooser: ProfileChooser,
    tools: IndexTools,
    index: InMemoryIndex,
) -> None:
    job = stage_prepared(store, text_bytes(REGULATION), chooser=chooser)
    run_index_job(job, tools=tools, store=store)
    once = indexed(index, job)

    run_index_job(job, tools=tools, store=store)

    assert indexed(index, job) == once


def test_other_documents_are_left_alone_when_indexing(
    store: FileStore,
    chooser: ProfileChooser,
    tools: IndexTools,
    index: InMemoryIndex,
) -> None:
    other = stage_prepared(store, text_bytes(PLAIN_TEXT), chooser=chooser)
    run_index_job(other, tools=tools, store=store)
    before = indexed(index, other)
    job = stage_prepared(store, text_bytes(REGULATION), chooser=chooser)

    run_index_job(job, tools=tools, store=store)

    assert indexed(index, other) == before


def test_a_version_without_chunks_is_removed_from_the_index(
    store: FileStore,
    chooser: ProfileChooser,
    tools: IndexTools,
    index: InMemoryIndex,
) -> None:
    job = stage_prepared(store, text_bytes(REGULATION), chooser=chooser)
    run_index_job(job, tools=tools, store=store)
    logical, _ = load(store, job)
    save_prepared(
        store,
        job.document.id,
        job.version.version,
        PreparedParts(logical=logical, chunks=()),
    )

    run_index_job(job, tools=tools, store=store)

    assert indexed(index, job) == []


def test_the_index_result_names_the_embedder_and_the_index(
    store: FileStore, chooser: ProfileChooser, tools: IndexTools
) -> None:
    job = stage_prepared(store, text_bytes(REGULATION), chooser=chooser)

    provenance = run_index_job(job, tools=tools, store=store).provenance()

    assert provenance == {"embedder": EMBEDDER, "index": INDEX_LABEL}
    assert json.loads(json.dumps(provenance)) == provenance  # plain JSON


def test_indexing_logs_the_chunk_count_without_document_text(
    store: FileStore,
    chooser: ProfileChooser,
    tools: IndexTools,
    caplog: pytest.LogCaptureFixture,
) -> None:
    job = stage_prepared(store, text_bytes(REGULATION), chooser=chooser)
    _, chunks = load(store, job)
    caplog.clear()

    with caplog.at_level(logging.INFO, logger="kb_server.indexing.handlers"):
        run_index_job(job, tools=tools, store=store)

    assert [record.getMessage() for record in caplog.records] == [
        f"document {job.document.id} v1 indexed: {len(chunks)} chunks"
    ]


# the index job: failures that retrying can't fix


def test_a_version_without_prepared_parts_fails_for_good(
    store: FileStore,
    chooser: ProfileChooser,
    tools: IndexTools,
    index: InMemoryIndex,
) -> None:
    job = stage_prepared(store, text_bytes(REGULATION), chooser=chooser)
    store.delete_folder(prepared_version_folder(job.document.id, job.version.version))

    with pytest.raises(PermanentJobError, match="prepare it again") as raised:
        run_index_job(job, tools=tools, store=store)

    assert str(job.document.id) in str(raised.value)
    assert indexed(index, job) == []


# the index job: failures worth retrying pass through unchanged


def test_an_embedder_error_is_not_permanent(
    store: FileStore,
    chooser: ProfileChooser,
    tools: IndexTools,
    embedder: RecordingEmbedder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def out_of_memory(texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(embedder, "embed_passages", out_of_memory)
    job = stage_prepared(store, text_bytes(REGULATION), chooser=chooser)

    with pytest.raises(RuntimeError, match="out of memory"):
        run_index_job(job, tools=tools, store=store)


def test_a_store_that_cant_be_read_is_not_mistaken_for_missing_parts(
    store: FileStore, chooser: ProfileChooser, tools: IndexTools
) -> None:
    job = stage_prepared(store, text_bytes(REGULATION), chooser=chooser)

    with pytest.raises(ConnectionError):
        run_index_job(job, tools=tools, store=FailingOpens(store))


# the delete job


def test_deleting_removes_the_documents_chunks_from_the_index(
    store: FileStore,
    chooser: ProfileChooser,
    tools: IndexTools,
    index: InMemoryIndex,
) -> None:
    job = stage_prepared(store, text_bytes(REGULATION), chooser=chooser)
    run_index_job(job, tools=tools, store=store)

    run_delete_job(deleting(job), index=index, store=store)

    assert indexed(index, job) == []


def test_deleting_leaves_other_documents_in_the_index(
    store: FileStore,
    chooser: ProfileChooser,
    tools: IndexTools,
    index: InMemoryIndex,
) -> None:
    other = stage_prepared(store, text_bytes(PLAIN_TEXT), chooser=chooser)
    run_index_job(other, tools=tools, store=store)
    before = indexed(index, other)
    job = stage_prepared(store, text_bytes(REGULATION), chooser=chooser)
    run_index_job(job, tools=tools, store=store)

    run_delete_job(deleting(job), index=index, store=store)

    assert indexed(index, other) == before


def test_deleting_removes_every_prepared_version(
    root: Path, store: FileStore, chooser: ProfileChooser, index: InMemoryIndex
) -> None:
    document_id = uuid.uuid4()
    stage_prepared(
        store, text_bytes(PLAIN_TEXT), chooser=chooser, document_id=document_id
    )
    job = stage_prepared(
        store,
        text_bytes(REGULATION),
        chooser=chooser,
        document_id=document_id,
        version=2,
    )

    run_delete_job(deleting(job), index=index, store=store)

    assert not (root / prepared_folder(document_id)).exists()


def test_the_delete_result_counts_the_prepared_files_it_deleted(
    root: Path, store: FileStore, chooser: ProfileChooser, index: InMemoryIndex
) -> None:
    job = stage_prepared(store, text_bytes(REGULATION), chooser=chooser)
    saved = len(saved_files(root, job))

    result = run_delete_job(deleting(job), index=index, store=store)

    assert saved > 0
    assert result.prepared_files_deleted == saved


def test_deleting_leaves_other_documents_prepared_output(
    store: FileStore, chooser: ProfileChooser, index: InMemoryIndex
) -> None:
    other = stage_prepared(store, text_bytes(PLAIN_TEXT), chooser=chooser)
    job = stage_prepared(store, text_bytes(REGULATION), chooser=chooser)

    run_delete_job(deleting(job), index=index, store=store)

    assert load_prepared(store, other.document.id, other.version.version) is not None


def test_deleting_leaves_the_original_for_the_worker_to_decide(
    store: FileStore, chooser: ProfileChooser, index: InMemoryIndex
) -> None:
    # another document may share the original; only the worker can tell
    job = stage_prepared(store, text_bytes(REGULATION), chooser=chooser)

    run_delete_job(deleting(job), index=index, store=store)

    assert store.exists(job.file.storage_key)


def test_deleting_a_document_never_prepared_or_indexed_does_nothing(
    store: FileStore, index: InMemoryIndex
) -> None:
    job = stage(store, text_bytes(REGULATION))

    run_delete_job(deleting(job), index=index, store=store)

    assert indexed(index, job) == []


def test_deleting_again_after_a_crash_is_harmless(
    store: FileStore,
    chooser: ProfileChooser,
    tools: IndexTools,
    index: InMemoryIndex,
) -> None:
    job = stage_prepared(store, text_bytes(REGULATION), chooser=chooser)
    run_index_job(job, tools=tools, store=store)

    run_delete_job(deleting(job), index=index, store=store)
    run_delete_job(deleting(job), index=index, store=store)

    assert indexed(index, job) == []


def deleting(job: ClaimedJob) -> ClaimedJob:
    """The same document, claimed by a delete job."""
    return dataclasses.replace(
        job, job=dataclasses.replace(job.job, kind="delete", version=None)
    )
