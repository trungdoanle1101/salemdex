import logging
import shutil
import tempfile
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Self

from document_processor.chunking import Chunk as DocChunk
from document_processor.chunking.chunker import Chunker
from document_processor.chunking.tokenizer import WordTokenizer
from document_processor.logical.profile import Profile
from document_processor.logical.profiles import PROFILES
from document_processor.preparation.preparer import PreparedDocument, Preparer
from kb_server.domain.chunks import KbChunks
from kb_server.domain.documents import Document, DocumentVersion
from kb_server.domain.files import DOCX, TEXT, StoredFile
from kb_server.domain.jobs import IndexingJob
from kb_server.indexing.prepared import (
    PreparationSettings,
    PreparedParts,
    delete_prepared,
    load_prepared,
    save_prepared,
    save_review_page,
)
from kb_server.storage.files import FileStore
from salemdex_retrieval import Chunk as IndexChunk
from salemdex_retrieval.embedders import DenseBinding, embed_chunks
from salemdex_retrieval.filters import and_, not_
from salemdex_retrieval.index import Index

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ClaimedJob:
    """What a job is about, loaded when it's claimed."""

    job: IndexingJob
    document: Document
    version: DocumentVersion
    file: StoredFile
    group_ids: frozenset[uuid.UUID]


class PermanentJobError(Exception):
    """Retrying can't help (a missing or unreadable file): the job fails at once."""


@dataclass(frozen=True, kw_only=True)
class ProfileChoice:
    prepared: PreparedDocument
    profile: str  # name@version that was used
    profiles_without_structure: tuple[str, ...]  # in the order they were tried


@dataclass(frozen=True, kw_only=True)
class ProfileChooser:
    """Prepares a file with the first candidate profile that finds structure, else the fallback.

    Carries the PreparationSettings that describe its preparers.
    """

    preparers: tuple[Preparer, ...]
    fallback_preparer: Preparer
    settings: PreparationSettings

    @classmethod
    def from_profiles(
        cls,
        *,
        profiles: Sequence[str],
        fallback_profile: str,
        chunker: Chunker,
    ) -> Self:
        resolved = [PROFILES.get(name) for name in profiles]
        fallback = PROFILES.get(fallback_profile)
        settings = PreparationSettings(
            profiles=tuple(_profile_label(p) for p in resolved),
            fallback_profile=_profile_label(fallback),
            chunker=_describe_chunker(chunker),
        )

        return cls(
            preparers=tuple(Preparer(profile=p, chunker=chunker) for p in resolved),
            fallback_preparer=Preparer(profile=fallback, chunker=chunker),
            settings=settings,
        )

    def prepare(self, path: Path) -> ProfileChoice:
        profiles_without_structure: list[str] = []
        for preparer in self.preparers:
            prepared = preparer.prepare(path)
            if any(node.is_section for node in prepared.logical.nodes):
                return ProfileChoice(
                    prepared=prepared,
                    profile=_profile_label_of(prepared),
                    profiles_without_structure=tuple(profiles_without_structure),
                )
            profiles_without_structure.append(_profile_label_of(prepared))
        prepared = self.fallback_preparer.prepare(path)
        return ProfileChoice(
            prepared=prepared,
            profile=_profile_label_of(prepared),
            profiles_without_structure=tuple(profiles_without_structure),
        )


@dataclass(frozen=True, kw_only=True)
class PrepareResult:
    profile: str
    profiles_without_structure: tuple[str, ...]
    settings: dict[str, Any]  # chooser.settings.describe()
    should_index: bool

    def provenance(self) -> dict[str, Any]:
        """Plain JSON, for document_versions.provenance["preparation"]"""
        return {
            "profile": self.profile,
            "profiles_without_structure": list(self.profiles_without_structure),
            "settings": self.settings,
        }


@dataclass(frozen=True, kw_only=True)
class IndexTools:
    """What an index job writes with: the model, the index, and the names to record.

    Attributes:
        binding: `KbChunks.dense`, paired with the model that fills it.
        embedder_identity: Which model that is, recorded in each version's index
            provenance; the embedder itself doesn't say.
        index: The index of `KbChunks` the chunks go into.
        index_label: Where that index lives, e.g. "qdrant:kb_chunks", for provenance;
            only the wiring knows it.
    """

    binding: DenseBinding
    embedder_identity: str
    index: Index
    index_label: str


@dataclass(frozen=True, kw_only=True)
class IndexResult:
    """What filled a version's vectors and where they went, for its provenance.

    Attributes:
        embedder: The model's identity.
        index: The index's label.
    """

    embedder: str
    index: str

    def provenance(self) -> dict[str, str]:
        """Plain JSON, for document_versions.provenance["index"]"""
        return {
            "embedder": self.embedder,
            "index": self.index,
        }


@dataclass(frozen=True, kw_only=True)
class DeleteResult:
    prepared_files_deleted: int


@dataclass(frozen=True, kw_only=True)
class JobTools:
    profile_chooser: ProfileChooser
    index_tools: IndexTools
    file_store: FileStore


def run_prepare_job(
    job: ClaimedJob, *, chooser: ProfileChooser, store: FileStore
) -> PrepareResult:
    file = job.file
    extension = _file_extension_for(file.media_type)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        source = _copy_original(store, file, work / f"document{extension}")
        choice = _prepare_file(chooser, source)
        _save(store, job, choice, review_page=work / "review.html")
    _log_warnings(job, choice)
    return PrepareResult(
        profile=choice.profile,
        profiles_without_structure=choice.profiles_without_structure,
        settings=chooser.settings.describe(),
        should_index=job.document.status == "active",
    )


def run_index_job(
    job: ClaimedJob, *, tools: IndexTools, store: FileStore
) -> IndexResult:
    """Make this version's prepared chunks the document's only chunks in the index.

    Safe to repeat: chunk ids come from the document, version and position, and
    `replace` removes every chunk the document had before writing these.

    Args:
        job: The claimed index job.
        tools: The model and index to write with.
        store: Where the version's prepared parts are.

    Returns:
        The model and index, for the version's provenance.

    Raises:
        PermanentJobError: The version has no prepared parts.
    """
    parts = _load_parts(job, store=store)
    chunks = _to_index_chunks(parts.chunks, job=job)
    embedded = embed_chunks(chunks, bindings=[tools.binding])
    tools.index.replace(
        where=and_(
            KbChunks.document_id.eq(job.document.id),
            KbChunks.document_version.eq(job.version.version),
        ),
        with_=embedded,
    )
    logger.info(
        "document %s v%d indexed: %d chunks",
        job.document.id,
        job.version.version,
        len(chunks),
    )
    return IndexResult(embedder=tools.embedder_identity, index=tools.index_label)


def remove_other_versions(job: ClaimedJob, *, index: Index) -> None:
    """Remove the document's chunks of every version but the job's.

    The worker calls it after the commit that makes the job's version the one search
    shows, never before: until then the old version is what readers see. Safe to
    repeat, and it also removes what an earlier cleanup left behind.

    Args:
        job: The index job just recorded; its version is the one to keep.
        index: The index of `KbChunks` holding the document's chunks.
    """
    index.delete(
        where=and_(
            KbChunks.document_id.eq(job.document.id),
            not_(KbChunks.document_version.eq(job.version.version)),
        )
    )


def run_delete_job(job: ClaimedJob, *, index: Index, store: FileStore) -> DeleteResult:
    """Remove the document's chunks from the index, then its prepared output.

    Safe to repeat: deleting what's already gone does nothing.

    Args:
        job: The claimed delete job.
        index: The index of `KbChunks` holding the document's chunks.
        store: Where the document's prepared output is.

    Returns:
        How many prepared files were deleted.
    """
    index.delete(where=KbChunks.document_id.eq(job.document.id))
    files_deleted = delete_prepared(store, job.document.id)
    logger.info(
        "document %s removed from the index, %d prepared files deleted",
        job.document.id,
        files_deleted,
    )
    return DeleteResult(prepared_files_deleted=files_deleted)


def _describe_chunker(chunker: Chunker) -> dict[str, Any]:
    """Return the chunker's type, its options and the name of its tokenizer.

    Raises:
        TypeError: The chunker isn't a dataclass instance, or its tokenizer has no
            name in `_TOKENIZER_NAMES`.
    """
    if not is_dataclass(chunker) or isinstance(chunker, type):
        raise TypeError(
            f"{type(chunker).__name__} must be a dataclass instance to be described"
        )

    options = {
        f.name: getattr(chunker, f.name)
        for f in fields(chunker)
        if f.name != "tokenizer"
    }
    described = {"kind": type(chunker).__name__, **options}
    tokenizer = getattr(chunker, "tokenizer", None)
    if tokenizer is not None:
        described["tokenizer"] = _tokenizer_name(tokenizer)
    return described


# What the chunk sizes are counted in, by tokenizer. A fixed name rather than the class
# name, so renaming a class doesn't make every prepared version look stale.
_TOKENIZER_NAMES: Mapping[type, str] = {WordTokenizer: "words"}


def _tokenizer_name(tokenizer: object) -> str:
    """Return the name provenance records for `tokenizer`.

    Raises:
        TypeError: The tokenizer's type has no name in `_TOKENIZER_NAMES`.
    """
    try:
        return _TOKENIZER_NAMES[type(tokenizer)]
    except KeyError:
        raise TypeError(
            f"{type(tokenizer).__name__} has no name to record in provenance; "
            "add it to _TOKENIZER_NAMES"
        ) from None


def _profile_label(profile: Profile) -> str:
    return f"{profile.name}@{profile.version}"


def _file_extension_for(media_type: str) -> str:
    if media_type == DOCX:
        return ".docx"
    elif media_type == TEXT:
        return ".txt"
    else:
        raise PermanentJobError(f"unsupported media type: {media_type}")


def _copy_original(store: FileStore, file: StoredFile, destination: Path) -> Path:
    try:
        original = store.open(file.storage_key)
    except FileNotFoundError:
        raise PermanentJobError(f"stored file {file.sha256} is missing") from None
    with original, destination.open("wb") as out:
        shutil.copyfileobj(original, out)
    return destination


def _prepare_file(chooser: ProfileChooser, path: Path) -> ProfileChoice:
    try:
        return chooser.prepare(path)
    except OSError:
        raise
    except Exception as error:
        raise PermanentJobError(
            f"couldn't prepare the file: {type(error).__name__}: {error}"[:500]
        ) from error


def _save(
    store: FileStore, job: ClaimedJob, choice: ProfileChoice, *, review_page: Path
) -> None:
    save_prepared(
        store,
        job.document.id,
        job.version.version,
        PreparedParts(logical=choice.prepared.logical, chunks=choice.prepared.chunks),
    )
    choice.prepared.write_review_page(review_page)
    save_review_page(store, job.document.id, job.version.version, review_page)


def _log_warnings(job: ClaimedJob, choice: ProfileChoice) -> None:
    counts = Counter(w.code for w in choice.prepared.summarize().warnings)
    if counts:
        logger.warning(
            "document %s v%d prepared with %s: %s",
            job.document.id,
            job.version.version,
            choice.profile,
            ", ".join(f"{code}={n}" for code, n in sorted(counts.items())),
        )


def _profile_label_of(prepared: PreparedDocument) -> str:
    return f"{prepared.logical.profile_name}@{prepared.logical.profile_version}"


def _load_parts(job: ClaimedJob, *, store: FileStore) -> PreparedParts:
    parts = load_prepared(store, job.document.id, job.version.version)
    if parts is None:
        raise PermanentJobError(
            f"document {job.document.id} v{job.version.version} has no prepared "
            "parts; prepare it again"
        )
    return parts


def _to_index_chunks(
    chunks: Sequence[DocChunk], *, job: ClaimedJob
) -> list[IndexChunk]:
    """Return the prepared chunks as `KbChunks` chunks, numbered in document order."""
    return [
        _to_index_chunk(chunk, position=position, job=job)
        for position, chunk in enumerate(chunks)
    ]


def _to_index_chunk(chunk: DocChunk, *, position: int, job: ClaimedJob) -> IndexChunk:
    """Return one prepared chunk as the index stores it.

    Its id is `uuid5(document_id, "<version>:<position>")`: writing the same version
    again overwrites it, and a new version's chunks never overwrite the old one's
    while both are in the index. Group ids are sorted, so a repeat writes the same
    payload.
    """
    return IndexChunk(
        id=uuid.uuid5(job.document.id, f"{job.version.version}:{position}"),
        text=chunk.text,
        embedded_text=chunk.contextualize(),
        field_values=[
            KbChunks.access_group_ids.value(sorted(job.group_ids)),
            KbChunks.collection_id.value(job.document.collection_id),
            KbChunks.document_id.value(job.document.id),
            KbChunks.document_version.value(job.version.version),
            KbChunks.position.value(position),
            KbChunks.headings.value(chunk.headings),
        ],
    )
