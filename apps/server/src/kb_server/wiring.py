from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from argon2 import PasswordHasher
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from document_processor.chunking.outline_chunker import OutlineChunker
from document_processor.chunking.tokenizer import WordTokenizer
from kb_server.domain.chunks import KbChunks
from kb_server.indexing.handlers import (
    IndexTools,
    JobTools,
    ProfileChooser,
)
from kb_server.indexing.worker import Worker
from kb_server.search.statements import CROSS_ENCODER, DENSE, dense_then_rerank
from kb_server.search.validity import ServedVersionsOnly
from kb_server.services.auth import AuthService
from kb_server.services.collections import CollectionService
from kb_server.services.documents import DocumentService
from kb_server.services.groups import GroupService
from kb_server.services.passwords import Passwords
from kb_server.services.search import SearchService
from kb_server.services.users import UserService
from kb_server.settings import Settings
from kb_server.storage.files import FileStore, LocalFileStore
from kb_server.storage.repositories.collections import CollectionRepository
from kb_server.storage.repositories.credentials import CredentialRepository
from kb_server.storage.repositories.documents import DocumentRepository
from kb_server.storage.repositories.files import StoredFileRepository
from kb_server.storage.repositories.groups import GroupRepository
from kb_server.storage.repositories.jobs import JobRepository
from kb_server.storage.repositories.users import UserRepository
from salemdex_retrieval import Engine as SalemdexEngine
from salemdex_retrieval import VectorSource
from salemdex_retrieval.embedders import DenseBinding, DenseEmbedder
from salemdex_retrieval.engine import PreparedStatement
from salemdex_retrieval.errors import ModelLoadError, SchemaMismatchError
from salemdex_retrieval.index import Index
from salemdex_retrieval.rerankers import Reranker

if TYPE_CHECKING:
    # for annotations only: importing these for real is left to the worker
    from qdrant_client import QdrantClient

    from document_processor.core.registry import Registry
    from document_processor.integrations.huggingface.reference import ModelReference


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, kw_only=True)
class Services:
    settings: Settings
    sql_engine: Engine
    files: FileStore
    session_factory: sessionmaker[Session]
    passwords: Passwords
    clock: Callable[[], datetime] = utc_now


def build_services(settings: Settings) -> Services:
    engine = create_engine(url=settings.database_url, pool_pre_ping=True)
    session_factory = sessionmaker(engine)
    return Services(
        settings=settings,
        sql_engine=engine,
        files=build_file_store(settings),
        session_factory=session_factory,
        passwords=Passwords(hasher=PasswordHasher()),
    )


def build_worker(services: Services, *, job_tools: JobTools) -> Worker:
    settings = services.settings
    return Worker(
        session_factory=services.session_factory,
        tools=job_tools,
        max_attempts=settings.job_max_attempts,
        poll_interval=timedelta(seconds=settings.worker_poll_seconds),
        job_timeout=timedelta(minutes=settings.job_timeout_minutes),
        now=services.clock,
    )


def close_services(services: Services) -> None:
    services.sql_engine.dispose()


def build_user_service(session: Session, services: Services) -> UserService:
    return UserService(
        users=UserRepository(session),
        groups=GroupRepository(session),
        credentials=CredentialRepository(session),
        passwords=services.passwords,
        now=services.clock,
    )


def build_auth_service(session: Session, services: Services) -> AuthService:
    return AuthService(
        users=UserRepository(session),
        credentials=CredentialRepository(session),
        passwords=services.passwords,
        session_ttl=timedelta(hours=services.settings.session_ttl_hours),
        now=services.clock,
    )


def build_group_service(session: Session) -> GroupService:
    return GroupService(groups=GroupRepository(session), users=UserRepository(session))


def build_collection_service(session: Session) -> CollectionService:
    return CollectionService(
        collections=CollectionRepository(session),
        groups=GroupRepository(session),
        users=UserRepository(session),
    )


def build_document_service(session: Session, services: Services) -> DocumentService:
    return DocumentService(
        documents=DocumentRepository(session),
        collections=CollectionRepository(session),
        users=UserRepository(session),
        groups=GroupRepository(session),
        stored_files=StoredFileRepository(session),
        jobs=JobRepository(session),
        max_upload_bytes=services.settings.max_upload_mb * 1024 * 1024,
        files=services.files,
    )


def build_file_store(settings: Settings) -> FileStore:
    match settings.storage_kind:
        case "local":
            assert settings.storage_path is not None
            return LocalFileStore(settings.storage_path)
        case "s3":
            raise NotImplementedError("s3 storage not yet implemented")


def build_search_service(
    services: Services, *, prepared_statement: PreparedStatement
) -> SearchService:
    """Return the search service the API holds for its whole life.

    Args:
        services: What every process builds; its settings and session factory are
            used.
        prepared_statement: The statement every search runs, prepared by
            `open_search_tools` with the models.

    Returns:
        The service, built once, not per request.
    """
    return SearchService(
        max_limit=services.settings.search_max_limit,
        session_factory=services.session_factory,
        prepared_statement=prepared_statement,
    )


class ToolSetupError(Exception):
    """The worker's or the API's tools can't be built.

    A setting is missing or wrong, a model can't be loaded, or Qdrant can't be reached
    or holds a collection that doesn't fit the schema. The message says which step failed, followed by the underlying reason. It names a
    KB_* variable only where this module checks that variable itself, so the name can't
    go stale when a library starts failing for another reason.
    """


@contextmanager
def open_job_tools(services: Services) -> Generator[JobTools, None, None]:
    """Yield the worker's tools, closing the Qdrant client however the worker stops.

    Built in this order: the embedder and its identity, the profile chooser (chunks
    sized in words), then the index of `KbChunks`, whose payload indexes
    `QdrantIndex.open` creates from the schema.

    Args:
        services: What every process builds; only its settings and file store are used.

    Yields:
        The profile chooser, the index tools and the file store.

    Raises:
        ToolSetupError: A setting is missing or wrong, the model can't be loaded, or
            Qdrant can't be reached or holds a collection that doesn't fit the schema.
    """
    settings = services.settings
    profile_chooser = _build_profile_chooser(settings)
    embedder, embedder_identity = _build_embedder(settings)
    qdrant_client = _open_qdrant_client(settings)
    try:
        index = _open_index(settings, client=qdrant_client)
        yield JobTools(
            profile_chooser=profile_chooser,
            index_tools=IndexTools(
                binding=DenseBinding(
                    vector=KbChunks.dense,
                    embedder=embedder,
                ),
                embedder_identity=embedder_identity,
                index=index,
                index_label=f"qdrant:{settings.qdrant_collection}",
            ),
            file_store=services.files,
        )
    finally:
        qdrant_client.close()


@contextmanager
def open_search_tools(
    services: Services,
) -> Generator[PreparedStatement, None, None]:
    """Yield the API's prepared statement, closing the Qdrant client however it stops.

    Built in this order: the embedder, the index of `KbChunks`, the reranker, then the
    engine registering them with the validity check, which prepares the statement.
    The source returns `KB_RERANK_DEPTH` candidates, all of which the reranker reads.

    A statement the engine refuses raises `StatementError`, not `ToolSetupError`: the
    statement is code, not configuration, so that's a bug, shown with its traceback.

    Args:
        services: What every process builds; its settings and session factory (for
            the validity check) are used.

    Yields:
        The prepared statement: dense retrieval, then reranking.

    Raises:
        ToolSetupError: A setting is missing or wrong, a model can't be loaded, or
            Qdrant can't be reached or holds a collection that doesn't fit the schema.
    """
    settings = services.settings
    embedder, _ = _build_embedder(settings)
    qdrant_client = _open_qdrant_client(settings)
    try:
        engine = SalemdexEngine(
            validity=ServedVersionsOnly(services.session_factory),
            sources={
                DENSE: VectorSource(
                    index=_open_index(settings, client=qdrant_client),
                    binding=DenseBinding(vector=KbChunks.dense, embedder=embedder),
                )
            },
            rerankers={CROSS_ENCODER: _build_reranker(settings)},
        )
        yield engine.prepare(dense_then_rerank(depth=settings.rerank_depth))
    finally:
        qdrant_client.close()


def _open_qdrant_client(settings: Settings) -> "QdrantClient":
    """Return a client for `KB_QDRANT_URL`; it connects only when first used."""
    # Imported here, like the models: the commands that never search or index don't
    # install it, and importing it at the top would load it in every process
    from qdrant_client import QdrantClient

    api_key = settings.qdrant_api_key
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=api_key.get_secret_value() if api_key is not None else None,
    )


def _build_embedder(settings: Settings) -> tuple[DenseEmbedder, str]:
    """Return the embedder loaded from `KB_MODELS_FOLDER`, and its identity.

    The identity names the model by its hub id and commit, since the embedder doesn't
    name itself; the index job records it in each version's provenance.

    Raises:
        ToolSetupError: The folder isn't set, the spec is unknown, the model isn't
            in the folder, or it can't be loaded.
    """
    from document_processor.integrations.huggingface.model_specs import (
        HUGGINGFACE_MODEL_SPECS,
    )
    from salemdex_retrieval.embedders.huggingface import HuggingFaceDenseEmbedder

    spec, folder = _find_model(
        settings.embedder_spec,
        specs=HUGGINGFACE_MODEL_SPECS,
        models_folder=settings.models_folder,
    )
    ref = spec.reference
    identity = f"huggingface:{ref.hub_model_id}@{ref.commit_hash}"
    try:
        embedder = HuggingFaceDenseEmbedder(
            model=folder,
            device=settings.embedder_device,
            batch_size=settings.embedder_batch_size,
        )
    except (ModelLoadError, ValueError) as error:
        raise ToolSetupError(
            f"can't load {settings.embedder_spec} from {folder}: {error}"
        ) from error
    return embedder, identity


def _build_reranker(settings: Settings) -> Reranker:
    """Return the reranker loaded from `KB_MODELS_FOLDER`.

    Raises:
        ToolSetupError: The folder isn't set, the spec is unknown, the model isn't
            in the folder, or it can't be loaded.
    """
    from document_processor.integrations.huggingface.model_specs import (
        HUGGINGFACE_RERANKER_SPECS,
    )
    from salemdex_retrieval.rerankers.huggingface import HuggingFaceReranker

    spec, folder = _find_model(
        settings.reranker_spec,
        specs=HUGGINGFACE_RERANKER_SPECS,
        models_folder=settings.models_folder,
    )
    try:
        return HuggingFaceReranker(
            model=folder,
            max_length=spec.max_length,
            device=settings.reranker_device,
            batch_size=settings.reranker_batch_size,
        )
    except (ModelLoadError, ValueError) as error:
        raise ToolSetupError(
            f"can't load {settings.reranker_spec} from {folder}: {error}"
        ) from error


class _ModelSpec(Protocol):
    """A spec naming a model by its hub id and commit."""

    @property
    def reference(self) -> "ModelReference": ...


def _find_model[S: _ModelSpec](
    spec_name: str, *, specs: "Registry[S]", models_folder: Path | None
) -> tuple[S, Path]:
    """Return the spec named `spec_name`, and its model's folder in `models_folder`.

    Raises:
        ToolSetupError: `models_folder` isn't set, the spec is unknown, or the model
            isn't in the folder.
    """
    from document_processor.integrations.huggingface.download import model_folder

    if models_folder is None:
        raise ToolSetupError(
            "KB_MODELS_FOLDER is not set: it names the folder holding all models"
        )
    try:
        spec = specs.get(spec_name)
    except KeyError as error:
        raise ToolSetupError(f"can't find the model: {error.args[0]}") from error
    folder = model_folder(spec.reference, models_folder=models_folder)
    # Checked here: given a path that isn't a folder, sentence-transformers would take
    # it for a hub id and fail confusingly, or try to download it
    if not folder.is_dir():
        raise ToolSetupError(
            f"{spec_name!r} isn't in KB_MODELS_FOLDER: {folder} is not a folder"
        )
    return spec, folder


def _build_profile_chooser(settings: Settings) -> ProfileChooser:
    """Return the chooser, its chunks sized in words (`KB_CHUNK_WORDS`).

    Raises:
        ToolSetupError: The chunk size is refused, or a profile is unknown.
    """
    try:
        chunker = OutlineChunker(
            tokenizer=WordTokenizer(), max_tokens=settings.chunk_words
        )
    except ValueError as error:
        raise ToolSetupError(f"can't build the chunker: {error}") from error
    try:
        return ProfileChooser.from_profiles(
            profiles=settings.profiles,
            fallback_profile=settings.fallback_profile,
            chunker=chunker,
        )
    except KeyError as error:
        raise ToolSetupError(f"can't set up the profiles: {error.args[0]}") from error


def _open_index(settings: Settings, *, client: "QdrantClient") -> Index:
    """Return the index of `KbChunks` in the Qdrant collection, created on first use.

    An existing collection is checked against the schema: its vectors, payload indexes
    and the fields its points were written with.

    Raises:
        ToolSetupError: Qdrant can't be reached or refuses the request, or the
            collection doesn't fit the schema (e.g. written by the old index code).
    """
    from qdrant_client.http.exceptions import (
        ResponseHandlingException,
        UnexpectedResponse,
    )

    from salemdex_retrieval.index.qdrant import QdrantIndex

    try:
        return QdrantIndex.open(
            schema=KbChunks,
            client=client,
            collection=settings.qdrant_collection,
        )
    # a connection failure: the URL is the only input to connecting
    except ResponseHandlingException as error:
        raise ToolSetupError(
            f"can't reach Qdrant at {settings.qdrant_url} (KB_QDRANT_URL). Is it running?"
        ) from error
    except UnexpectedResponse as error:
        raise ToolSetupError(
            f"Qdrant at {settings.qdrant_url} refused the request: {error}"
        ) from error
    except SchemaMismatchError as error:
        raise ToolSetupError(
            f"can't open the Qdrant collection {settings.qdrant_collection}: {error}"
        ) from error
