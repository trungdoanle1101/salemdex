import uuid
from collections.abc import Mapping
from collections.abc import Set as AbstractSet

from sqlalchemy.orm import Session, sessionmaker

from kb_server.domain.chunks import Guarded, KbChunks
from kb_server.domain.credentials import Principal
from kb_server.domain.errors import InvalidRequest
from kb_server.domain.search import DocumentSearched, Passage
from kb_server.search.statements import BuildStatement
from kb_server.services.access import read_access
from kb_server.storage.repositories.documents import DocumentRepository
from kb_server.storage.repositories.users import UserRepository
from salemdex_retrieval import Engine, Everything, or_
from salemdex_retrieval.engine import Passage as FoundPassage
from salemdex_retrieval.filters import Filter


class SearchService:
    """Searches the index for passages, showing each asker only what they may read.

    Unlike the other services, it lives for the whole app and holds a session factory,
    not a request's session: each database read is a short transaction of its own, so
    no connection is held while the models embed and rerank.
    """

    def __init__(
        self,
        *,
        max_limit: int,
        session_factory: sessionmaker[Session],
        retrieval_engine: Engine,
        build_statement: BuildStatement,
    ) -> None:
        """Build the service.

        Args:
            max_limit: The most passages one search may ask for (`KB_SEARCH_MAX_LIMIT`).
            session_factory: Opens the short transactions for the asker's groups and
                the titles.
            retrieval_engine: Runs statements, with the validity check at every gate.
            build_statement: Turns a question into the statement to run.
        """
        self._max_limit = max_limit
        self._session_factory = session_factory
        self._retrieval_engine = retrieval_engine
        self._build_statement = build_statement

    def search(
        self, principal: Principal, question: str, *, limit: int | None = None
    ) -> list[Passage]:
        """Return the passages that best answer `question`, best first.

        Args:
            principal: Who asks; they see only documents they may read.
            question: What to search for.
            limit: The most passages to return; left out, the maximum allowed.

        Returns:
            At most `limit` passages. Fewer when fewer are found or visible: a hidden
            document is never hinted at.

        Raises:
            InvalidRequest: `limit` is above the maximum allowed.
        """
        if limit is None:
            limit = self._max_limit
        if limit > self._max_limit:
            raise InvalidRequest(f"limit can be at most {self._max_limit}")

        retrieval = self._retrieval_engine.session(
            access=self._read_access_filter(principal)
        )
        found = retrieval.execute(self._build_statement(question, limit=limit))
        titles = self._read_titles({_document_id(passage) for passage in found})
        # A document deleted or deactivated since the last gate has no title: dropped
        return [
            _to_passage(passage, titles=titles)
            for passage in found
            if _document_id(passage) in titles
        ]

    def _read_access_filter(self, principal: Principal) -> Filter:
        """Return the filter of what `principal` may see, read in a short transaction."""
        with self._session_factory.begin() as session:
            access = read_access(principal, users=UserRepository(session))
        if access is None:
            return Everything()
        # An empty `in_` matches nothing, so a reader needs no case of their own
        return or_(
            Guarded.access_group_ids.any_of(access.group_ids),
            Guarded.collection_id.in_(access.managed_collection_ids),
        )

    def _read_titles(
        self, document_ids: AbstractSet[uuid.UUID]
    ) -> dict[uuid.UUID, str]:
        """Return the titles of the served documents among these, in a short transaction."""
        with self._session_factory.begin() as session:
            return DocumentRepository(session).served_titles(document_ids)


def _document_id(passage: FoundPassage) -> uuid.UUID:
    """Return the id of the document `passage` comes from."""
    return passage.chunk[KbChunks.document_id]


def _to_passage(passage: FoundPassage, *, titles: Mapping[uuid.UUID, str]) -> Passage:
    """Return a found passage as search shows it, with its document's title."""
    chunk = passage.chunk
    return Passage(
        text=chunk.text,
        headings=tuple(chunk[KbChunks.headings]),
        document=DocumentSearched(
            id=chunk[KbChunks.document_id],
            title=titles[chunk[KbChunks.document_id]],
            version=chunk[KbChunks.document_version],
        ),
    )
