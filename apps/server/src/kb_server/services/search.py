from kb_server.domain.chunks import Guarded
from kb_server.domain.credentials import Principal
from kb_server.search.statements import BuildStatement
from kb_server.storage.repositories.users import UserRepository
from salemdex_retrieval import Engine, Everything, or_
from salemdex_retrieval.engine import Passage
from salemdex_retrieval.filters import Filter


class SearchService:
    def __init__(
        self,
        *,
        users: UserRepository,
        retrieval_engine: Engine,
        build_statement: BuildStatement,
    ) -> None:
        self._retrieval_engine = retrieval_engine
        self._users = users
        self._build_statement = build_statement

    def search(
        self, principal: Principal, question: str, *, limit: int
    ) -> list[Passage]:
        session = self._retrieval_engine.session(access=self._access_filter(principal))
        statement = self._build_statement(question, limit=limit)
        return session.execute(statement)

    def _access_filter(self, principal: Principal) -> Filter:
        if principal.is_admin:
            return Everything()

        groups = self._users.list_group_ids(principal.user.id)
        reader = Guarded.access_group_ids.any_of(groups)
        if principal.user.role == "manager":
            managed_collections = self._users.list_managed_collection_ids(
                principal.user.id
            )
            return or_(
                reader,
                Guarded.collection_id.in_(managed_collections),
            )

        return reader
