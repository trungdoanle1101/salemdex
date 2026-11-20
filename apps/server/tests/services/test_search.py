"""Searching: which passages each asker is shown."""

import uuid
from collections.abc import Callable

import pytest
from sqlalchemy import update
from sqlalchemy.orm import Session

from kb_server.domain.collections import Collection
from kb_server.domain.credentials import Principal
from kb_server.domain.documents import Document
from kb_server.domain.errors import InvalidRequest
from kb_server.domain.groups import Group
from kb_server.domain.users import User, UserRole
from kb_server.search.statements import DenseThenRerank
from kb_server.services.auth import AuthService
from kb_server.services.search import SearchService
from kb_server.storage.models import DocumentRow
from kb_server.storage.repositories.collections import CollectionRepository
from kb_server.storage.repositories.documents import DocumentRepository
from kb_server.storage.repositories.groups import GroupRepository
from kb_server.wiring import Services
from salemdex_retrieval import Engine, Source
from salemdex_retrieval.engine import AlwaysValid
from salemdex_retrieval.index.in_memory import InMemoryIndex
from salemdex_retrieval.rerankers.fake import FakeReranker
from tests.support import QUESTION, SEARCH_BINDING, sign_in

# people and places


@pytest.fixture
def login(
    auth: AuthService, make_user: Callable[..., User]
) -> Callable[[str, UserRole], Principal]:
    def make(email: str, role: UserRole) -> Principal:
        make_user(email, role=role)
        return sign_in(auth, email)[1]

    return make


@pytest.fixture
def hr(groups: GroupRepository) -> Group:
    return groups.add("hr")


@pytest.fixture
def it(groups: GroupRepository) -> Group:
    return groups.add("it")


@pytest.fixture
def hr_docs(collections: CollectionRepository) -> Collection:
    return collections.add(slug="hr-docs", title="HR documents", description="")


@pytest.fixture
def it_docs(collections: CollectionRepository) -> Collection:
    return collections.add(slug="it-docs", title="IT documents", description="")


@pytest.fixture
def admin(login: Callable[[str, UserRole], Principal]) -> Principal:
    return login("admin@example.com", "admin")


@pytest.fixture
def hr_manager(
    login: Callable[[str, UserRole], Principal],
    collections: CollectionRepository,
    hr_docs: Collection,
) -> Principal:
    """A manager of hr-docs, in no group."""
    principal = login("manager@example.com", "manager")
    collections.add_manager(collection_id=hr_docs.id, user_id=principal.user.id)
    return principal


@pytest.fixture
def hr_reader(
    login: Callable[[str, UserRole], Principal], groups: GroupRepository, hr: Group
) -> Principal:
    principal = login("an@example.com", "reader")
    groups.add_member(user_id=principal.user.id, group_id=hr.id)
    return principal


def front_end_acting_for(
    auth: AuthService, make_user: Callable[..., User], admin: Principal, email: str
) -> Principal:
    service_user = make_user("openwebui@example.com")
    key = auth.create_front_end_key(admin, service_user_id=service_user.id, name="owui")
    principal = auth.resolve(key.token, acting_for=email)
    assert principal is not None
    return principal


def found(search_service: SearchService, asker: Principal) -> set[uuid.UUID]:
    """The ids of the documents `asker` finds for QUESTION."""
    passages = search_service.search(asker, QUESTION)
    return {passage.document.id for passage in passages}


# who finds what


def test_a_reader_finds_only_documents_sharing_one_of_their_groups(
    search_service: SearchService,
    publish: Callable[..., Document],
    hr_reader: Principal,
    hr: Group,
    it: Group,
    hr_docs: Collection,
) -> None:
    shared = publish(hr_docs, group_ids=(hr.id, it.id))
    publish(hr_docs, group_ids=(it.id,))
    publish(hr_docs)  # private

    assert found(search_service, hr_reader) == {shared.id}


def test_a_manager_also_finds_every_document_of_the_collections_they_manage(
    search_service: SearchService,
    publish: Callable[..., Document],
    hr_manager: Principal,
    it: Group,
    hr_docs: Collection,
    it_docs: Collection,
) -> None:
    private = publish(hr_docs)
    for_it = publish(hr_docs, group_ids=(it.id,))
    publish(it_docs, group_ids=(it.id,))
    publish(it_docs)

    assert found(search_service, hr_manager) == {private.id, for_it.id}


def test_a_manager_of_no_collection_finds_what_their_groups_allow(
    search_service: SearchService,
    publish: Callable[..., Document],
    login: Callable[[str, UserRole], Principal],
    groups: GroupRepository,
    hr: Group,
    hr_docs: Collection,
) -> None:
    # an empty in_ matches nothing, so the manager part adds no document
    manager = login("manager@example.com", "manager")
    groups.add_member(user_id=manager.user.id, group_id=hr.id)
    shared = publish(hr_docs, group_ids=(hr.id,))
    publish(hr_docs)

    assert found(search_service, manager) == {shared.id}


def test_an_admin_finds_every_document(
    search_service: SearchService,
    publish: Callable[..., Document],
    admin: Principal,
    it: Group,
    hr_docs: Collection,
    it_docs: Collection,
) -> None:
    published = {
        publish(hr_docs).id,
        publish(it_docs).id,
        publish(it_docs, group_ids=(it.id,)).id,
    }

    assert found(search_service, admin) == published


def test_a_front_end_acting_for_an_admin_finds_only_what_the_admins_groups_allow(
    search_service: SearchService,
    publish: Callable[..., Document],
    auth: AuthService,
    make_user: Callable[..., User],
    groups: GroupRepository,
    admin: Principal,
    hr: Group,
    hr_docs: Collection,
) -> None:
    groups.add_member(user_id=admin.user.id, group_id=hr.id)
    shared = publish(hr_docs, group_ids=(hr.id,))
    publish(hr_docs)
    acting = front_end_acting_for(auth, make_user, admin, "admin@example.com")

    assert found(search_service, acting) == {shared.id}


def test_a_front_end_acting_for_a_manager_finds_what_the_manager_finds(
    search_service: SearchService,
    publish: Callable[..., Document],
    auth: AuthService,
    make_user: Callable[..., User],
    admin: Principal,
    hr_manager: Principal,
    hr_docs: Collection,
    it_docs: Collection,
) -> None:
    private = publish(hr_docs)
    publish(it_docs)
    acting = front_end_acting_for(auth, make_user, admin, "manager@example.com")

    assert found(search_service, acting) == {private.id}


# which versions are shown


def test_only_the_indexed_version_of_a_document_is_shown(
    search_service: SearchService,
    publish: Callable[..., Document],
    admin: Principal,
    hr_docs: Collection,
) -> None:
    # version 2 is in the index, but its index job hasn't moved indexed_version yet
    document = publish(hr_docs, versions=(1, 2))

    passages = search_service.search(admin, QUESTION)

    assert [
        (passage.document.id, passage.document.version) for passage in passages
    ] == [(document.id, 1)]


def test_a_document_being_deleted_is_never_shown(
    search_service: SearchService,
    publish: Callable[..., Document],
    documents: DocumentRepository,
    session: Session,
    admin: Principal,
    hr_docs: Collection,
) -> None:
    kept = publish(hr_docs)
    deleted = publish(hr_docs)
    documents.mark_deleting(deleted.id)
    session.flush()

    assert found(search_service, admin) == {kept.id}


def test_search_returns_at_most_the_limit(
    search_service: SearchService,
    publish: Callable[..., Document],
    admin: Principal,
    hr_docs: Collection,
) -> None:
    for _ in range(3):
        publish(hr_docs)

    assert len(search_service.search(admin, QUESTION, limit=2)) == 2


def test_without_a_limit_search_returns_up_to_the_maximum(
    search_service: SearchService,
    services: Services,
    publish: Callable[..., Document],
    admin: Principal,
    hr_docs: Collection,
) -> None:
    maximum = services.settings.search_max_limit
    for _ in range(maximum + 1):
        publish(hr_docs)

    assert len(search_service.search(admin, QUESTION)) == maximum


def test_a_limit_above_the_maximum_is_refused_naming_the_maximum(
    search_service: SearchService, services: Services, admin: Principal
) -> None:
    maximum = services.settings.search_max_limit

    with pytest.raises(InvalidRequest, match=f"at most {maximum}"):
        search_service.search(admin, QUESTION, limit=maximum + 1)


# what a passage shows


def test_a_passage_shows_its_text_headings_and_document(
    search_service: SearchService,
    publish: Callable[..., Document],
    admin: Principal,
    hr_docs: Collection,
) -> None:
    headings = ("Chương III. Thời giờ nghỉ ngơi", "Điều 12. Nghỉ phép năm")
    document = publish(hr_docs, title="Quy chế nhân sự 2026", headings=headings)

    [passage] = search_service.search(admin, QUESTION)

    assert QUESTION in passage.text
    assert passage.headings == headings
    assert passage.document.id == document.id
    assert passage.document.title == "Quy chế nhân sự 2026"
    assert passage.document.version == 1


def test_a_renamed_document_shows_its_new_title_without_reindexing(
    search_service: SearchService,
    publish: Callable[..., Document],
    session: Session,
    admin: Principal,
    hr_docs: Collection,
) -> None:
    document = publish(hr_docs, title="Quy chế cũ")
    session.execute(
        update(DocumentRow)
        .where(DocumentRow.id == document.id)
        .values(title="Quy chế mới")
    )
    session.flush()

    [passage] = search_service.search(admin, QUESTION)

    assert passage.document.title == "Quy chế mới"


def test_a_passage_whose_document_stops_being_served_after_the_last_gate_is_dropped(
    services: Services,
    search_index: InMemoryIndex,
    publish: Callable[..., Document],
    documents: DocumentRepository,
    session: Session,
    admin: Principal,
    hr_docs: Collection,
) -> None:
    # Validity that keeps everything stands in for a delete landing between the last
    # gate and the title read: the passage arrives, but its document has no title
    kept = publish(hr_docs)
    deleted = publish(hr_docs)
    documents.mark_deleting(deleted.id)
    session.flush()
    source = Source(name="dense", index=search_index, binding=SEARCH_BINDING, depth=10)
    search_service = SearchService(
        max_limit=services.settings.search_max_limit,
        session_factory=services.session_factory,
        retrieval_engine=Engine(validity=AlwaysValid()),
        build_statement=DenseThenRerank(source=source, reranker=FakeReranker()),
    )

    passages = search_service.search(admin, QUESTION)

    assert [passage.document.id for passage in passages] == [kept.id]
