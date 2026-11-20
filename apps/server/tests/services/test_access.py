"""Deciding what an asker may read."""

from collections.abc import Callable

import pytest

from kb_server.domain.collections import Collection
from kb_server.domain.credentials import Principal
from kb_server.domain.groups import Group
from kb_server.domain.users import User, UserRole
from kb_server.services.access import read_access
from kb_server.services.auth import AuthService
from kb_server.storage.repositories.collections import CollectionRepository
from kb_server.storage.repositories.groups import GroupRepository
from kb_server.storage.repositories.users import UserRepository
from tests.support import sign_in


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
def hr_docs(collections: CollectionRepository) -> Collection:
    return collections.add(slug="hr-docs", title="HR", description="")


def test_an_admin_reads_everything(
    users: UserRepository, login: Callable[[str, UserRole], Principal]
) -> None:
    assert read_access(login("admin@example.com", "admin"), users=users) is None


def test_a_reader_reads_by_their_groups(
    users: UserRepository,
    groups: GroupRepository,
    login: Callable[[str, UserRole], Principal],
    hr: Group,
) -> None:
    reader = login("an@example.com", "reader")
    groups.add_member(user_id=reader.user.id, group_id=hr.id)

    access = read_access(reader, users=users)

    assert access is not None
    assert hr.id in access.group_ids
    assert access.managed_collection_ids == frozenset()


def test_a_manager_also_reads_the_collections_they_manage(
    users: UserRepository,
    collections: CollectionRepository,
    login: Callable[[str, UserRole], Principal],
    hr_docs: Collection,
) -> None:
    manager = login("manager@example.com", "manager")
    collections.add_manager(collection_id=hr_docs.id, user_id=manager.user.id)

    access = read_access(manager, users=users)

    assert access is not None
    assert access.managed_collection_ids == frozenset({hr_docs.id})


def test_a_reader_with_a_manager_row_left_behind_manages_nothing(
    users: UserRepository,
    collections: CollectionRepository,
    login: Callable[[str, UserRole], Principal],
    hr_docs: Collection,
) -> None:
    demoted = login("demoted@example.com", "reader")
    collections.add_manager(collection_id=hr_docs.id, user_id=demoted.user.id)

    access = read_access(demoted, users=users)

    assert access is not None
    assert access.managed_collection_ids == frozenset()


def test_a_front_end_acting_for_an_admin_reads_only_by_the_admins_groups(
    users: UserRepository,
    auth: AuthService,
    make_user: Callable[..., User],
    groups: GroupRepository,
    login: Callable[[str, UserRole], Principal],
    hr: Group,
) -> None:
    admin = login("admin@example.com", "admin")
    groups.add_member(user_id=admin.user.id, group_id=hr.id)
    service_user = make_user("openwebui@example.com")
    key = auth.create_front_end_key(admin, service_user_id=service_user.id, name="owui")
    acting = auth.resolve(key.token, acting_for="admin@example.com")
    assert acting is not None

    access = read_access(acting, users=users)

    assert access is not None
    assert hr.id in access.group_ids
    assert access.managed_collection_ids == frozenset()
