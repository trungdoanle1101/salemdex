import uuid

import pytest

from kb_server.domain.errors import EmailTaken
from kb_server.storage.repositories.collections import CollectionRepository
from kb_server.storage.repositories.groups import GroupRepository
from kb_server.storage.repositories.users import UserRepository


def add(users: UserRepository, email: str = "an@example.com") -> None:
    users.add(email=email, display_name="An", role="reader", password_hash="hash")


def test_a_new_user_is_saved_with_defaults(users: UserRepository) -> None:
    users.add(
        email=" An@Example.com ",
        display_name="An",
        role="manager",
        password_hash="hash",
    )

    user = users.get_by_email("an@example.com")
    assert user is not None
    assert user.email == "an@example.com"
    assert user.role == "manager"
    assert user.is_active
    assert user.created_at.tzinfo is not None


def test_get_finds_a_user_by_id(users: UserRepository) -> None:
    add(users)
    user = users.get_by_email("an@example.com")
    assert user is not None

    assert users.get(user.id) == user
    assert users.get(uuid.uuid4()) is None


def test_get_by_email_ignores_case(users: UserRepository) -> None:
    add(users)

    assert users.get_by_email("AN@example.com") is not None
    assert users.get_by_email("binh@example.com") is None


def test_an_email_is_taken_whatever_its_case(users: UserRepository) -> None:
    add(users)

    with pytest.raises(EmailTaken):
        add(users, "AN@Example.com")

    # the savepoint undid only the failed insert
    assert users.get_by_email("an@example.com") is not None


def test_get_credentials_by_email_returns_the_password_hash(
    users: UserRepository,
) -> None:
    add(users)

    found = users.get_credentials_by_email("an@example.com")

    assert found is not None
    user, password_hash = found
    assert user.email == "an@example.com"
    assert password_hash == "hash"
    assert users.get_credentials_by_email("binh@example.com") is None


def test_list_group_ids_gives_only_the_users_own_groups(
    users: UserRepository, groups: GroupRepository
) -> None:
    an = users.add(
        email="an@example.com", display_name="An", role="reader", password_hash="h"
    )
    binh = users.add(
        email="binh@example.com", display_name="Binh", role="reader", password_hash="h"
    )
    hr, it = groups.add("hr"), groups.add("it")
    groups.add_member(user_id=an.id, group_id=hr.id)
    groups.add_member(user_id=an.id, group_id=it.id)
    groups.add_member(user_id=binh.id, group_id=it.id)

    assert users.list_group_ids(an.id) == {hr.id, it.id}
    assert users.list_group_ids(uuid.uuid4()) == set()


def test_list_managed_collection_ids_gives_only_the_users_own_collections(
    users: UserRepository, collections: CollectionRepository
) -> None:
    an = users.add(
        email="an@example.com", display_name="An", role="manager", password_hash="h"
    )
    binh = users.add(
        email="binh@example.com", display_name="Binh", role="manager", password_hash="h"
    )
    hr_docs = collections.add(slug="hr-docs", title="HR", description="")
    it_docs = collections.add(slug="it-docs", title="IT", description="")
    collections.add_manager(collection_id=hr_docs.id, user_id=an.id)
    collections.add_manager(collection_id=it_docs.id, user_id=binh.id)

    assert users.list_managed_collection_ids(an.id) == {hr_docs.id}
    assert users.list_managed_collection_ids(uuid.uuid4()) == set()
