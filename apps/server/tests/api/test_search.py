"""POST /search: what the API takes and returns."""

from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from kb_server.app import create_app
from kb_server.domain.collections import Collection
from kb_server.domain.documents import Document
from kb_server.domain.groups import Group
from kb_server.domain.users import User
from kb_server.services.auth import AuthService
from kb_server.services.search import SearchService
from kb_server.storage.repositories.collections import CollectionRepository
from kb_server.storage.repositories.groups import GroupRepository
from kb_server.wiring import Services
from tests.support import QUESTION, bearer, sign_in


@pytest.fixture
def search_client(
    services: Services, search_service: SearchService
) -> Iterator[TestClient]:
    with TestClient(create_app(services, search_service=search_service)) as client:
        yield client


@pytest.fixture
def hr(groups: GroupRepository) -> Group:
    return groups.add("hr")


@pytest.fixture
def hr_docs(collections: CollectionRepository) -> Collection:
    return collections.add(slug="hr-docs", title="HR documents", description="")


@pytest.fixture
def reader_token(
    auth: AuthService,
    make_user: Callable[..., User],
    groups: GroupRepository,
    hr: Group,
) -> str:
    """A reader in the hr group."""
    reader = make_user("an@example.com")
    groups.add_member(user_id=reader.id, group_id=hr.id)
    return sign_in(auth, "an@example.com")[0]


def test_search_returns_the_passages_the_asker_may_read(
    search_client: TestClient,
    publish: Callable[..., Document],
    reader_token: str,
    hr: Group,
    hr_docs: Collection,
) -> None:
    shared = publish(hr_docs, group_ids=(hr.id,), headings=("Điều 12. Nghỉ phép năm",))
    publish(hr_docs)  # private

    response = search_client.post(
        "/api/v1/search", json={"question": QUESTION}, headers=bearer(reader_token)
    )

    assert response.status_code == 200
    [passage] = response.json()["passages"]
    assert QUESTION in passage["text"]
    assert passage["headings"] == ["Điều 12. Nghỉ phép năm"]
    assert passage["document"] == {
        "id": str(shared.id),
        "title": "Quy chế nhân sự",
        "version": 1,
    }


def test_nothing_found_is_an_empty_list_not_an_error(
    search_client: TestClient, reader_token: str
) -> None:
    response = search_client.post(
        "/api/v1/search", json={"question": QUESTION}, headers=bearer(reader_token)
    )

    assert response.status_code == 200
    assert response.json() == {"passages": []}


def test_a_limit_above_the_maximum_is_refused(
    search_client: TestClient, services: Services, reader_token: str
) -> None:
    maximum = services.settings.search_max_limit

    response = search_client.post(
        "/api/v1/search",
        json={"question": QUESTION, "limit": maximum + 1},
        headers=bearer(reader_token),
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert f"at most {maximum}" in response.json()["error"]["message"]


@pytest.mark.parametrize(
    "body",
    [
        {"question": "   "},
        {"question": "x" * 501},
        {"question": QUESTION, "limit": 0},
        {"question": QUESTION, "collection": "hr-docs"},
    ],
    ids=["blank question", "long question", "zero limit", "unknown field"],
)
def test_a_malformed_search_is_refused(
    search_client: TestClient, reader_token: str, body: dict[str, object]
) -> None:
    response = search_client.post(
        "/api/v1/search", json=body, headers=bearer(reader_token)
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_search_needs_a_token(search_client: TestClient) -> None:
    response = search_client.post("/api/v1/search", json={"question": QUESTION})

    assert response.status_code == 401
