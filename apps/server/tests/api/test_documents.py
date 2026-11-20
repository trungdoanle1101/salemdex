"""POST /api/v1/documents (multipart upload), GET and DELETE /api/v1/documents/{id}."""

import io
import uuid
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from kb_server.domain.collections import Collection
from kb_server.domain.groups import Group
from kb_server.domain.users import User, UserRole
from kb_server.services.auth import AuthService
from kb_server.storage.repositories.collections import CollectionRepository
from kb_server.storage.repositories.documents import DocumentRepository
from kb_server.storage.repositories.groups import GroupRepository
from kb_server.storage.repositories.users import UserRepository
from tests.support import MAX_UPLOAD_MB, bearer, sign_in

DOCUMENTS = "/api/v1/documents"
CONTENT = "Điều 1. Phạm vi điều chỉnh\n".encode()
DOCUMENT_FIELDS = {
    "id",
    "collection_id",
    "title",
    "status",
    "current_version",
    "indexed_version",
    "created_at",
    "updated_at",
    "groups",
    "file",
    "job",
}


@pytest.fixture
def headers_for(
    auth: AuthService, make_user: Callable[..., User]
) -> Callable[[str, UserRole], dict[str, str]]:
    def make(email: str, role: UserRole) -> dict[str, str]:
        make_user(email, role=role)
        return bearer(sign_in(auth, email)[0])

    return make


@pytest.fixture
def hr(groups: GroupRepository) -> Group:
    return groups.add("hr")


@pytest.fixture
def hr_docs(collections: CollectionRepository, hr: Group) -> Collection:
    collection = collections.add(slug="hr-docs", title="HR documents", description="")
    collections.set_groups(collection.id, [hr.id])
    return collection


@pytest.fixture
def admin_headers(
    headers_for: Callable[[str, UserRole], dict[str, str]],
) -> dict[str, str]:
    return headers_for("admin@example.com", "admin")


def post(
    client: TestClient,
    headers: dict[str, str],
    fields: dict[str, Any],
    *,
    content: bytes = CONTENT,
    filename: str = "quy-che.txt",
) -> Any:
    return client.post(
        DOCUMENTS,
        headers=headers,
        data=fields,
        files={"file": (filename, io.BytesIO(content), "application/octet-stream")},
    )


def upload(
    client: TestClient, headers: dict[str, str], collection: Collection, **fields: Any
) -> dict[str, Any]:
    content = fields.pop("content", CONTENT)
    response = post(
        client,
        headers,
        {"collection_id": str(collection.id), **fields},
        content=content,
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def error_code(response: Any) -> str:
    code: str = response.json()["error"]["code"]
    return code


# uploading


def test_an_upload_returns_the_new_document(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection
) -> None:
    response = post(client, admin_headers, {"collection_id": str(hr_docs.id)})

    assert response.status_code == 201
    body = response.json()
    assert set(body) == DOCUMENT_FIELDS
    assert body["collection_id"] == str(hr_docs.id)
    assert body["title"] == "quy-che"
    assert body["status"] == "active"
    assert body["current_version"] == 1
    assert [group["name"] for group in body["groups"]] == ["hr"]
    assert body["file"]["filename"] == "quy-che.txt"
    assert body["file"]["size"] == len(CONTENT)
    assert body["file"]["media_type"] == "text/plain"
    assert set(body["job"]) == {"job_id", "status", "error"}
    assert body["job"]["status"] == "queued"
    assert body["job"]["error"] is None


def test_a_title_can_be_given(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection
) -> None:
    body = upload(client, admin_headers, hr_docs, title="  Quy chế 2026 ")

    assert body["title"] == "Quy chế 2026"


def test_group_ids_are_sent_as_repeated_fields(
    client: TestClient,
    groups: GroupRepository,
    admin_headers: dict[str, str],
    hr: Group,
    hr_docs: Collection,
) -> None:
    it = groups.add("it")

    body = upload(client, admin_headers, hr_docs, group_ids=[str(hr.id), str(it.id)])

    assert sorted(group["name"] for group in body["groups"]) == ["hr", "it"]


@pytest.mark.parametrize("value", ["true", "1", "on"])
def test_private_is_a_form_flag(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection, value: str
) -> None:
    assert upload(client, admin_headers, hr_docs, private=value)["groups"] == []


# refused uploads


def test_upload_needs_a_token(client: TestClient, hr_docs: Collection) -> None:
    assert post(client, {}, {"collection_id": str(hr_docs.id)}).status_code == 401


@pytest.mark.parametrize(
    ("extra", "field"),
    [
        ({"title": "T" * 501}, "title"),
        ({"group_ids": ["not-a-uuid"]}, "group_ids.0"),
    ],
    ids=["title too long", "malformed group id"],
)
def test_a_malformed_field_is_422(
    client: TestClient,
    admin_headers: dict[str, str],
    hr_docs: Collection,
    extra: dict[str, Any],
    field: str,
) -> None:
    response = post(client, admin_headers, {"collection_id": str(hr_docs.id), **extra})

    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["field"] == field


@pytest.mark.parametrize(
    "form", [{}, {"collection_id": "not-a-uuid"}], ids=["missing", "malformed"]
)
def test_the_collection_id_is_required_and_checked(
    client: TestClient, admin_headers: dict[str, str], form: dict[str, Any]
) -> None:
    response = post(client, admin_headers, form)

    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["field"] == "collection_id"


def test_no_file_is_422(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection
) -> None:
    response = client.post(
        DOCUMENTS, headers=admin_headers, data={"collection_id": str(hr_docs.id)}
    )

    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["field"] == "file"


def test_private_and_group_ids_together_are_422(
    client: TestClient, admin_headers: dict[str, str], hr: Group, hr_docs: Collection
) -> None:
    response = post(
        client,
        admin_headers,
        {
            "collection_id": str(hr_docs.id),
            "group_ids": [str(hr.id)],
            "private": "true",
        },
    )

    assert response.status_code == 422
    assert error_code(response) == "invalid_request"


def test_unknown_group_ids_are_422(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection
) -> None:
    response = post(
        client,
        admin_headers,
        {"collection_id": str(hr_docs.id), "group_ids": [str(uuid.uuid4())]},
    )

    assert response.status_code == 422


def test_the_same_file_again_is_409_naming_the_document(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection
) -> None:
    first = upload(client, admin_headers, hr_docs)

    response = post(
        client, admin_headers, {"collection_id": str(hr_docs.id)}, filename="other.txt"
    )

    assert response.status_code == 409
    assert error_code(response) == "duplicate_file"
    assert response.json()["error"]["details"] == {"document_id": first["id"]}


def test_a_file_over_the_limit_is_413(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection
) -> None:
    too_big = b"x" * (MAX_UPLOAD_MB * 1024 * 1024 + 1)

    response = post(
        client, admin_headers, {"collection_id": str(hr_docs.id)}, content=too_big
    )

    assert response.status_code == 413
    assert error_code(response) == "file_too_large"


def test_an_unsupported_file_is_415(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection
) -> None:
    response = post(
        client,
        admin_headers,
        {"collection_id": str(hr_docs.id)},
        content=b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n",
        filename="quy-che.pdf",
    )

    assert response.status_code == 415
    assert error_code(response) == "unsupported_file_type"
    assert ".docx" in response.json()["error"]["message"]


def test_an_empty_file_is_422(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection
) -> None:
    response = post(
        client, admin_headers, {"collection_id": str(hr_docs.id)}, content=b""
    )

    assert response.status_code == 422


def test_who_may_upload(
    client: TestClient,
    groups: GroupRepository,
    users: UserRepository,
    headers_for: Callable[[str, UserRole], dict[str, str]],
    hr: Group,
    hr_docs: Collection,
) -> None:
    outsider = headers_for("binh@example.com", "reader")
    hr_reader = headers_for("an@example.com", "reader")
    an = users.get_by_email("an@example.com")
    assert an is not None
    groups.add_member(user_id=an.id, group_id=hr.id)
    other_manager = headers_for("other-manager@example.com", "manager")
    fields = {"collection_id": str(hr_docs.id)}

    hidden = post(client, outsider, fields)
    assert (hidden.status_code, error_code(hidden)) == (404, "collection_not_found")
    assert post(client, hr_reader, fields).status_code == 403
    assert post(client, other_manager, fields).status_code == 403


# reading


def test_get_returns_what_the_upload_returned(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection
) -> None:
    uploaded = upload(client, admin_headers, hr_docs)

    response = client.get(f"{DOCUMENTS}/{uploaded['id']}", headers=admin_headers)

    assert response.status_code == 200
    assert response.json() == uploaded


def test_a_document_you_cannot_read_is_404(
    client: TestClient,
    headers_for: Callable[[str, UserRole], dict[str, str]],
    admin_headers: dict[str, str],
    hr_docs: Collection,
) -> None:
    uploaded = upload(client, admin_headers, hr_docs)
    outsider = headers_for("binh@example.com", "reader")

    response = client.get(f"{DOCUMENTS}/{uploaded['id']}", headers=outsider)

    assert response.status_code == 404
    assert error_code(response) == "document_not_found"


def test_an_unknown_or_malformed_document_id(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    assert (
        client.get(f"{DOCUMENTS}/{uuid.uuid4()}", headers=admin_headers).status_code
        == 404
    )
    assert (
        client.get(f"{DOCUMENTS}/not-a-uuid", headers=admin_headers).status_code == 422
    )


def test_reading_needs_a_token(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection
) -> None:
    uploaded = upload(client, admin_headers, hr_docs)

    assert client.get(f"{DOCUMENTS}/{uploaded['id']}").status_code == 401


# deleting


def test_a_delete_is_accepted_with_no_body_and_the_document_is_gone_at_once(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection
) -> None:
    uploaded = upload(client, admin_headers, hr_docs)
    url = f"{DOCUMENTS}/{uploaded['id']}"

    response = client.delete(url, headers=admin_headers)

    assert response.status_code == 202
    assert response.content == b""
    assert client.get(url, headers=admin_headers).status_code == 404


def test_who_may_delete(
    client: TestClient,
    groups: GroupRepository,
    users: UserRepository,
    headers_for: Callable[[str, UserRole], dict[str, str]],
    admin_headers: dict[str, str],
    hr: Group,
    hr_docs: Collection,
) -> None:
    url = f"{DOCUMENTS}/{upload(client, admin_headers, hr_docs)['id']}"
    outsider = headers_for("binh@example.com", "reader")
    hr_reader = headers_for("an@example.com", "reader")
    an = users.get_by_email("an@example.com")
    assert an is not None
    groups.add_member(user_id=an.id, group_id=hr.id)

    hidden = client.delete(url, headers=outsider)
    assert (hidden.status_code, error_code(hidden)) == (404, "document_not_found")
    assert client.delete(url, headers=hr_reader).status_code == 403
    assert client.get(url, headers=hr_reader).status_code == 200


def test_deleting_an_unknown_or_malformed_document_id(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    assert (
        client.delete(f"{DOCUMENTS}/{uuid.uuid4()}", headers=admin_headers).status_code
        == 404
    )
    assert (
        client.delete(f"{DOCUMENTS}/not-a-uuid", headers=admin_headers).status_code
        == 422
    )


def test_deleting_needs_a_token(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection
) -> None:
    uploaded = upload(client, admin_headers, hr_docs)

    assert client.delete(f"{DOCUMENTS}/{uploaded['id']}").status_code == 401


# listing

LIST_ITEM_FIELDS = {
    "id",
    "title",
    "collection_id",
    "current_version",
    "indexed_version",
}


def list_documents(client: TestClient, headers: dict[str, str], **params: Any) -> Any:
    return client.get(DOCUMENTS, headers=headers, params=params)


def test_the_list_shows_each_documents_id_title_and_versions(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection
) -> None:
    uploaded = upload(client, admin_headers, hr_docs, title="Quy chế")

    response = list_documents(client, admin_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["next_cursor"] is None
    [item] = body["items"]
    assert set(item) == LIST_ITEM_FIELDS
    assert item == {
        "id": uploaded["id"],
        "title": "Quy chế",
        "collection_id": str(hr_docs.id),
        "current_version": 1,
        "indexed_version": None,
    }


def test_the_document_and_the_list_show_the_version_search_serves(
    client: TestClient,
    admin_headers: dict[str, str],
    hr_docs: Collection,
    documents: DocumentRepository,
) -> None:
    uploaded = upload(client, admin_headers, hr_docs)
    # what the worker records once the index job is done
    documents.set_indexed_version(uuid.UUID(uploaded["id"]), 1)

    document = client.get(f"{DOCUMENTS}/{uploaded['id']}", headers=admin_headers)
    [item] = list_documents(client, admin_headers).json()["items"]

    assert document.json()["indexed_version"] == 1
    assert item["indexed_version"] == 1


def test_following_next_cursor_reaches_every_document_once(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection
) -> None:
    uploaded = {
        upload(client, admin_headers, hr_docs, content=f"Điều {n}".encode())["id"]
        for n in range(3)
    }

    first = list_documents(client, admin_headers, limit=2).json()
    second = list_documents(
        client, admin_headers, limit=2, cursor=first["next_cursor"]
    ).json()

    assert len(first["items"]) == 2 and first["next_cursor"] is not None
    assert len(second["items"]) == 1 and second["next_cursor"] is None
    listed = [item["id"] for item in first["items"] + second["items"]]
    assert sorted(listed) == sorted(uploaded)


def test_a_reader_lists_only_what_they_may_read(
    client: TestClient,
    admin_headers: dict[str, str],
    headers_for: Callable[[str, UserRole], dict[str, str]],
    users: UserRepository,
    groups: GroupRepository,
    hr: Group,
    hr_docs: Collection,
) -> None:
    shared = upload(client, admin_headers, hr_docs, content=b"shared")
    upload(client, admin_headers, hr_docs, content=b"private", private="true")
    reader_headers = headers_for("an@example.com", "reader")
    reader = users.get_by_email("an@example.com")
    assert reader is not None
    groups.add_member(user_id=reader.id, group_id=hr.id)

    items = list_documents(client, reader_headers).json()["items"]

    assert [item["id"] for item in items] == [shared["id"]]


def test_a_hidden_collection_is_404(
    client: TestClient,
    headers_for: Callable[[str, UserRole], dict[str, str]],
    hr_docs: Collection,
) -> None:
    outsider_headers = headers_for("binh@example.com", "reader")

    response = list_documents(client, outsider_headers, collection_id=str(hr_docs.id))

    assert response.status_code == 404
    assert error_code(response) == "collection_not_found"


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({"limit": 0}, id="limit 0"),
        pytest.param({"limit": -1}, id="negative limit"),
        pytest.param({"limit": 101}, id="limit above 100"),
        pytest.param({"limit": "many"}, id="limit not a number"),
        pytest.param({"collection_id": "hr-docs"}, id="collection not a uuid"),
    ],
)
def test_bad_list_parameters_are_422(
    client: TestClient, admin_headers: dict[str, str], params: dict[str, Any]
) -> None:
    response = list_documents(client, admin_headers, **params)

    assert response.status_code == 422
    assert error_code(response) == "invalid_request"


def test_a_cursor_not_made_by_the_server_is_422(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    response = list_documents(client, admin_headers, cursor="not a cursor")

    assert response.status_code == 422
    assert response.json()["error"]["message"] == "cursor is not valid"


def test_an_overlong_cursor_is_refused_before_it_is_decoded(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    response = list_documents(client, admin_headers, cursor="a" * 201)

    assert response.status_code == 422
    assert [d["field"] for d in response.json()["error"]["details"]] == ["cursor"]


def test_listing_needs_a_token(client: TestClient) -> None:
    response = client.get(DOCUMENTS)

    assert response.status_code == 401
    assert error_code(response) == "not_authenticated"
