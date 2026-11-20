"""The cursors that page through GET /documents."""

import base64
import re
import uuid
from datetime import UTC, datetime, timedelta, timezone

import pytest

from kb_server.api.documents import decode_cursor, encode_cursor
from kb_server.domain.documents import DocumentPosition
from kb_server.domain.errors import InvalidRequest

POSITION = DocumentPosition(
    created_at=datetime(2026, 11, 20, 9, 14, 3, 512874, tzinfo=UTC),
    id=uuid.UUID("3f2a9c1e-8b4d-4e7a-9c2f-1d5e6b7a8c90"),
)


def cursor_of(text: str | bytes) -> str:
    raw = text.encode() if isinstance(text, str) else text
    return base64.urlsafe_b64encode(raw).decode()


def test_a_cursor_gives_back_the_exact_position_microseconds_included() -> None:
    decoded = decode_cursor(encode_cursor(POSITION))

    assert decoded == POSITION
    assert decoded is not None and decoded.created_at.microsecond == 512874


def test_a_time_in_another_zone_gives_back_the_same_instant() -> None:
    hanoi = timezone(timedelta(hours=7))
    position = DocumentPosition(
        created_at=POSITION.created_at.astimezone(hanoi), id=POSITION.id
    )

    assert decode_cursor(encode_cursor(position)) == POSITION


def test_a_cursor_needs_no_escaping_in_a_query_string() -> None:
    cursor = encode_cursor(POSITION)

    assert cursor is not None
    assert re.fullmatch(r"[A-Za-z0-9_=-]+", cursor)


def test_no_position_means_no_cursor_and_back() -> None:
    assert encode_cursor(None) is None
    assert decode_cursor(None) is None


@pytest.mark.parametrize(
    "cursor",
    [
        pytest.param("", id="empty"),
        pytest.param("not a cursor!", id="not base64"),
        pytest.param(cursor_of(b"\xff\xfe"), id="not UTF-8"),
        pytest.param(cursor_of("2026-11-20T09:14:03+00:00"), id="no separator"),
        pytest.param(cursor_of(f"a|b|{POSITION.id}"), id="too many parts"),
        pytest.param(cursor_of(f"yesterday|{POSITION.id}"), id="not a time"),
        pytest.param(cursor_of("2026-11-20T09:14:03+00:00|42"), id="not a uuid"),
        pytest.param(cursor_of(f"2026-11-20T09:14:03|{POSITION.id}"), id="no offset"),
    ],
)
def test_a_cursor_not_made_by_the_server_is_refused(cursor: str) -> None:
    with pytest.raises(InvalidRequest, match="cursor is not valid"):
        decode_cursor(cursor)
