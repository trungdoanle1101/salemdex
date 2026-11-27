import logging
from collections.abc import Generator
from contextlib import contextmanager

import pytest

from kb_server import app as app_module
from kb_server.domain.chunks import KbChunks
from kb_server.search.validity import ServedVersionsOnly
from kb_server.wiring import Services
from salemdex_retrieval.engine import PreparedStatement
from salemdex_retrieval.index.in_memory import InMemoryIndex
from tests.support import prepare_search


def test_starting_the_api_logs_the_statement_search_runs(
    services: Services,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    prepared = prepare_search(
        InMemoryIndex(schema=KbChunks),
        validity=ServedVersionsOnly(services.session_factory),
    )

    @contextmanager
    def fake_open_search_tools(
        _services: Services,
    ) -> Generator[PreparedStatement, None, None]:
        yield prepared

    monkeypatch.setattr(app_module, "open_search_tools", fake_open_search_tools)

    with (
        caplog.at_level(logging.INFO, logger="kb_server.app"),
        app_module._open_search_tools_or_log(services),
    ):
        pass

    logged = (
        "search runs search('chunks').retrieve(scan('dense', depth=10))"
        ".rerank('cross_encoder').take()"
    )
    assert [record.getMessage() for record in caplog.records] == [logged]
