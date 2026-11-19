from pathlib import Path

import pytest
from pydantic import ValidationError

from kb_server.settings import Settings


def settings(**values: object) -> Settings:
    return Settings(
        _env_file=None,  # ignore .env
        database_url="postgresql+psycopg://kb@localhost/kb",
        storage_path=Path("/srv/kb"),
        **values,  # type: ignore[arg-type]
    )


def test_the_default_search_limit_fits_the_default_rerank_depth() -> None:
    defaults = settings()

    assert defaults.search_max_limit <= defaults.rerank_depth


def test_a_search_limit_equal_to_the_rerank_depth_is_accepted() -> None:
    equal = settings(search_max_limit=10, rerank_depth=10)

    assert equal.search_max_limit == equal.rerank_depth


def test_a_search_limit_above_the_rerank_depth_is_refused() -> None:
    with pytest.raises(ValidationError, match="KB_SEARCH_MAX_LIMIT.*KB_RERANK_DEPTH"):
        settings(search_max_limit=20, rerank_depth=10)


def test_a_refused_setting_never_prints_the_inputs_which_hold_the_password() -> None:
    with pytest.raises(ValidationError) as refused:
        settings(search_max_limit=20, rerank_depth=10)

    message = str(refused.value)
    assert "input_value" not in message
    assert "postgresql" not in message
