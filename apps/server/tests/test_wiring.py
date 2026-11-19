"""The wiring: what's built for the worker, and what the other processes don't load."""

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from kb_server.domain.chunks import KbChunks
from kb_server.settings import Settings
from kb_server.wiring import WorkerSetupError, _build_embedder, _build_reranker


def test_the_fields_every_search_and_delete_filter_by_are_indexed() -> None:
    # Qdrant builds a payload index only for these. Without one, a reader's narrow access
    # filter makes a large collection's search approximate, and deleting a document
    # scans every point.
    indexed = {name for name, field in KbChunks.fields().items() if field.indexed}

    assert indexed == {"access_group_ids", "collection_id", "document_id"}


WORKER_ONLY_MODULES = (
    "torch",
    "transformers",
    "sentence_transformers",
    "qdrant_client",
)


def test_the_api_and_the_commands_load_none_of_the_workers_dependencies(
    tmp_path: Path,
) -> None:
    # W5. A fresh interpreter, because this test process has loaded them already; run
    # from an empty folder so no .env is read.
    script = (
        "import sys\n"
        "import kb_server.app, kb_server.manage\n"
        "kb_server.app.create_app()\n"
        f"print(','.join(m for m in {WORKER_ONLY_MODULES!r} if m in sys.modules))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == ""


def model_settings(*, models_folder: Path | None, **values: object) -> Settings:
    return Settings(
        _env_file=None,  # ignore .env
        database_url="postgresql+psycopg://kb@localhost/kb",
        storage_path=Path("/srv/kb"),
        models_folder=models_folder,
        **values,  # type: ignore[arg-type]
    )


BUILDERS = pytest.mark.parametrize(
    ("build", "spec_setting"),
    [(_build_embedder, "embedder_spec"), (_build_reranker, "reranker_spec")],
    ids=["embedder", "reranker"],
)


@BUILDERS
def test_a_model_without_a_models_folder_is_refused(
    build: Callable[[Settings], object], spec_setting: str
) -> None:
    with pytest.raises(WorkerSetupError, match="KB_MODELS_FOLDER is not set"):
        build(model_settings(models_folder=None))


@BUILDERS
def test_an_unknown_model_spec_is_refused_listing_the_known_ones(
    build: Callable[[Settings], object], spec_setting: str, tmp_path: Path
) -> None:
    settings = model_settings(models_folder=tmp_path, **{spec_setting: "no-such-model"})

    with pytest.raises(WorkerSetupError, match="'no-such-model'; known: "):
        build(settings)


@BUILDERS
def test_a_model_missing_from_the_models_folder_is_refused_naming_its_folder(
    build: Callable[[Settings], object], spec_setting: str, tmp_path: Path
) -> None:
    settings = model_settings(models_folder=tmp_path)
    spec_name = getattr(settings, spec_setting)

    with pytest.raises(WorkerSetupError) as refused:
        build(settings)

    message = str(refused.value)
    assert f"{spec_name!r} isn't in KB_MODELS_FOLDER" in message
    assert str(tmp_path) in message
