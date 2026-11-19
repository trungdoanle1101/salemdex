"""The wiring: what's built for the worker, and what the other processes don't load."""

import subprocess
import sys
from pathlib import Path

from kb_server.domain.chunks import KbChunks


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
