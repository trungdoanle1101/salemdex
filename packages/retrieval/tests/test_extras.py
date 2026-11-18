"""The core imports none of the libraries its extras install."""

import subprocess
import sys

# The modules that need an extra, and the libraries the extras install.
EXTRA_MODULES = (
    "salemdex_retrieval.index.qdrant",
    "salemdex_retrieval.embedders.huggingface",
)
EXTRA_LIBRARIES = ("qdrant_client", "torch", "sentence_transformers", "transformers")


def test_importing_every_core_module_loads_no_extra_library() -> None:
    # A fresh interpreter, because this test process may have loaded them already.
    # Every module is imported, so a new one that needs an extra fails here until it is
    # listed in EXTRA_MODULES. pkgutil.walk_packages would import the extra packages
    # to look inside them, so the walk skips them itself.
    script = (
        "import importlib, pkgutil, sys\n"
        f"extra = {EXTRA_MODULES!r}\n"
        "def import_all(package):\n"
        "    for info in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):\n"
        "        if info.name.startswith(extra):\n"
        "            continue\n"
        "        module = importlib.import_module(info.name)\n"
        "        if info.ispkg:\n"
        "            import_all(module)\n"
        "import_all(importlib.import_module('salemdex_retrieval'))\n"
        f"print(','.join(m for m in {EXTRA_LIBRARIES!r} if m in sys.modules))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == ""
