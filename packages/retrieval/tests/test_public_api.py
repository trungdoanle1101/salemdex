"""The names the package exports at its top level, written out in full.

A name joins or leaves the top level only by changing this list, so the change shows
in review. Which names belong there: the package docstring.
"""

import salemdex_retrieval
import salemdex_retrieval.engine

TOP_LEVEL = [
    "Chunk",
    "Dense",
    "EmbeddedChunk",
    "Engine",
    "Everything",
    "LookupScan",
    "Many",
    "One",
    "Rrf",
    "Schema",
    "SearchableIndex",
    "Sparse",
    "VectorScan",
    "Window",
    "and_",
    "not_",
    "or_",
    "scan",
    "search",
]


def test_the_top_level_exports_exactly_the_agreed_names() -> None:
    assert sorted(salemdex_retrieval.__all__) == sorted(TOP_LEVEL)


def test_every_exported_name_can_be_imported() -> None:
    for name in TOP_LEVEL:
        assert hasattr(salemdex_retrieval, name), name


def test_execute_is_not_exported_so_every_search_runs_prepared() -> None:
    assert "execute" not in salemdex_retrieval.__all__
    assert "execute" not in salemdex_retrieval.engine.__all__
    assert not hasattr(salemdex_retrieval, "execute")
    assert not hasattr(salemdex_retrieval.engine, "execute")


def test_the_engine_tier_exports_the_prepared_statement_and_not_the_session() -> None:
    assert "PreparedStatement" in salemdex_retrieval.engine.__all__
    assert "Session" not in salemdex_retrieval.engine.__all__
    for name in salemdex_retrieval.engine.__all__:
        assert hasattr(salemdex_retrieval.engine, name), name
