"""Engines and sessions: the one way to run a search statement.

Each guarantee slot is bound where its value is fixed, so no call site passes it:

- validity, on the `Engine`: it doesn't depend on who asks, so it's set once for the
  process;
- access, on the `Session`: one asker's limits apply to everything they run, so every
  statement of the session gets the same filter.

A statement holds neither. It says what to search, the same whoever runs it, so it
can be logged, replayed and written by an agent without being able to leave out or
widen the access filter. Like SQLAlchemy's `Engine`, `Session` and `select()`.
"""

from __future__ import annotations

from dataclasses import dataclass

from salemdex_retrieval.engine.executor import execute
from salemdex_retrieval.engine.results import Passage
from salemdex_retrieval.engine.statements import Search
from salemdex_retrieval.engine.validity import Validity
from salemdex_retrieval.filters import Filter, check_is_filter


@dataclass(frozen=True, kw_only=True)
class Engine:
    """A search engine holding the validity check, for the whole life of the app.

    Frozen, so the check can't be swapped under open sessions. An app with nothing to
    check passes `AlwaysValid()`, which says so explicitly.

    Example:
        engine = Engine(validity=ServerValidity(session_factory))
        passages = engine.session(access=access).execute(statement)

    Attributes:
        validity: Which chunks may be served now, asked at every gate.
    """

    validity: Validity

    def session(self, *, access: Filter) -> Session:
        """Return a session running statements for one asker, limited by `access`.

        Args:
            access: What the asker may see; `Everything()` for an app without access
                control.

        Returns:
            A session of this engine.

        Raises:
            TypeError: `access` isn't a filter.
        """
        return Session(engine=self, access=access)


@dataclass(frozen=True, kw_only=True)
class Session:
    """A session running statements for one asker, every search limited by `access`.

    Build it with `Engine.session`. Frozen, so its access can't change between the
    statements it runs.

    Attributes:
        engine: The engine whose validity check it runs at every gate.
        access: What the asker may see, added to every source's search.
    """

    engine: Engine
    access: Filter

    def __post_init__(self) -> None:
        """Check the access filter.

        Raises:
            TypeError: `access` isn't a filter.
        """
        check_is_filter(self.access, owner="Session")

    def execute(self, statement: Search) -> list[Passage]:
        """Run `statement` with this session's access and its engine's validity.

        Args:
            statement: The search to run.

        Returns:
            At most the statement's `Take` limit of passages, best first, each with
            its trail.

        Raises:
            StatementError: The stages come in an order the steps don't allow.
            SchemaMismatchError: The access filter uses a field that isn't in a
                source's schema.
            ValueError: An embedder or the reranker returned the wrong number of rows
                or scores, or a number that doesn't fit.
            TypeError: A score or a vector number isn't a real number.
        """
        return execute(statement, access=self.access, validity=self.engine.validity)
