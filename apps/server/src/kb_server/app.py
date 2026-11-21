import logging
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager

from fastapi import FastAPI

from kb_server.api import (
    auth,
    collections,
    documents,
    groups,
    health,
    keys,
    search,
    users,
)
from kb_server.api.errors import register_error_handlers
from kb_server.services.search import SearchService
from kb_server.settings import Settings
from kb_server.wiring import (
    Services,
    ToolSetupError,
    build_search_service,
    build_services,
    close_services,
    open_search_tools,
)
from salemdex_retrieval.engine import PreparedStatement

API_V1 = "/api/v1"

logger = logging.getLogger(__name__)


def create_app(
    services: Services | None = None, *, search_service: SearchService | None = None
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        # Given services (tests): no models are loaded; search works only when a
        # search service is given too
        if services is not None:
            app.state.services = services
            app.state.search_service = search_service
            yield
            return
        built = build_services(Settings())
        try:
            with _open_search_tools_or_log(built) as prepared_statement:
                app.state.services = built
                app.state.search_service = build_search_service(
                    built, prepared_statement=prepared_statement
                )
                yield
        finally:
            close_services(built)

    app = FastAPI(title="knowledge base", lifespan=lifespan)
    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(auth.router, prefix=API_V1)
    app.include_router(users.router, prefix=API_V1)
    app.include_router(users.me_router, prefix=API_V1)
    app.include_router(keys.router, prefix=API_V1)
    app.include_router(keys.admin_router, prefix=API_V1)
    app.include_router(groups.router, prefix=API_V1)
    app.include_router(collections.router, prefix=API_V1)
    app.include_router(documents.router, prefix=API_V1)
    app.include_router(search.router, prefix=API_V1)
    return app


@contextmanager
def _open_search_tools_or_log(
    services: Services,
) -> Generator[PreparedStatement, None, None]:
    """Yield `open_search_tools`, logging in one line why it can't be built.

    Raises:
        ToolSetupError: The search tools can't be built; uvicorn then stops.
    """
    try:
        with open_search_tools(services) as prepared_statement:
            yield prepared_statement
    except ToolSetupError as error:
        # Uvicorn prints the traceback after this; the line says what to fix
        logger.error("The API can't start: %s", error)
        raise


app = create_app()
