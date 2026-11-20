import uuid
from typing import Annotated

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from kb_server.api.deps import CurrentPrincipal, SearchServiceDep
from kb_server.domain.search import Passage

router = APIRouter(prefix="/search", tags=["search"])

# The reranker reads the question and each passage together within its token limit,
# so a long question crowds out the passage it is judging
MAX_QUESTION_CHARS = 500

Question = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS
    ),
]


class SearchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Question
    # left out: as many as the server allows (KB_SEARCH_MAX_LIMIT)
    limit: Annotated[int, Field(gt=0)] | None = None


class PassageDocumentOut(BaseModel):
    id: uuid.UUID
    title: str
    version: int


class PassageOut(BaseModel):
    text: str
    headings: list[str]
    document: PassageDocumentOut


class SearchOut(BaseModel):
    passages: list[PassageOut]

    @classmethod
    def from_passages(cls, passages: list[Passage]) -> "SearchOut":
        return SearchOut(
            passages=[
                PassageOut(
                    text=passage.text,
                    headings=list(passage.headings),
                    document=PassageDocumentOut(
                        id=passage.document.id,
                        title=passage.document.title,
                        version=passage.document.version,
                    ),
                )
                for passage in passages
            ]
        )


@router.post("")
def search(
    body: SearchIn, principal: CurrentPrincipal, search_service: SearchServiceDep
) -> SearchOut:
    passages = search_service.search(principal, body.question, limit=body.limit)
    return SearchOut.from_passages(passages)
