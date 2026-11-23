import dataclasses
import uuid

import pytest

from salemdex_retrieval.chunks import Chunk, EmbeddedChunk
from salemdex_retrieval.embedders.fake import FakeDenseEmbedder
from salemdex_retrieval.embedders.protocol import DenseBinding
from salemdex_retrieval.engine.engine import Engine, PreparedStatement
from salemdex_retrieval.engine.results import Found, Fused, Reranked
from salemdex_retrieval.engine.sources import VectorSource
from salemdex_retrieval.engine.stages import Rrf, source
from salemdex_retrieval.engine.statements import Statement, search
from salemdex_retrieval.engine.validity import AlwaysValid
from salemdex_retrieval.evaluation.questions import KeyPoint, Question, Quote
from salemdex_retrieval.evaluation.runs import ReturnedChunk, Run, run_statement
from salemdex_retrieval.filters import Everything
from salemdex_retrieval.index.in_memory import InMemoryIndex
from salemdex_retrieval.rerankers.fake import FakeReranker
from salemdex_retrieval.schema import One, Schema
from salemdex_retrieval.vectors import Dense

# Invented text only: the real question sets are restricted and never appear in tests

SIZE = 64


class Notes(Schema):
    document = One(str, indexed=True)
    body = Dense(size=SIZE, distance="cosine")
    title = Dense(size=SIZE, distance="cosine")


EMBEDDER = FakeDenseEmbedder(size=SIZE)
FINGERPRINT = uuid.uuid4()

# The body and the title of each note hold different words, so a search of the
# bodies and a search of the titles find different notes, and a merge's union is
# larger than either source's depth
NOTES = [
    ("handbook", "rain falls in spring", "calendar"),
    ("handbook", "rain falls in autumn", "calendar"),
    ("handbook", "rain falls in winter", "calendar"),
    ("guide", "a short list of tools", "sun and heat"),
    ("guide", "a long list of tools", "sun and wind"),
    ("guide", "an empty list of tools", "sun and snow"),
]


def build_index() -> InMemoryIndex:
    index = InMemoryIndex(schema=Notes)
    index.put(embed_note(*note) for note in NOTES)
    return index


def embed_note(document: str, body: str, title: str) -> EmbeddedChunk:
    """Return a note whose body vector embeds `body` and title vector `title`."""
    [body_numbers, title_numbers] = EMBEDDER.embed_passages([body, title])
    return EmbeddedChunk(
        chunk=Chunk(
            id=uuid.uuid4(),
            text=body,
            field_values=[Notes.document.value(document)],
        ),
        vector_values=[
            Notes.body.value(body_numbers),
            Notes.title.value(title_numbers),
        ],
    )


def prepare(statement: Statement, *, index: InMemoryIndex) -> PreparedStatement:
    engine = Engine(
        validity=AlwaysValid(),
        sources={
            "body": VectorSource(
                index=index, binding=DenseBinding(vector=Notes.body, embedder=EMBEDDER)
            ),
            "title": VectorSource(
                index=index, binding=DenseBinding(vector=Notes.title, embedder=EMBEDDER)
            ),
        },
        rerankers={"cross_encoder": FakeReranker()},
    )
    return engine.prepare(statement)


def question(id: str, text: str = "rain sun", *, answerable: bool = True) -> Question:
    key_points = (
        (KeyPoint(quotes=(Quote(text="rain", where=Everything()),)),)
        if answerable
        else ()
    )
    return Question(id=id, text=text, tags=(), key_points=key_points)


def run(statement: Statement, *questions: Question) -> tuple[Run, PreparedStatement]:
    prepared = prepare(statement, index=build_index())
    returned = run_statement(
        prepared, questions=questions, fingerprint=FINGERPRINT, where=Everything()
    )
    return returned, prepared


BODY_ONLY = search().retrieve(source("body", depth=2)).take()
BODY_AND_TITLE = (
    search()
    .retrieve(source("body", depth=2), source("title", depth=2))
    .merge(Rrf())
    .take()
)
BODY_THEN_RERANK = (
    search().retrieve(source("body", depth=4)).rerank("cross_encoder").take()
)


def test_a_run_holds_one_list_for_each_question_by_its_id() -> None:
    result, _ = run(BODY_ONLY, question("q-01"), question("q-02", "list of tools"))

    assert list(result.chunks_by_question) == ["q-01", "q-02"]


def test_an_unanswerable_question_is_run_like_any_other() -> None:
    result, _ = run(BODY_ONLY, question("q-01", answerable=False))

    assert len(result.chunks_by_question["q-01"]) == 2


def test_no_questions_give_an_empty_run() -> None:
    result, _ = run(BODY_ONLY)

    assert dict(result.chunks_by_question) == {}


@pytest.mark.parametrize(
    ("statement", "candidates"),
    [(BODY_ONLY, 2), (BODY_AND_TITLE, 4), (BODY_THEN_RERANK, 4)],
    ids=["one source", "a merge", "a rerank"],
)
def test_a_run_keeps_every_candidate_the_statement_finds(
    statement: Statement, candidates: int
) -> None:
    result, _ = run(statement, question("q-01"))

    assert len(result.chunks_by_question["q-01"]) == candidates


@pytest.mark.parametrize(
    "statement",
    [BODY_ONLY, BODY_AND_TITLE, BODY_THEN_RERANK],
    ids=["one source", "a merge", "a rerank"],
)
def test_a_run_keeps_the_order_and_trails_the_statement_returned(
    statement: Statement,
) -> None:
    result, prepared = run(statement, question("q-01"))
    # A limit far above any depth: what the statement returns when nothing is cut
    passages = prepared.execute(question="rain sun", limit=1000, where=Everything())

    assert result.chunks_by_question["q-01"] == tuple(
        ReturnedChunk(chunk_id=passage.chunk.id, trail=passage.trail)
        for passage in passages
    )


def test_each_returned_chunk_carries_every_stage_that_ranked_it() -> None:
    merged, _ = run(BODY_AND_TITLE, question("q-01"))
    reranked, _ = run(BODY_THEN_RERANK, question("q-01"))

    assert all(
        isinstance(returned.trail[0], Found) and isinstance(returned.trail[-1], Fused)
        for returned in merged.chunks_by_question["q-01"]
    )
    assert all(
        [type(entry) for entry in returned.trail] == [Found, Reranked]
        for returned in reranked.chunks_by_question["q-01"]
    )


def test_a_run_holds_chunk_ids_and_trails_but_no_text() -> None:
    result, _ = run(BODY_ONLY, question("q-01"))

    [returned, *_] = result.chunks_by_question["q-01"]
    assert isinstance(returned, ReturnedChunk)
    assert isinstance(returned.chunk_id, uuid.UUID)
    assert [field.name for field in dataclasses.fields(returned)] == [
        "chunk_id",
        "trail",
    ]


def test_a_run_finds_only_the_chunks_where_allows() -> None:
    index = build_index()
    in_guide = {chunk.id for chunk in index.find(where=Notes.document.eq("guide"))}
    prepared = prepare(BODY_AND_TITLE, index=index)

    result = run_statement(
        prepared,
        questions=[question("q-01")],
        fingerprint=FINGERPRINT,
        where=Notes.document.eq("guide"),
    )

    returned_ids = {returned.chunk_id for returned in result.chunks_by_question["q-01"]}
    assert returned_ids
    assert returned_ids <= in_guide


def test_a_run_records_its_statement_and_the_evaluators_fingerprint() -> None:
    result, _ = run(BODY_AND_TITLE, question("q-01"))

    assert result.statement == BODY_AND_TITLE
    assert result.evaluator_fingerprint == FINGERPRINT


def test_a_run_cannot_be_changed() -> None:
    result, _ = run(BODY_ONLY, question("q-01"))

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.statement = BODY_AND_TITLE  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.chunks_by_question["q-02"] = ()  # type: ignore[index]
