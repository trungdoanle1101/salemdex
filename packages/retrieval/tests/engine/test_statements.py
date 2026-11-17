from typing import Any

import pytest

from salemdex_retrieval.embedders.fake import FakeDenseEmbedder
from salemdex_retrieval.embedders.protocol import DenseBinding
from salemdex_retrieval.engine.stages import Merge, Rerank, Retrieve, Rrf, Source, Take
from salemdex_retrieval.engine.statements import (
    CandidateLists,
    Candidates,
    Question,
    Search,
    replay,
    search,
)
from salemdex_retrieval.errors import StatementError
from salemdex_retrieval.index.in_memory import InMemoryIndex
from salemdex_retrieval.rerankers.fake import FakeReranker
from salemdex_retrieval.schema import One, Schema
from salemdex_retrieval.vectors import Dense


class Chunks(Schema):
    position = One(int)
    dense = Dense(size=8, distance="cosine")
    title = Dense(size=8, distance="cosine")


CHUNKS = InMemoryIndex(schema=Chunks)
EMBEDDER = FakeDenseEmbedder(size=8)
DENSE = Source(
    name="dense",
    index=CHUNKS,
    binding=DenseBinding(vector=Chunks.dense, embedder=EMBEDDER),
    depth=50,
)
TITLE = Source(
    name="title",
    index=CHUNKS,
    binding=DenseBinding(vector=Chunks.title, embedder=EMBEDDER),
    depth=50,
)
RERANKER = FakeReranker()


# Starting a statement


def test_a_search_starts_from_its_question_with_no_stages() -> None:
    question = search("thời hạn nộp hồ sơ")

    assert isinstance(question, Question)
    assert question.question == "thời hạn nộp hồ sơ"
    assert question.stages == ()


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_a_search_refuses_a_question_without_text(question: str) -> None:
    with pytest.raises(ValueError, match="question"):
        search(question)


def test_a_search_strips_whitespace_from_either_end_of_its_question() -> None:
    assert search("  thời hạn nộp hồ sơ?\n").question == "thời hạn nộp hồ sơ?"


@pytest.mark.parametrize("state", [Question, CandidateLists, Candidates, Search])
@pytest.mark.parametrize("question", ["", "   "])
def test_a_state_built_by_hand_refuses_a_question_without_text(
    state: type[Search], question: str
) -> None:
    with pytest.raises(ValueError, match="question"):
        state(question=question, stages=())


def test_a_state_built_by_hand_refuses_a_question_that_isnt_a_str() -> None:
    with pytest.raises(TypeError, match="question"):
        Search(question=None, stages=())  # type: ignore[arg-type]


# Each step adds one stage


def test_retrieving_from_one_source_gives_candidates() -> None:
    candidates = search("q").retrieve(DENSE)

    assert isinstance(candidates, Candidates)
    assert candidates.stages == (Retrieve(sources=(DENSE,)),)


def test_retrieving_from_several_sources_gives_lists_to_merge() -> None:
    lists = search("q").retrieve(DENSE, TITLE)

    assert isinstance(lists, CandidateLists)
    assert lists.stages == (Retrieve(sources=(DENSE, TITLE)),)


def test_merging_lists_gives_candidates() -> None:
    candidates = search("q").retrieve(DENSE, TITLE).merge(Rrf())

    assert isinstance(candidates, Candidates)
    assert candidates.stages[-1] == Merge(fusion=Rrf())


def test_a_full_statement_holds_every_stage_in_order_and_its_question() -> None:
    statement = (
        search("q").retrieve(DENSE, TITLE).merge(Rrf(k=30)).rerank(RERANKER).take(10)
    )

    assert isinstance(statement, Search)
    assert statement.question == "q"
    assert statement.stages == (
        Retrieve(sources=(DENSE, TITLE)),
        Merge(fusion=Rrf(k=30)),
        Rerank(reranker=RERANKER),
        Take(limit=10),
    )


def test_reranking_can_be_repeated() -> None:
    candidates = search("q").retrieve(DENSE).rerank(RERANKER).rerank(RERANKER)

    assert [type(stage) for stage in candidates.stages] == [Retrieve, Rerank, Rerank]


def test_a_stage_checks_its_settings_when_the_step_is_taken() -> None:
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        search("q").retrieve(DENSE).take(0)


# Statements are immutable data


def test_branching_leaves_the_statement_and_the_other_branch_unchanged() -> None:
    base = search("q").retrieve(DENSE)

    short = base.take(5)
    reranked = base.rerank(RERANKER).take(5)

    assert base.stages == (Retrieve(sources=(DENSE,)),)
    assert len(short.stages) == 2
    assert len(reranked.stages) == 3


def test_two_statements_built_alike_are_equal() -> None:
    def build() -> Search:
        return search("q").retrieve(DENSE).take(10)

    assert build() == build()


def test_a_statement_never_shows_its_question_in_repr() -> None:
    question = "lương phòng tài chính"
    states = [
        search(question),
        search(question).retrieve(DENSE, TITLE),
        search(question).retrieve(DENSE),
        search(question).retrieve(DENSE).take(10),
    ]

    for state in states:
        assert question not in repr(state)
        assert state.question == question


# The order of steps: refused by mypy (each `type: ignore` must stay needed, or mypy
# reports it unused) and at run time


def test_nothing_but_a_merge_follows_several_lists() -> None:
    lists = search("q").retrieve(DENSE, TITLE)

    with pytest.raises(AttributeError):
        lists.rerank(RERANKER)  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        lists.take(10)  # type: ignore[attr-defined]


def test_a_question_must_be_retrieved_before_anything_else() -> None:
    with pytest.raises(AttributeError):
        search("q").take(10)  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        search("q").merge(Rrf())  # type: ignore[attr-defined]


def test_one_list_needs_no_merge() -> None:
    with pytest.raises(AttributeError):
        search("q").retrieve(DENSE).merge(Rrf())  # type: ignore[attr-defined]


def test_a_complete_search_takes_no_further_step() -> None:
    complete = search("q").retrieve(DENSE).take(10)

    with pytest.raises(AttributeError):
        complete.take(5)  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        complete.rerank(RERANKER)  # type: ignore[attr-defined]


# Statements built at run time are replayed through the steps


@pytest.mark.parametrize(
    "statement",
    [
        pytest.param(search("q").retrieve(DENSE).take(10), id="retrieve, take"),
        pytest.param(
            search("q")
            .retrieve(DENSE, TITLE)
            .merge(Rrf())
            .rerank(RERANKER)
            .rerank(RERANKER)
            .take(3),
            id="retrieve, merge, rerank twice, take",
        ),
    ],
)
def test_replaying_a_statement_built_by_the_steps_gives_an_equal_one(
    statement: Search,
) -> None:
    assert replay(statement) == statement


def test_replaying_a_statement_built_by_hand_gives_it_as_the_steps_would() -> None:
    # As config might give it: a list, not a tuple
    by_hand = Search(
        question=" q ",
        stages=[Retrieve(sources=(DENSE,)), Take(limit=10)],  # type: ignore[arg-type]
    )

    assert replay(by_hand) == search("q").retrieve(DENSE).take(10)


@pytest.mark.parametrize(
    ("stages", "message"),
    [
        pytest.param(
            (Take(limit=5),),
            r"stages\[0\] is a Take, which can't follow Question",
            id="no retrieve first",
        ),
        pytest.param(
            (Retrieve(sources=(DENSE,)), Retrieve(sources=(TITLE,)), Take(limit=5)),
            r"stages\[1\] is a Retrieve, which can't follow Candidates",
            id="a second retrieve",
        ),
        pytest.param(
            (Retrieve(sources=(DENSE, TITLE)), Take(limit=5)),
            r"stages\[1\] is a Take, which can't follow CandidateLists",
            id="several lists not merged",
        ),
        pytest.param(
            (Retrieve(sources=(DENSE,)), Merge(fusion=Rrf()), Take(limit=5)),
            r"stages\[1\] is a Merge, which can't follow Candidates",
            id="one list merged",
        ),
        pytest.param(
            (Retrieve(sources=(DENSE,)), Take(limit=5), Take(limit=3)),
            r"stages\[2\] is a Take, which can't follow Search",
            id="a stage after the take",
        ),
        pytest.param(
            (Retrieve(sources=(DENSE,)), "rerank", Take(limit=5)),
            r"stages\[1\] is a str, which can't follow Candidates",
            id="something that isn't a stage",
        ),
    ],
)
def test_replaying_refuses_a_stage_the_state_before_it_doesnt_offer(
    stages: tuple[Any, ...], message: str
) -> None:
    with pytest.raises(StatementError, match=message):
        replay(Search(question="q", stages=stages))


@pytest.mark.parametrize(
    ("stages", "state"),
    [
        pytest.param((), "Question", id="no stages"),
        pytest.param((Retrieve(sources=(DENSE, TITLE)),), "CandidateLists", id="lists"),
        pytest.param(
            (Retrieve(sources=(DENSE,)), Rerank(reranker=RERANKER)),
            "Candidates",
            id="one list",
        ),
    ],
)
def test_replaying_refuses_stages_that_stop_before_a_take(
    stages: tuple[Any, ...], state: str
) -> None:
    with pytest.raises(StatementError, match=f"ends at {state}; end with a Take"):
        replay(Search(question="q", stages=stages))
