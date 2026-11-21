from typing import Any

import pytest

from salemdex_retrieval.engine.stages import (
    Merge,
    Rerank,
    Retrieve,
    Rrf,
    Take,
    source,
)
from salemdex_retrieval.engine.statements import (
    CandidateLists,
    Candidates,
    Question,
    Statement,
    replay,
    search,
)
from salemdex_retrieval.errors import StatementError

DENSE = source("dense", depth=50)
TITLE = source("title", depth=50)
RERANKER = "cross_encoder"


# Starting a statement


def test_a_search_starts_with_no_stages() -> None:
    question = search()

    assert isinstance(question, Question)
    assert question.stages == ()


# Each step adds one stage


def test_retrieving_from_one_source_gives_candidates() -> None:
    candidates = search().retrieve(DENSE)

    assert isinstance(candidates, Candidates)
    assert candidates.stages == (Retrieve(sources=(DENSE,)),)


def test_retrieving_from_several_sources_gives_lists_to_merge() -> None:
    lists = search().retrieve(DENSE, TITLE)

    assert isinstance(lists, CandidateLists)
    assert lists.stages == (Retrieve(sources=(DENSE, TITLE)),)


def test_merging_lists_gives_candidates() -> None:
    candidates = search().retrieve(DENSE, TITLE).merge(Rrf())

    assert isinstance(candidates, Candidates)
    assert candidates.stages[-1] == Merge(fusion=Rrf())


def test_a_full_statement_holds_every_stage_in_order() -> None:
    statement = search().retrieve(DENSE, TITLE).merge(Rrf(k=30)).rerank(RERANKER).take()

    assert isinstance(statement, Statement)
    assert statement.stages == (
        Retrieve(sources=(DENSE, TITLE)),
        Merge(fusion=Rrf(k=30)),
        Rerank(reranker=RERANKER),
        Take(),
    )


def test_reranking_can_be_repeated() -> None:
    candidates = search().retrieve(DENSE).rerank(RERANKER).rerank(RERANKER)

    assert [type(stage) for stage in candidates.stages] == [Retrieve, Rerank, Rerank]


def test_a_stage_checks_its_settings_when_the_step_is_taken() -> None:
    with pytest.raises(ValueError, match="reranker"):
        search().retrieve(DENSE).rerank("  ")


def test_a_retrieve_step_refuses_two_sources_with_the_same_name() -> None:
    with pytest.raises(ValueError, match="share a name: 'dense'"):
        search().retrieve(DENSE, source("dense", depth=20))


# Statements are immutable data


def test_branching_leaves_the_statement_and_the_other_branch_unchanged() -> None:
    base = search().retrieve(DENSE)

    short = base.take()
    reranked = base.rerank(RERANKER).take()

    assert base.stages == (Retrieve(sources=(DENSE,)),)
    assert len(short.stages) == 2
    assert len(reranked.stages) == 3


def test_two_statements_built_alike_are_equal() -> None:
    # Names and settings only, so two built apart compare equal: the same
    # configuration, whoever built it
    def build() -> Statement:
        return search().retrieve(source("dense", depth=50)).rerank("bge").take()

    assert build() == build()


def test_statements_differing_only_in_a_depth_are_not_equal() -> None:
    shallow = search().retrieve(source("dense", depth=20)).take()
    deep = search().retrieve(source("dense", depth=50)).take()

    assert shallow != deep


# The order of steps: refused by mypy (each `type: ignore` must stay needed, or mypy
# reports it unused) and at run time


def test_nothing_but_a_merge_follows_several_lists() -> None:
    lists = search().retrieve(DENSE, TITLE)

    with pytest.raises(AttributeError):
        lists.rerank(RERANKER)  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        lists.take()  # type: ignore[attr-defined]


def test_a_question_must_be_retrieved_before_anything_else() -> None:
    with pytest.raises(AttributeError):
        search().take()  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        search().merge(Rrf())  # type: ignore[attr-defined]


def test_one_list_needs_no_merge() -> None:
    with pytest.raises(AttributeError):
        search().retrieve(DENSE).merge(Rrf())  # type: ignore[attr-defined]


def test_a_complete_statement_takes_no_further_step() -> None:
    complete = search().retrieve(DENSE).take()

    with pytest.raises(AttributeError):
        complete.take()  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        complete.rerank(RERANKER)  # type: ignore[attr-defined]


# Statements built at run time are replayed through the steps


@pytest.mark.parametrize(
    "statement",
    [
        pytest.param(search().retrieve(DENSE).take(), id="retrieve, take"),
        pytest.param(
            search()
            .retrieve(DENSE, TITLE)
            .merge(Rrf())
            .rerank(RERANKER)
            .rerank(RERANKER)
            .take(),
            id="retrieve, merge, rerank twice, take",
        ),
    ],
)
def test_replaying_a_statement_built_by_the_steps_gives_an_equal_one(
    statement: Statement,
) -> None:
    assert replay(statement) == statement


def test_replaying_a_statement_built_by_hand_gives_it_as_the_steps_would() -> None:
    # As config might give it: a list, not a tuple
    by_hand = Statement(
        stages=[Retrieve(sources=(DENSE,)), Take()],  # type: ignore[arg-type]
    )

    assert replay(by_hand) == search().retrieve(DENSE).take()


@pytest.mark.parametrize(
    ("stages", "message"),
    [
        pytest.param(
            (Take(),),
            r"stages\[0\] is a Take, which can't follow Question",
            id="no retrieve first",
        ),
        pytest.param(
            (Retrieve(sources=(DENSE,)), Retrieve(sources=(TITLE,)), Take()),
            r"stages\[1\] is a Retrieve, which can't follow Candidates",
            id="a second retrieve",
        ),
        pytest.param(
            (Retrieve(sources=(DENSE, TITLE)), Take()),
            r"stages\[1\] is a Take, which can't follow CandidateLists",
            id="several lists not merged",
        ),
        pytest.param(
            (Retrieve(sources=(DENSE,)), Merge(fusion=Rrf()), Take()),
            r"stages\[1\] is a Merge, which can't follow Candidates",
            id="one list merged",
        ),
        pytest.param(
            (Retrieve(sources=(DENSE,)), Take(), Take()),
            r"stages\[2\] is a Take, which can't follow Statement",
            id="a stage after the take",
        ),
        pytest.param(
            (Retrieve(sources=(DENSE,)), "rerank", Take()),
            r"stages\[1\] is a str, which can't follow Candidates",
            id="something that isn't a stage",
        ),
    ],
)
def test_replaying_refuses_a_stage_the_state_before_it_doesnt_offer(
    stages: tuple[Any, ...], message: str
) -> None:
    with pytest.raises(StatementError, match=message):
        replay(Statement(stages=stages))


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
        replay(Statement(stages=stages))
