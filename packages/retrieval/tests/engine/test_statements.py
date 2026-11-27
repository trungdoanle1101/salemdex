from typing import Any

import pytest

from salemdex_retrieval.engine.stages import (
    Merge,
    Rerank,
    Retrieve,
    Rrf,
    Take,
    Window,
    scan,
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
from salemdex_retrieval.schema import One, Schema

DENSE = scan("dense", depth=50)
TITLE = scan("title", depth=50)
RERANKER = "cross_encoder"


class Chunks(Schema):
    document = One(str)
    position = One(int)


WINDOW = Window(
    partition_by=(Chunks.document,), order_by=Chunks.position, chunks_each_side=1
)


# Starting a statement


def test_a_search_starts_with_no_stages() -> None:
    question = search("chunks")

    assert isinstance(question, Question)
    assert question.index == "chunks"
    assert question.stages == ()


def test_a_search_strips_whitespace_from_either_end_of_its_index_name() -> None:
    assert search(" chunks\n").index == "chunks"


@pytest.mark.parametrize("name", ["", "   ", "\n\t"])
def test_a_search_refuses_an_index_name_without_text(name: str) -> None:
    with pytest.raises(ValueError, match="index"):
        search(name)


def test_a_search_refuses_an_index_name_that_isnt_a_str() -> None:
    with pytest.raises(TypeError, match="index"):
        search(None)  # type: ignore[arg-type]


def test_every_step_keeps_the_index_the_search_started_from() -> None:
    lists = search("chunks").retrieve(DENSE, TITLE)
    candidates = lists.merge(Rrf()).rerank(RERANKER)

    assert lists.index == candidates.index == candidates.take().index == "chunks"


# Each step adds one stage


def test_retrieving_with_one_scan_gives_candidates() -> None:
    candidates = search("chunks").retrieve(DENSE)

    assert isinstance(candidates, Candidates)
    assert candidates.stages == (Retrieve(scans=(DENSE,)),)


def test_retrieving_with_several_scans_gives_lists_to_merge() -> None:
    lists = search("chunks").retrieve(DENSE, TITLE)

    assert isinstance(lists, CandidateLists)
    assert lists.stages == (Retrieve(scans=(DENSE, TITLE)),)


def test_merging_lists_gives_candidates() -> None:
    candidates = search("chunks").retrieve(DENSE, TITLE).merge(Rrf())

    assert isinstance(candidates, Candidates)
    assert candidates.stages[-1] == Merge(fusion=Rrf())


def test_a_full_statement_holds_every_stage_in_order() -> None:
    statement = (
        search("chunks").retrieve(DENSE, TITLE).merge(Rrf(k=30)).rerank(RERANKER).take()
    )

    assert isinstance(statement, Statement)
    assert statement.stages == (
        Retrieve(scans=(DENSE, TITLE)),
        Merge(fusion=Rrf(k=30)),
        Rerank(reranker=RERANKER),
        Take(),
    )


def test_taking_with_a_window_holds_it_in_the_take() -> None:
    statement = search("chunks").retrieve(DENSE).take(window=WINDOW)

    assert statement.stages[-1] == Take(window=WINDOW)


def test_taking_without_a_window_holds_none() -> None:
    assert search("chunks").retrieve(DENSE).take().stages[-1] == Take(window=None)


def test_reranking_can_be_repeated() -> None:
    candidates = search("chunks").retrieve(DENSE).rerank(RERANKER).rerank(RERANKER)

    assert [type(stage) for stage in candidates.stages] == [Retrieve, Rerank, Rerank]


def test_a_stage_checks_its_settings_when_the_step_is_taken() -> None:
    with pytest.raises(ValueError, match="reranker"):
        search("chunks").retrieve(DENSE).rerank("  ")


def test_a_retrieve_step_refuses_two_scans_with_the_same_name() -> None:
    with pytest.raises(ValueError, match="share a name: 'dense'"):
        search("chunks").retrieve(DENSE, scan("dense", depth=20))


# Statements are immutable data


def test_branching_leaves_the_statement_and_the_other_branch_unchanged() -> None:
    base = search("chunks").retrieve(DENSE)

    short = base.take()
    reranked = base.rerank(RERANKER).take()

    assert base.stages == (Retrieve(scans=(DENSE,)),)
    assert len(short.stages) == 2
    assert len(reranked.stages) == 3


def test_two_statements_built_alike_are_equal() -> None:
    # Names and settings only, so two built apart compare equal: the same
    # configuration, whoever built it
    def build() -> Statement:
        return search("chunks").retrieve(scan("dense", depth=50)).rerank("bge").take()

    assert build() == build()


def test_statements_differing_only_in_a_depth_are_not_equal() -> None:
    shallow = search("chunks").retrieve(scan("dense", depth=20)).take()
    deep = search("chunks").retrieve(scan("dense", depth=50)).take()

    assert shallow != deep


def test_statements_differing_only_in_their_index_are_not_equal() -> None:
    # Two indexes may share a schema and scan names, yet hold different chunks
    chunks = search("chunks").retrieve(DENSE).take()
    entries = search("entries").retrieve(DENSE).take()

    assert chunks != entries


def test_statements_differing_only_in_their_window_are_not_equal() -> None:
    plain = search("chunks").retrieve(DENSE).take()
    windowed = search("chunks").retrieve(DENSE).take(window=WINDOW)

    assert plain != windowed


# The order of steps: refused by mypy (each `type: ignore` must stay needed, or mypy
# reports it unused) and at run time


def test_nothing_but_a_merge_follows_several_lists() -> None:
    lists = search("chunks").retrieve(DENSE, TITLE)

    with pytest.raises(AttributeError):
        lists.rerank(RERANKER)  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        lists.take()  # type: ignore[attr-defined]


def test_a_question_must_be_retrieved_before_anything_else() -> None:
    with pytest.raises(AttributeError):
        search("chunks").take()  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        search("chunks").merge(Rrf())  # type: ignore[attr-defined]


def test_one_list_needs_no_merge() -> None:
    with pytest.raises(AttributeError):
        search("chunks").retrieve(DENSE).merge(Rrf())  # type: ignore[attr-defined]


def test_a_complete_statement_takes_no_further_step() -> None:
    complete = search("chunks").retrieve(DENSE).take()

    with pytest.raises(AttributeError):
        complete.take()  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        complete.rerank(RERANKER)  # type: ignore[attr-defined]


# Statements built at run time are replayed through the steps


@pytest.mark.parametrize(
    "statement",
    [
        pytest.param(search("chunks").retrieve(DENSE).take(), id="retrieve, take"),
        pytest.param(
            search("entries").retrieve(DENSE).take(window=WINDOW),
            id="another index, take with a window",
        ),
        pytest.param(
            search("chunks")
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
        index=" chunks",
        stages=[Retrieve(scans=(DENSE,)), Take()],  # type: ignore[arg-type]
    )

    assert replay(by_hand) == search("chunks").retrieve(DENSE).take()


@pytest.mark.parametrize(
    ("stages", "message"),
    [
        pytest.param(
            (Take(),),
            r"stages\[0\] is a Take, which can't follow Question",
            id="no retrieve first",
        ),
        pytest.param(
            (Retrieve(scans=(DENSE,)), Retrieve(scans=(TITLE,)), Take()),
            r"stages\[1\] is a Retrieve, which can't follow Candidates",
            id="a second retrieve",
        ),
        pytest.param(
            (Retrieve(scans=(DENSE, TITLE)), Take()),
            r"stages\[1\] is a Take, which can't follow CandidateLists",
            id="several lists not merged",
        ),
        pytest.param(
            (Retrieve(scans=(DENSE,)), Merge(fusion=Rrf()), Take()),
            r"stages\[1\] is a Merge, which can't follow Candidates",
            id="one list merged",
        ),
        pytest.param(
            (Retrieve(scans=(DENSE,)), Take(), Take()),
            r"stages\[2\] is a Take, which can't follow Statement",
            id="a stage after the take",
        ),
        pytest.param(
            (Retrieve(scans=(DENSE,)), "rerank", Take()),
            r"stages\[1\] is a str, which can't follow Candidates",
            id="something that isn't a stage",
        ),
    ],
)
def test_replaying_refuses_a_stage_the_state_before_it_doesnt_offer(
    stages: tuple[Any, ...], message: str
) -> None:
    with pytest.raises(StatementError, match=message):
        replay(Statement(index="chunks", stages=stages))


@pytest.mark.parametrize(
    ("stages", "state"),
    [
        pytest.param((), "Question", id="no stages"),
        pytest.param((Retrieve(scans=(DENSE, TITLE)),), "CandidateLists", id="lists"),
        pytest.param(
            (Retrieve(scans=(DENSE,)), Rerank(reranker=RERANKER)),
            "Candidates",
            id="one list",
        ),
    ],
)
def test_replaying_refuses_stages_that_stop_before_a_take(
    stages: tuple[Any, ...], state: str
) -> None:
    with pytest.raises(StatementError, match=f"ends at {state}; end with a Take"):
        replay(Statement(index="chunks", stages=stages))


def test_replaying_refuses_an_index_name_without_text() -> None:
    # A statement built by hand skipped `search`, so its index is checked here
    with pytest.raises(ValueError, match="index"):
        replay(Statement(index=" ", stages=(Retrieve(scans=(DENSE,)), Take())))


def test_replaying_refuses_a_take_whose_window_was_changed_after_it_was_built() -> None:
    # Replaying builds each stage again, so its checks run again
    take = Take()
    object.__setattr__(take, "window", "not a window")

    with pytest.raises(TypeError, match="window"):
        replay(Statement(index="chunks", stages=(Retrieve(scans=(DENSE,)), take)))


# Written out, for a log line


def test_a_statement_is_written_as_the_steps_that_build_it() -> None:
    statement = (
        search("chunks")
        .retrieve(DENSE, scan("sparse", depth=20))
        .merge(Rrf(k=30))
        .rerank(RERANKER)
        .take()
    )

    assert str(statement) == (
        "search('chunks')"
        ".retrieve(scan('dense', depth=50), scan('sparse', depth=20))"
        ".merge(Rrf(k=30))"
        ".rerank('cross_encoder')"
        ".take()"
    )


def test_a_window_is_written_with_its_fields_by_name() -> None:
    statement = search("chunks").retrieve(DENSE).take(window=WINDOW)

    assert str(statement).endswith(
        ".take(window=Window(partition_by=(document,), order_by=position, "
        "chunks_each_side=1))"
    )


def test_a_window_of_several_fields_or_none_is_written_as_python_writes_a_tuple() -> (
    None
):
    def written(*partition_by: One[Any]) -> str:
        window = Window(
            partition_by=partition_by, order_by=Chunks.position, chunks_each_side=1
        )
        return str(search("chunks").retrieve(DENSE).take(window=window))

    assert "partition_by=(document, position)," in written(
        Chunks.document, Chunks.position
    )
    assert "partition_by=()," in written()


def test_a_statement_part_way_through_is_written_up_to_its_last_step() -> None:
    assert str(search("chunks").retrieve(DENSE)) == (
        "search('chunks').retrieve(scan('dense', depth=50))"
    )


def test_writing_a_statement_built_by_hand_never_fails() -> None:
    # It may hold anything until it's replayed, and a log line must still print
    by_hand = Statement(index="chunks", stages=(Take(), "rerank"))  # type: ignore[arg-type]

    assert str(by_hand) == "search('chunks').take().<'rerank'>"


def test_the_repr_stays_the_exact_dataclass_form() -> None:
    assert repr(search("chunks")) == "Question(index='chunks', stages=())"
