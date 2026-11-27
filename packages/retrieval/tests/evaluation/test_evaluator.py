import dataclasses
import uuid
from collections.abc import Callable, Iterable

import pytest

from salemdex_retrieval.chunks import Chunk, EmbeddedChunk
from salemdex_retrieval.embedders.fake import FakeDenseEmbedder
from salemdex_retrieval.embedders.protocol import DenseBinding
from salemdex_retrieval.engine.engine import Engine, PreparedStatement
from salemdex_retrieval.engine.results import Found
from salemdex_retrieval.engine.sources import SearchableIndex, VectorScan
from salemdex_retrieval.engine.stages import scan
from salemdex_retrieval.engine.statements import search
from salemdex_retrieval.engine.validity import AlwaysValid
from salemdex_retrieval.errors import SchemaMismatchError
from salemdex_retrieval.evaluation.evaluator import Evaluator
from salemdex_retrieval.evaluation.judges.protocol import Judgment
from salemdex_retrieval.evaluation.judges.quote import QuoteJudge
from salemdex_retrieval.evaluation.questions import KeyPoint, Question, Quote
from salemdex_retrieval.evaluation.runs import ReturnedChunk, Run
from salemdex_retrieval.filters import Filter
from salemdex_retrieval.index.in_memory import InMemoryIndex
from salemdex_retrieval.schema import One, Schema
from salemdex_retrieval.vectors import Dense

# Invented text only: the real question sets are restricted and never appear in tests

SIZE = 64


class Notes(Schema):
    document = One(str, indexed=True)
    body = Dense(size=SIZE, distance="cosine")


class Other(Schema):
    document = One(str, indexed=True)


EMBEDDER = FakeDenseEmbedder(size=SIZE)
IN_HANDBOOK = Notes.document.eq("handbook")
IN_GUIDE = Notes.document.eq("guide")

NOTES = [
    ("handbook", "staff may work from home two days a week"),
    ("handbook", "requests go to the team lead in writing"),
    ("guide", "the team lead answers within five days"),
]


class CountingIndex(InMemoryIndex):
    """An in-memory index counting how often its chunks are read."""

    def __init__(self) -> None:
        super().__init__(schema=Notes)
        self.finds = 0

    def find(self, *, where: Filter) -> list[Chunk]:
        self.finds += 1
        return super().find(where=where)


def build_index() -> CountingIndex:
    index = CountingIndex()
    index.put(embed_note(document, text) for document, text in NOTES)
    return index


def embed_note(document: str, text: str) -> EmbeddedChunk:
    [numbers] = EMBEDDER.embed_passages([text])
    return EmbeddedChunk(
        chunk=Chunk(
            id=uuid.uuid4(), text=text, field_values=[Notes.document.value(document)]
        ),
        vector_values=[Notes.body.value(numbers)],
    )


def quote(text: str, *, where: Filter = IN_HANDBOOK) -> Quote:
    return Quote(text=text, where=where)


def key_point(*quotes: Quote) -> KeyPoint:
    return KeyPoint(quotes=quotes)


def question(id: str, *key_points: KeyPoint) -> Question:
    return Question(
        id=id, text="Can staff work from home?", tags=(), key_points=key_points
    )


FOUND = quote("two days a week")
ALSO_FOUND = quote("in writing")
NOT_FOUND = quote("three days a week")
ONLY_ELSEWHERE = quote("within five days")  # in the guide, but placed in the handbook


def test_building_refuses_an_empty_question_set() -> None:
    with pytest.raises(ValueError, match="question set is empty"):
        Evaluator.from_index([], index=build_index())


def test_building_refuses_two_questions_sharing_an_id() -> None:
    # Ids are stripped when a question is built, so these two are alike
    questions = [question("q-01", key_point(FOUND)), question(" q-01 ")]

    with pytest.raises(ValueError, match="'q-01'"):
        Evaluator.from_index(questions, index=build_index())


def test_a_quote_placed_by_another_schemas_field_is_refused_before_the_index_is_read() -> (
    None
):
    index = build_index()
    misplaced = quote("two days a week", where=Other.document.eq("handbook"))

    with pytest.raises(SchemaMismatchError):
        Evaluator.from_index([question("q-01", key_point(misplaced))], index=index)
    assert index.finds == 0


def test_the_corpus_is_every_chunk_of_the_index_by_id_read_once() -> None:
    index = build_index()

    evaluator = Evaluator.from_index([question("q-01")], index=index)

    assert [chunk.text for chunk in evaluator.corpus.values()] == [
        text for _, text in NOTES
    ]
    assert all(chunk_id == chunk.id for chunk_id, chunk in evaluator.corpus.items())
    assert index.finds == 1


def test_the_corpus_is_a_snapshot_chunks_indexed_later_arent_in_it() -> None:
    index = build_index()
    evaluator = Evaluator.from_index([question("q-01")], index=index)

    index.put([embed_note("handbook", "a note written later")])

    assert len(evaluator.corpus) == len(NOTES)


def test_the_evaluator_keeps_its_own_copy_of_the_questions() -> None:
    questions = [question("q-01")]
    evaluator = Evaluator.from_index(questions, index=build_index())

    questions.append(question("q-02"))

    assert [q.id for q in evaluator.questions] == ["q-01"]


def test_an_evaluator_cannot_be_changed() -> None:
    evaluator = Evaluator.from_index([question("q-01")], index=build_index())

    with pytest.raises(dataclasses.FrozenInstanceError):
        evaluator.fingerprint = uuid.uuid4()  # type: ignore[misc]
    with pytest.raises(TypeError):
        evaluator.corpus[uuid.uuid4()] = next(iter(evaluator.corpus.values()))  # type: ignore[index]


def test_each_evaluator_has_its_own_fingerprint() -> None:
    questions = [question("q-01")]
    index = build_index()

    first = Evaluator.from_index(questions, index=index)
    second = Evaluator.from_index(questions, index=index)

    assert first.fingerprint != second.fingerprint


def test_every_quote_has_a_record_of_where_it_occurs() -> None:
    evaluator = Evaluator.from_index(
        [
            question("q-01", key_point(FOUND, NOT_FOUND)),
            question("q-02", key_point(ALSO_FOUND)),
        ],
        index=build_index(),
    )

    assert [
        (found.question_id, found.key_point, found.quote, found.is_found)
        for found in evaluator.quote_occurrences
    ] == [("q-01", 0, 0, True), ("q-01", 0, 1, False), ("q-02", 0, 0, True)]


def test_quotes_not_found_lists_only_the_quotes_no_chunk_in_their_place_holds() -> None:
    evaluator = Evaluator.from_index(
        [
            question("q-01", key_point(FOUND, NOT_FOUND)),
            question("q-02", key_point(ONLY_ELSEWHERE)),
        ],
        index=build_index(),
    )

    assert [
        (found.question_id, found.key_point, found.quote)
        for found in evaluator.quotes_not_found
    ] == [("q-01", 0, 1), ("q-02", 0, 0)]


def test_a_question_is_gradable_when_every_key_point_has_a_quote_found() -> None:
    evaluator = Evaluator.from_index(
        [question("q-01", key_point(FOUND), key_point(ALSO_FOUND))],
        index=build_index(),
    )

    assert [q.id for q in evaluator.gradable_questions] == ["q-01"]


def test_a_question_is_not_gradable_when_a_key_point_has_no_quote_found() -> None:
    evaluator = Evaluator.from_index(
        [question("q-01", key_point(FOUND), key_point(NOT_FOUND))],
        index=build_index(),
    )

    assert evaluator.gradable_questions == ()


def test_a_quote_found_only_outside_its_place_doesnt_make_its_question_gradable() -> (
    None
):
    evaluator = Evaluator.from_index(
        [question("q-01", key_point(ONLY_ELSEWHERE))], index=build_index()
    )

    assert evaluator.gradable_questions == ()


def test_one_alternative_found_is_enough_though_the_other_stays_listed() -> None:
    evaluator = Evaluator.from_index(
        [question("q-01", key_point(NOT_FOUND, FOUND))], index=build_index()
    )

    assert [q.id for q in evaluator.gradable_questions] == ["q-01"]
    assert [found.quote for found in evaluator.quotes_not_found] == [0]


def test_an_unanswerable_question_is_gradable() -> None:
    evaluator = Evaluator.from_index([question("q-01")], index=build_index())

    assert [q.id for q in evaluator.gradable_questions] == ["q-01"]


def test_gradable_questions_keep_the_order_given() -> None:
    evaluator = Evaluator.from_index(
        [
            question("q-03", key_point(FOUND)),
            question("q-01", key_point(NOT_FOUND)),
            question("q-02"),
        ],
        index=build_index(),
    )

    assert [q.id for q in evaluator.gradable_questions] == ["q-03", "q-02"]


def prepare_body_search(index: InMemoryIndex) -> PreparedStatement:
    engine = Engine(
        validity=AlwaysValid(),
        indexes={
            "chunks": SearchableIndex(
                index=index,
                scans={
                    "body": VectorScan(
                        binding=DenseBinding(vector=Notes.body, embedder=EMBEDDER)
                    )
                },
            )
        },
        rerankers={},
    )
    return engine.prepare(search("chunks").retrieve(scan("body", depth=3)).take())


def test_a_run_asks_every_question_gradable_or_not_and_carries_the_fingerprint() -> (
    None
):
    index = build_index()
    evaluator = Evaluator.from_index(
        [question("q-01", key_point(FOUND)), question("q-02", key_point(NOT_FOUND))],
        index=index,
    )

    run = evaluator.run_statement(prepare_body_search(index))

    assert list(run.chunks_by_question) == ["q-01", "q-02"]
    assert all(len(chunks) == len(NOTES) for chunks in run.chunks_by_question.values())
    assert run.evaluator_fingerprint == evaluator.fingerprint


def test_a_run_finds_only_the_chunks_where_allows() -> None:
    index = build_index()
    evaluator = Evaluator.from_index([question("q-01")], index=index)
    in_guide = {chunk.id for chunk in index.find(where=IN_GUIDE)}

    run = evaluator.run_statement(prepare_body_search(index), where=IN_GUIDE)

    assert {
        returned.chunk_id for returned in run.chunks_by_question["q-01"]
    } == in_guide


def test_a_run_searches_every_chunk_unless_told_otherwise() -> None:
    index = build_index()
    evaluator = Evaluator.from_index([question("q-01")], index=index)

    run = evaluator.run_statement(prepare_body_search(index))

    assert {returned.chunk_id for returned in run.chunks_by_question["q-01"]} == set(
        evaluator.corpus
    )


def test_gradable_questions_and_quotes_not_found_are_worked_out_once() -> None:
    evaluator = Evaluator.from_index(
        [question("q-01", key_point(FOUND, NOT_FOUND))], index=build_index()
    )

    assert evaluator.gradable_questions is evaluator.gradable_questions
    assert evaluator.quotes_not_found is evaluator.quotes_not_found


# Judging: runs built by hand, so each pool is exactly what a test says

STATEMENT = search("chunks").retrieve(scan("body", depth=3)).take()


def run_of(
    evaluator: Evaluator,
    chunk_ids_by_question: dict[str, list[uuid.UUID]],
    *,
    fingerprint: uuid.UUID | None = None,
) -> Run:
    return Run(
        statement=STATEMENT,
        chunks_by_question={
            question_id: tuple(
                ReturnedChunk(
                    chunk_id=chunk_id,
                    trail=(Found(scan="body", rank=rank, score=0.5),),
                )
                for rank, chunk_id in enumerate(chunk_ids, start=1)
            )
            for question_id, chunk_ids in chunk_ids_by_question.items()
        },
        evaluator_fingerprint=fingerprint or evaluator.fingerprint,
    )


class RecordingJudge(QuoteJudge):
    """The quote judge, recording which chunks it's given for which question."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[uuid.UUID]]] = []

    def judge(self, question: Question, chunks: Iterable[Chunk]) -> list[Judgment]:
        chunks = list(chunks)
        self.calls.append((question.id, [chunk.id for chunk in chunks]))
        return super().judge(question, chunks)


class BrokenJudge(QuoteJudge):
    """The quote judge, its judgments spoilt by `spoil` before they're returned."""

    def __init__(self, spoil: Callable[[list[Judgment]], list[Judgment]]) -> None:
        self.spoil = spoil

    def judge(self, question: Question, chunks: Iterable[Chunk]) -> list[Judgment]:
        return self.spoil(super().judge(question, chunks))


def evaluator_with_ids(
    *questions: Question,
) -> tuple[Evaluator, list[uuid.UUID]]:
    """Return an evaluator of `questions`, and its chunk ids in the order of NOTES."""
    evaluator = Evaluator.from_index(questions, index=build_index())
    return evaluator, list(evaluator.corpus)


def test_the_pool_is_every_runs_top_chunks_in_the_order_first_seen() -> None:
    evaluator, [home, writing, answers] = evaluator_with_ids(question("q-01"))
    first = run_of(evaluator, {"q-01": [answers, home, writing]})
    second = run_of(evaluator, {"q-01": [writing, answers, home]})
    judge = RecordingJudge()

    evaluator.judge_top_chunks([first, second], depth=2, judge=judge)

    assert judge.calls == [("q-01", [answers, home, writing])]


def test_a_chunk_two_runs_share_is_judged_once() -> None:
    evaluator, [home, writing, _] = evaluator_with_ids(question("q-01"))
    first = run_of(evaluator, {"q-01": [home, writing]})
    second = run_of(evaluator, {"q-01": [home, writing]})
    judge = RecordingJudge()

    assessment = evaluator.judge_top_chunks([first, second], depth=2, judge=judge)

    assert judge.calls == [("q-01", [home, writing])]
    assert len(assessment.judgments) == 2


def test_chunks_below_the_depth_are_left_unjudged() -> None:
    evaluator, [home, writing, answers] = evaluator_with_ids(question("q-01"))
    run = run_of(evaluator, {"q-01": [home, writing, answers]})

    assessment = evaluator.judge_top_chunks([run], depth=2, judge=QuoteJudge())

    assert assessment.judgment_of("q-01", writing) is not None
    assert assessment.judgment_of("q-01", answers) is None


def test_the_assessment_records_the_depth_it_was_judged_to() -> None:
    evaluator, [home, writing, answers] = evaluator_with_ids(question("q-01"))
    run = run_of(evaluator, {"q-01": [home, writing, answers]})

    assessment = evaluator.judge_top_chunks([run], depth=2, judge=QuoteJudge())

    assert assessment.judged_depth == 2


def test_every_question_is_judged_gradable_or_not() -> None:
    evaluator, [home, *_] = evaluator_with_ids(
        question("q-01", key_point(FOUND)),
        question("q-02", key_point(NOT_FOUND)),
        question("q-03"),
    )
    run = run_of(evaluator, {"q-01": [home], "q-02": [home], "q-03": [home]})
    judge = RecordingJudge()

    evaluator.judge_top_chunks([run], depth=1, judge=judge)

    assert [question_id for question_id, _ in judge.calls] == ["q-01", "q-02", "q-03"]


def test_the_assessment_holds_the_judges_grades_under_its_name() -> None:
    evaluator, [home, writing, _] = evaluator_with_ids(
        question("q-01", key_point(FOUND))
    )
    run = run_of(evaluator, {"q-01": [home, writing]})

    assessment = evaluator.judge_top_chunks([run], depth=2, judge=QuoteJudge())

    assert assessment.judge == "quote"
    holds_quote = assessment.judgment_of("q-01", home)
    lacks_quote = assessment.judgment_of("q-01", writing)
    assert holds_quote is not None and lacks_quote is not None
    assert (holds_quote.grade, holds_quote.key_points) == ("answered", (0,))
    assert (lacks_quote.grade, lacks_quote.key_points) == ("unrelated", ())


def test_the_assessment_cannot_be_changed_through_its_judgments() -> None:
    evaluator, [home, *_] = evaluator_with_ids(question("q-01"))
    run = run_of(evaluator, {"q-01": [home]})

    assessment = evaluator.judge_top_chunks([run], depth=1, judge=QuoteJudge())

    with pytest.raises(TypeError):
        assessment.judgments[("q-02", home)] = assessment.judgments[("q-01", home)]  # type: ignore[index]


@pytest.mark.parametrize("depth", [0, -1])
def test_judging_refuses_a_depth_that_isnt_positive(depth: int) -> None:
    evaluator, [home, *_] = evaluator_with_ids(question("q-01"))
    run = run_of(evaluator, {"q-01": [home]})

    with pytest.raises(ValueError, match="depth"):
        evaluator.judge_top_chunks([run], depth=depth, judge=QuoteJudge())


def test_judging_refuses_no_runs() -> None:
    evaluator, _ = evaluator_with_ids(question("q-01"))

    with pytest.raises(ValueError, match="no runs"):
        evaluator.judge_top_chunks([], depth=1, judge=QuoteJudge())


def test_judging_refuses_a_run_from_another_evaluator() -> None:
    evaluator, [home, *_] = evaluator_with_ids(question("q-01"))
    other = run_of(evaluator, {"q-01": [home]}, fingerprint=uuid.uuid4())

    with pytest.raises(ValueError, match="another evaluator"):
        evaluator.judge_top_chunks([other], depth=1, judge=QuoteJudge())


def test_judging_refuses_a_chunk_the_corpus_doesnt_hold() -> None:
    evaluator, _ = evaluator_with_ids(question("q-01"))
    run = run_of(evaluator, {"q-01": [uuid.uuid4()]})

    with pytest.raises(ValueError, match="isn't in the corpus"):
        evaluator.judge_top_chunks([run], depth=1, judge=QuoteJudge())


def drop_last(judged: list[Judgment]) -> list[Judgment]:
    return judged[:-1]


def judge_first_twice(judged: list[Judgment]) -> list[Judgment]:
    return [judged[0], *judged]


def swap_first_for_another_chunk(judged: list[Judgment]) -> list[Judgment]:
    return [dataclasses.replace(judged[0], chunk_id=uuid.uuid4()), *judged[1:]]


def name_another_question(judged: list[Judgment]) -> list[Judgment]:
    return [dataclasses.replace(judged[0], question_id="q-99"), *judged[1:]]


def name_another_judge(judged: list[Judgment]) -> list[Judgment]:
    return [dataclasses.replace(judged[0], judge="llm"), *judged[1:]]


@pytest.mark.parametrize(
    ("spoil", "message"),
    [
        (drop_last, "left 1 of the 2 given unjudged"),
        (judge_first_twice, "twice"),
        (swap_first_for_another_chunk, "1 chunks it wasn't given"),
        (name_another_question, "for question 'q-99'"),
        (name_another_judge, "named for judge 'llm'"),
    ],
)
def test_judging_refuses_a_judge_that_doesnt_judge_each_chunk_given_once(
    spoil: Callable[[list[Judgment]], list[Judgment]], message: str
) -> None:
    evaluator, [home, writing, _] = evaluator_with_ids(question("q-01"))
    run = run_of(evaluator, {"q-01": [home, writing]})

    with pytest.raises(ValueError, match=message):
        evaluator.judge_top_chunks([run], depth=2, judge=BrokenJudge(spoil))


def name_a_key_point_the_question_lacks(judged: list[Judgment]) -> list[Judgment]:
    return [
        dataclasses.replace(judged[0], grade="answered", key_points=(0, 1)),
        *judged[1:],
    ]


def test_judging_refuses_a_judgment_naming_a_key_point_the_question_lacks() -> None:
    # A judgment doesn't know its question, so only the evaluator can check this
    evaluator, [home, writing, _] = evaluator_with_ids(
        question("q-01", key_point(FOUND))
    )
    run = run_of(evaluator, {"q-01": [home, writing]})
    judge = BrokenJudge(name_a_key_point_the_question_lacks)

    with pytest.raises(ValueError, match="key point"):
        evaluator.judge_top_chunks([run], depth=2, judge=judge)


# Scoring: a run and an assessment joined, question by question


def test_scoring_leaves_out_questions_with_a_quote_not_found_and_keeps_them() -> None:
    evaluator, [home, *_] = evaluator_with_ids(
        question("q-01", key_point(FOUND)),
        question("q-02", key_point(NOT_FOUND)),
        question("q-03"),
    )
    run = run_of(evaluator, {"q-01": [home], "q-02": [home], "q-03": [home]})
    assessment = evaluator.judge_top_chunks([run], depth=1, judge=QuoteJudge())

    card = evaluator.score(run, assessment=assessment)

    # An unanswerable question is gradable: abstention scores it
    assert [row.question.id for row in card.questions] == ["q-01", "q-03"]
    assert [question.id for question in card.questions_left_out] == ["q-02"]


def test_scoring_keeps_each_questions_top_chunks_to_the_judged_depth() -> None:
    evaluator, [home, writing, answers] = evaluator_with_ids(
        question("q-01", key_point(FOUND))
    )
    run = run_of(evaluator, {"q-01": [writing, home, answers]})
    assessment = evaluator.judge_top_chunks([run], depth=2, judge=QuoteJudge())

    card = evaluator.score(run, assessment=assessment)

    [row] = card.questions
    assert [chunk.returned.chunk_id for chunk in row.chunks] == [writing, home]
    assert row.first_answered_rank == 2


def test_a_question_with_fewer_chunks_than_the_judged_depth_keeps_them_all() -> None:
    evaluator, [home, *_] = evaluator_with_ids(question("q-01", key_point(FOUND)))
    run = run_of(evaluator, {"q-01": [home]})
    assessment = evaluator.judge_top_chunks([run], depth=3, judge=QuoteJudge())

    [row] = evaluator.score(run, assessment=assessment).questions

    assert [chunk.returned.chunk_id for chunk in row.chunks] == [home]


def test_a_runs_score_doesnt_depend_on_the_runs_that_shared_its_pool() -> None:
    evaluator, [home, writing, answers] = evaluator_with_ids(
        question("q-01", key_point(FOUND))
    )
    # `home` holds the quote: third in this run, so below the depth, but first in
    # the other, so judged when the two are pooled
    run = run_of(evaluator, {"q-01": [writing, answers, home]})
    other = run_of(evaluator, {"q-01": [home, writing, answers]})

    alone = evaluator.judge_top_chunks([run], depth=2, judge=QuoteJudge())
    pooled = evaluator.judge_top_chunks([run, other], depth=2, judge=QuoteJudge())

    card_alone = evaluator.score(run, assessment=alone)
    card_pooled = evaluator.score(run, assessment=pooled)
    assert card_alone.questions == card_pooled.questions
    assert card_pooled.hit_rate_at(2) == 0.0


def test_the_scorecard_carries_the_runs_statement_and_the_judged_depth() -> None:
    evaluator, [home, *_] = evaluator_with_ids(question("q-01", key_point(FOUND)))
    run = run_of(evaluator, {"q-01": [home]})
    assessment = evaluator.judge_top_chunks([run], depth=4, judge=QuoteJudge())

    card = evaluator.score(run, assessment=assessment)

    assert card.statement == STATEMENT
    assert card.judged_depth == 4


def test_scoring_refuses_a_run_that_wasnt_judged_in_the_assessment() -> None:
    evaluator, [home, writing, _] = evaluator_with_ids(
        question("q-01", key_point(FOUND))
    )
    judged = run_of(evaluator, {"q-01": [home]})
    never_judged = run_of(evaluator, {"q-01": [writing]})
    assessment = evaluator.judge_top_chunks([judged], depth=1, judge=QuoteJudge())

    with pytest.raises(ValueError, match="wasn't judged in this assessment"):
        evaluator.score(never_judged, assessment=assessment)


def test_scoring_refuses_a_run_from_another_evaluator() -> None:
    evaluator, [home, *_] = evaluator_with_ids(question("q-01", key_point(FOUND)))
    run = run_of(evaluator, {"q-01": [home]})
    other = run_of(evaluator, {"q-01": [home]}, fingerprint=uuid.uuid4())
    assessment = evaluator.judge_top_chunks([run], depth=1, judge=QuoteJudge())

    with pytest.raises(ValueError, match="another evaluator"):
        evaluator.score(other, assessment=assessment)


# Comparing: two runs scored on one assessment


def test_comparing_scores_both_runs_on_the_one_assessment() -> None:
    evaluator, [home, writing, answers] = evaluator_with_ids(
        question("q-01", key_point(FOUND))
    )
    candidate = run_of(evaluator, {"q-01": [home, writing]})
    baseline = run_of(evaluator, {"q-01": [writing, answers]})
    assessment = evaluator.judge_top_chunks(
        [candidate, baseline], depth=2, judge=QuoteJudge()
    )

    comparison = evaluator.compare(candidate, baseline=baseline, assessment=assessment)

    assert comparison.candidate == evaluator.score(candidate, assessment=assessment)
    assert comparison.baseline == evaluator.score(baseline, assessment=assessment)


def test_comparing_finds_a_question_the_candidate_answers_and_the_baseline_misses() -> (
    None
):
    evaluator, [home, writing, answers] = evaluator_with_ids(
        question("q-01", key_point(FOUND))
    )
    candidate = run_of(evaluator, {"q-01": [home, writing]})
    baseline = run_of(evaluator, {"q-01": [writing, answers]})
    assessment = evaluator.judge_top_chunks(
        [candidate, baseline], depth=2, judge=QuoteJudge()
    )

    comparison = evaluator.compare(candidate, baseline=baseline, assessment=assessment)

    [change] = comparison.questions_improved_at(2)
    assert (change.question.id, change.baseline_rank, change.candidate_rank) == (
        "q-01",
        None,
        1,
    )


def test_comparing_refuses_a_baseline_that_wasnt_judged_in_the_assessment() -> None:
    evaluator, [home, writing, _] = evaluator_with_ids(
        question("q-01", key_point(FOUND))
    )
    candidate = run_of(evaluator, {"q-01": [home]})
    baseline = run_of(evaluator, {"q-01": [writing]})
    assessment = evaluator.judge_top_chunks([candidate], depth=1, judge=QuoteJudge())

    with pytest.raises(ValueError, match="wasn't judged in this assessment"):
        evaluator.compare(candidate, baseline=baseline, assessment=assessment)


def test_comparing_refuses_a_run_from_another_evaluator() -> None:
    evaluator, [home, *_] = evaluator_with_ids(question("q-01", key_point(FOUND)))
    candidate = run_of(evaluator, {"q-01": [home]})
    other = run_of(evaluator, {"q-01": [home]}, fingerprint=uuid.uuid4())
    assessment = evaluator.judge_top_chunks([candidate], depth=1, judge=QuoteJudge())

    with pytest.raises(ValueError, match="another evaluator"):
        evaluator.compare(candidate, baseline=other, assessment=assessment)
