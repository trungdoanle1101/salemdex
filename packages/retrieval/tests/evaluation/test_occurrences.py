import dataclasses
import uuid

import pytest

from salemdex_retrieval.chunks import Chunk
from salemdex_retrieval.errors import SchemaMismatchError
from salemdex_retrieval.evaluation.occurrences import (
    QuoteOccurrences,
    find_quote_occurrences,
)
from salemdex_retrieval.evaluation.questions import KeyPoint, Question, Quote
from salemdex_retrieval.filters import Everything, Filter
from salemdex_retrieval.schema import One, Schema

# Invented text only: the real question sets are restricted and never appear in tests


class Notes(Schema):
    document = One(str, indexed=True)


class Titles(Schema):
    title = One(str)


IN_HANDBOOK = Notes.document.eq("handbook")
IN_GUIDE = Notes.document.eq("guide")


def chunk(text: str, *, document: str = "handbook") -> Chunk:
    return Chunk(
        id=uuid.uuid4(), text=text, field_values=[Notes.document.value(document)]
    )


def quote(text: str, *, where: Filter = IN_HANDBOOK) -> Quote:
    return Quote(text=text, where=where)


def question(id: str, *key_points: KeyPoint) -> Question:
    return Question(
        id=id, text="Can staff work from home?", tags=(), key_points=key_points
    )


def key_point(*quotes: Quote) -> KeyPoint:
    return KeyPoint(quotes=quotes)


HOME = chunk("Staff may work from home.")
FRIDAYS = chunk("Except on Fridays.")
GUIDE_HOME = chunk("Staff may work from home.", document="guide")
HOURS = chunk("The office opens at 8.")
CORPUS = [HOME, FRIDAYS, GUIDE_HOME, HOURS]


def only(occurrences: list[QuoteOccurrences]) -> QuoteOccurrences:
    [found] = occurrences
    return found


# What a quote's occurrences say


def test_a_quote_in_its_place_is_found_in_the_chunks_holding_it() -> None:
    found = only(
        find_quote_occurrences(
            [question("q-01", key_point(quote("work from home")))], chunks=CORPUS
        )
    )

    assert found.is_found
    assert found.chunk_ids_in_place == (HOME.id,)


def test_a_quote_also_outside_its_place_lists_those_chunks_apart() -> None:
    found = only(
        find_quote_occurrences(
            [question("q-01", key_point(quote("work from home")))], chunks=CORPUS
        )
    )

    assert found.chunk_ids_elsewhere == (GUIDE_HOME.id,)


def test_a_quote_whose_where_is_wrong_is_found_only_elsewhere() -> None:
    # The loader mapped it to a document the text isn't in
    in_other = quote("except on fridays", where=Notes.document.eq("other"))

    found = only(
        find_quote_occurrences([question("q-01", key_point(in_other))], chunks=CORPUS)
    )

    assert not found.is_found
    assert found.chunk_ids_in_place == ()
    assert found.chunk_ids_elsewhere == (FRIDAYS.id,)


def test_a_quote_whose_text_is_nowhere_is_found_nowhere() -> None:
    found = only(
        find_quote_occurrences(
            [question("q-01", key_point(quote("remote work is allowed")))],
            chunks=CORPUS,
        )
    )

    assert not found.is_found
    assert found.chunk_ids_in_place == ()
    assert found.chunk_ids_elsewhere == ()


def test_a_quote_split_across_two_chunks_is_found_nowhere() -> None:
    found = only(
        find_quote_occurrences(
            [question("q-01", key_point(quote("home. except")))], chunks=CORPUS
        )
    )

    assert not found.is_found
    assert found.chunk_ids_elsewhere == ()


def test_a_quote_held_by_several_chunks_lists_them_all_in_corpus_order() -> None:
    again = chunk("As said, staff may work from home.")

    found = only(
        find_quote_occurrences(
            [question("q-01", key_point(quote("work from home", where=Everything())))],
            chunks=[HOME, HOURS, GUIDE_HOME, again],
        )
    )

    assert found.chunk_ids_in_place == (HOME.id, GUIDE_HOME.id, again.id)


def test_a_quote_matches_whatever_its_case_and_whitespace() -> None:
    found = only(
        find_quote_occurrences(
            [question("q-01", key_point(quote("STAFF  may\nwork")))], chunks=CORPUS
        )
    )

    assert found.chunk_ids_in_place == (HOME.id,)


def test_a_quote_only_in_the_embedded_text_is_found_nowhere() -> None:
    headed = Chunk(
        id=uuid.uuid4(),
        text="Staff may rest.",
        embedded_text="Leave policy > Staff may rest.",
        field_values=[Notes.document.value("handbook")],
    )

    found = only(
        find_quote_occurrences(
            [question("q-01", key_point(quote("leave policy")))], chunks=[headed]
        )
    )

    assert not found.is_found
    assert found.chunk_ids_elsewhere == ()


# One record per quote


def test_every_quote_gets_a_record_naming_its_question_key_point_and_position() -> None:
    questions = [
        question(
            "q-01",
            key_point(quote("work from home"), quote("remote work is allowed")),
            key_point(quote("except on fridays")),
        ),
        question("q-02", key_point(quote("opens at 8"))),
    ]

    occurrences = find_quote_occurrences(questions, chunks=CORPUS)

    assert [(o.question_id, o.key_point, o.quote, o.is_found) for o in occurrences] == [
        ("q-01", 0, 0, True),
        ("q-01", 0, 1, False),
        ("q-01", 1, 0, True),
        ("q-02", 0, 0, True),
    ]


def test_an_alternative_not_found_is_listed_even_when_another_is() -> None:
    # The key point can still be found, but the broken quote is reported for fixing
    either = key_point(quote("work from home"), quote("remote work is allowed"))

    occurrences = find_quote_occurrences([question("q-01", either)], chunks=CORPUS)

    assert [o.is_found for o in occurrences] == [True, False]


def test_an_unanswerable_question_has_no_quotes_to_find() -> None:
    assert find_quote_occurrences([question("q-01")], chunks=CORPUS) == []


def test_no_questions_give_no_occurrences() -> None:
    assert find_quote_occurrences([], chunks=CORPUS) == []


def test_an_empty_corpus_finds_no_quote() -> None:
    found = only(
        find_quote_occurrences(
            [question("q-01", key_point(quote("work from home")))], chunks=[]
        )
    )

    assert not found.is_found
    assert found.chunk_ids_elsewhere == ()


def test_questions_and_chunks_may_be_generators() -> None:
    questions = (q for q in [question("q-01", key_point(quote("work from home")))])
    chunks = (c for c in CORPUS)

    found = only(find_quote_occurrences(questions, chunks=chunks))

    assert found.chunk_ids_in_place == (HOME.id,)


# What it trusts


def test_a_where_outside_the_chunks_schema_raises_on_a_chunk_holding_the_text() -> None:
    # The evaluator checks every `where` against the index's schema first; the text
    # is checked before the place, so only a chunk holding the text reaches it
    wrong_schema = quote("work from home", where=Titles.title.eq("Leave"))

    with pytest.raises(SchemaMismatchError):
        find_quote_occurrences(
            [question("q-01", key_point(wrong_schema))], chunks=CORPUS
        )


def test_quote_occurrences_cant_be_changed() -> None:
    found = only(
        find_quote_occurrences(
            [question("q-01", key_point(quote("work from home")))], chunks=CORPUS
        )
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        found.chunk_ids_in_place = ()  # type: ignore[misc]
