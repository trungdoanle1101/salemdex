"""`BM25Embedder`: BM25's term-frequency part for passages, presence for questions.

The index supplies IDF (`Sparse(modifier="idf")`), so together they score as BM25; the
last test checks that end to end against the textbook formula.
"""

import math
import uuid
from collections.abc import Sequence
from typing import Any

import pytest

from salemdex_retrieval.chunks import Chunk, EmbeddedChunk
from salemdex_retrieval.embedders import (
    BM25Embedder,
    SparseEmbedder,
    measure_average_length,
    split_words,
)
from salemdex_retrieval.filters import Everything
from salemdex_retrieval.index import InMemoryIndex
from salemdex_retrieval.schema import One, Schema
from salemdex_retrieval.vectors import Sparse

# Invented text only

# Pinned: a change to the hash would leave every index unsearchable, so it fails here
LEAVE = 167927192


def weight(
    frequency: int, *, length: int, average_length: float, k1: float, b: float
) -> float:
    """Return BM25's term-frequency part for a term `frequency` times in a passage."""
    norm = k1 * (1 - b + b * length / average_length)
    return frequency * (k1 + 1) / (frequency + norm)


def term_id(term: str) -> int:
    """Return the id the embedder gives `term`, read from a one-word question."""
    [weights] = BM25Embedder(average_length=4).embed_queries([term])
    [found] = weights
    return found


def test_it_is_a_sparse_embedder() -> None:
    usable: SparseEmbedder = BM25Embedder(average_length=4)

    assert isinstance(usable, BM25Embedder)


# Words


def test_split_words_case_folds_and_drops_punctuation() -> None:
    assert split_words("Annual LEAVE: twelve days, per year.") == [
        "annual",
        "leave",
        "twelve",
        "days",
        "per",
        "year",
    ]


def test_split_words_keeps_vietnamese_diacritics_which_tell_words_apart() -> None:
    assert split_words("lao lão Lão") == ["lao", "lão", "lão"]


def test_split_words_reads_a_letter_and_its_combining_mark_as_the_letter() -> None:
    # "ạ" as one character, and as "a" followed by a combining dot below
    assert split_words("ạ") == split_words("ạ")


def test_split_words_of_a_text_without_words_is_empty() -> None:
    assert split_words(" ... ") == []


# Ids


def test_a_terms_id_is_pinned_for_every_process_and_version() -> None:
    assert term_id("leave") == LEAVE
    assert term_id("lão") == 2551386314
    assert term_id("điều") == 3316639078


def test_a_question_and_a_passage_give_one_term_the_same_id() -> None:
    embedder = BM25Embedder(average_length=4)

    [passage] = embedder.embed_passages(["leave"])
    [question] = embedder.embed_queries(["Leave?"])

    assert passage.keys() == question.keys() == {LEAVE}


# Passages: BM25's term-frequency part


def test_a_passage_weighs_each_term_by_bm25s_term_frequency_part() -> None:
    embedder = BM25Embedder(average_length=4, k1=1.2, b=0.75)

    [weights] = embedder.embed_passages(["leave days leave"])

    assert weights[LEAVE] == pytest.approx(
        weight(2, length=3, average_length=4, k1=1.2, b=0.75)
    )
    assert weights[term_id("days")] == pytest.approx(
        weight(1, length=3, average_length=4, k1=1.2, b=0.75)
    )


def test_repeating_a_term_adds_less_each_time() -> None:
    embedder = BM25Embedder(average_length=8)

    once, twice, thrice = (
        embedder.embed_passages([" ".join(["leave"] * count + ["x"] * (4 - count))])[0][
            LEAVE
        ]
        for count in (1, 2, 3)
    )

    assert once < twice < thrice
    assert twice - once > thrice - twice


def test_a_longer_passage_weighs_a_term_less() -> None:
    embedder = BM25Embedder(average_length=4)

    [short, long] = embedder.embed_passages(["leave days", "leave " + "x " * 10])

    assert short[LEAVE] > long[LEAVE]


def test_with_b_zero_length_is_ignored() -> None:
    embedder = BM25Embedder(average_length=4, b=0.0)

    [short, long] = embedder.embed_passages(["leave days", "leave " + "x " * 10])

    assert short[LEAVE] == pytest.approx(long[LEAVE])


def test_with_k1_zero_every_term_present_weighs_one() -> None:
    embedder = BM25Embedder(average_length=4, k1=0.0)

    [weights] = embedder.embed_passages(["leave leave leave days"])

    assert weights == {LEAVE: pytest.approx(1.0), term_id("days"): pytest.approx(1.0)}


def test_a_passage_without_words_has_no_weights() -> None:
    assert BM25Embedder(average_length=4).embed_passages([" ... ", ""]) == [{}, {}]


def test_each_passage_gets_its_own_weights_in_order() -> None:
    embedder = BM25Embedder(average_length=4)

    weights = embedder.embed_passages(["leave", "days", "leave"])

    assert [set(each) for each in weights] == [{LEAVE}, {term_id("days")}, {LEAVE}]
    assert embedder.embed_passages([]) == []


def test_the_default_average_length_is_256() -> None:
    [weights] = BM25Embedder().embed_passages(["leave"])

    assert weights[LEAVE] == pytest.approx(
        weight(1, length=1, average_length=256, k1=1.2, b=0.75)
    )


# Questions: presence


def test_a_question_weighs_each_distinct_term_one_however_often_it_appears() -> None:
    [weights] = BM25Embedder(average_length=4).embed_queries(["leave days, leave?"])

    assert weights == {LEAVE: 1.0, term_id("days"): 1.0}


def test_a_question_without_words_has_no_weights() -> None:
    assert BM25Embedder(average_length=4).embed_queries(["?"]) == [{}]


# Tokenizing


def test_a_tokenizer_given_replaces_split_words_on_both_sides() -> None:
    def by_bar(text: str) -> Sequence[str]:
        return text.split("|")

    embedder = BM25Embedder(average_length=4, tokenize=by_bar)

    [passage] = embedder.embed_passages(["annual leave|days"])
    [question] = embedder.embed_queries(["annual leave"])

    assert len(passage) == 2
    assert question.keys() <= passage.keys()


# Settings


@pytest.mark.parametrize(
    ("settings", "error", "named"),
    [
        pytest.param({"average_length": 0}, ValueError, "average_length", id="zero"),
        pytest.param({"average_length": -1.0}, ValueError, "average_length", id="neg"),
        pytest.param(
            {"average_length": math.inf}, ValueError, "average_length", id="inf"
        ),
        pytest.param(
            {"average_length": math.nan}, ValueError, "average_length", id="nan"
        ),
        pytest.param({"average_length": "4"}, TypeError, "average_length", id="str"),
        pytest.param({"k1": -0.1}, ValueError, "k1", id="negative k1"),
        pytest.param({"k1": math.nan}, ValueError, "k1", id="nan k1"),
        pytest.param({"k1": math.inf}, ValueError, "k1", id="inf k1"),
        pytest.param({"k1": "1.2"}, TypeError, "k1", id="str k1"),
        pytest.param({"k1": True}, TypeError, "k1", id="bool k1"),
        pytest.param({"b": -0.1}, ValueError, r"\bb\b", id="b below 0"),
        pytest.param({"b": 1.1}, ValueError, r"\bb\b", id="b above 1"),
        pytest.param({"b": math.nan}, ValueError, r"\bb\b", id="nan b"),
        pytest.param({"b": True}, TypeError, r"\bb\b", id="bool b"),
    ],
)
def test_a_setting_out_of_range_is_refused_naming_it(
    settings: dict[str, Any], error: type[Exception], named: str
) -> None:
    with pytest.raises(error, match=named):
        BM25Embedder(**settings)


def test_a_fractional_average_length_is_taken() -> None:
    [weights] = BM25Embedder(average_length=5.4).embed_passages(["leave"])

    assert weights[LEAVE] == pytest.approx(
        weight(1, length=1, average_length=5.4, k1=1.2, b=0.75)
    )


@pytest.mark.parametrize(("k1", "b"), [(0.0, 0.0), (2.0, 1.0)])
def test_the_ends_of_each_range_are_taken(k1: float, b: float) -> None:
    BM25Embedder(average_length=4, k1=k1, b=b)


# Measuring the average length


def test_the_average_length_is_the_mean_number_of_words_unrounded() -> None:
    assert measure_average_length(["annual leave", "days of annual leave, yes"]) == 3.5


def test_measuring_reads_words_as_the_tokenizer_given_splits_them() -> None:
    def by_bar(text: str) -> Sequence[str]:
        return text.split("|")

    assert measure_average_length(["a b|c", "d"], tokenize=by_bar) == 1.5


def test_measuring_no_texts_is_refused() -> None:
    with pytest.raises(ValueError, match="no text"):
        measure_average_length([])


# With the index: BM25


class Docs(Schema):
    position = One(int)
    terms = Sparse(modifier="idf")


TEXTS = [
    "annual leave is twelve days",
    "leave days carry over to the next year leave",
    "late payment is charged interest",
    "scope of application",
    "the year of the leave",
]


def textbook_bm25(texts: list[str], question: str) -> dict[int, float]:
    """Return each text's BM25 score for `question`, from the formula, by position."""
    documents = [split_words(text) for text in texts]
    average_length = sum(map(len, documents)) / len(documents)
    terms = set(split_words(question))
    scores: dict[int, float] = {}
    for position, document in enumerate(documents):
        score = 0.0
        for term in terms:
            frequency = document.count(term)
            if frequency:
                holding = sum(term in other for other in documents)
                idf = math.log((len(documents) - holding + 0.5) / (holding + 0.5) + 1)
                score += idf * weight(
                    frequency,
                    length=len(document),
                    average_length=average_length,
                    k1=1.2,
                    b=0.75,
                )
        if score:
            scores[position] = score
    return scores


def test_with_an_idf_vector_the_index_scores_passages_by_bm25() -> None:
    embedder = BM25Embedder(average_length=measure_average_length(TEXTS))
    index = InMemoryIndex(schema=Docs)
    index.put(
        EmbeddedChunk(
            chunk=Chunk(
                id=uuid.UUID(int=position + 1),
                text=text,
                field_values=[Docs.position.value(position)],
            ),
            vector_values=[Docs.terms.value(weights)],
        )
        for position, (text, weights) in enumerate(
            zip(TEXTS, embedder.embed_passages(TEXTS), strict=True)
        )
    )
    question = "how many leave days per year"
    [asked] = embedder.embed_queries([question])

    hits = index.search(Docs.terms.query(asked), where=Everything(), limit=10)

    expected = textbook_bm25(TEXTS, question)
    assert {hit.chunk[Docs.position] for hit in hits} == set(expected)
    for hit in hits:
        assert hit.score == pytest.approx(expected[hit.chunk[Docs.position]])
