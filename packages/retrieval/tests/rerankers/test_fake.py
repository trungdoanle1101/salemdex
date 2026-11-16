import unicodedata

from salemdex_retrieval.rerankers.fake import FakeReranker
from salemdex_retrieval.rerankers.protocol import score_texts


def test_the_same_question_and_text_get_the_same_score_from_any_instance() -> None:
    first = FakeReranker().score("thời hạn nộp hồ sơ", ["nộp hồ sơ trong 30 ngày"])
    second = FakeReranker().score("thời hạn nộp hồ sơ", ["nộp hồ sơ trong 30 ngày"])

    assert first == second


def test_the_score_is_the_share_of_question_words_the_text_holds() -> None:
    scores = FakeReranker().score(
        "deadline for the filing",
        ["deadline for the filing", "the filing", "nothing related"],
    )

    assert scores == [1.0, 0.5, 0.0]


def test_a_long_text_with_every_question_word_outranks_a_short_one_with_half() -> None:
    long_text = "the filing deadline " + "padding " * 50
    short_text = "deadline only"

    long_score, short_score = FakeReranker().score(
        "filing deadline", [long_text, short_text]
    )

    assert long_score == 1.0
    assert short_score == 0.5


def test_repeated_words_count_once() -> None:
    scores = FakeReranker().score(
        "deadline deadline filing", ["deadline deadline deadline", "deadline"]
    )

    assert scores == [0.5, 0.5]


def test_case_punctuation_and_unicode_form_dont_change_the_score() -> None:
    question = "Hạn nộp?"
    decomposed = unicodedata.normalize("NFD", "HẠN, NỘP!")

    [score] = FakeReranker().score(question, [decomposed])

    assert score == 1.0


def test_a_question_without_words_scores_every_text_zero() -> None:
    scores = FakeReranker().score("?!", ["some text", ""])

    assert scores == [0.0, 0.0]


def test_its_scores_pass_the_checks_on_reranker_output() -> None:
    texts = ["the filing deadline", "unrelated", ""]

    scores = score_texts("filing deadline", texts, reranker=FakeReranker())

    assert scores == [1.0, 0.0, 0.0]
