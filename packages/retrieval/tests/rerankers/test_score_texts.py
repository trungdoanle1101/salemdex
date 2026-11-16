import math
from collections.abc import Sequence

import pytest

from salemdex_retrieval.rerankers.protocol import score_texts


class FixedReranker:
    """A reranker returning the scores it was built with, recording every call."""

    def __init__(self, scores: Sequence[object]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, question: str, texts: Sequence[str]) -> Sequence[float]:
        self.calls.append((question, list(texts)))
        # The scores may hold anything, to test the checks on what an adapter returns
        return self.scores  # type: ignore[return-value]


def test_each_text_gets_its_score_in_order() -> None:
    reranker = FixedReranker([0.2, 0.9, 0.5])

    scores = score_texts("question", ["a", "b", "c"], reranker=reranker)

    assert scores == [0.2, 0.9, 0.5]


def test_the_reranker_gets_the_question_and_every_text_in_one_call() -> None:
    reranker = FixedReranker([0.2, 0.9])

    score_texts("question", ["a", "b"], reranker=reranker)

    assert reranker.calls == [("question", ["a", "b"])]


def test_integer_scores_come_back_as_floats() -> None:
    scores = score_texts("question", ["a", "b"], reranker=FixedReranker([1, 2]))

    assert scores == [1.0, 2.0]
    assert all(type(score) is float for score in scores)


def test_no_texts_give_no_scores_without_calling_the_reranker() -> None:
    reranker = FixedReranker([])

    scores = score_texts("question", [], reranker=reranker)

    assert scores == []
    assert reranker.calls == []


@pytest.mark.parametrize("scores", [[0.5], [0.5, 0.5, 0.5]])
def test_a_reranker_returning_another_number_of_scores_is_refused(
    scores: list[float],
) -> None:
    reranker = FixedReranker(scores)

    with pytest.raises(ValueError, match="2 texts"):
        score_texts("question", ["a", "b"], reranker=reranker)


@pytest.mark.parametrize("bad_score", [math.nan, math.inf, -math.inf])
def test_a_score_that_isnt_finite_is_refused(bad_score: float) -> None:
    reranker = FixedReranker([0.5, bad_score])

    with pytest.raises(ValueError, match="finite"):
        score_texts("question", ["a", "b"], reranker=reranker)


@pytest.mark.parametrize("bad_score", [True, "0.5", None])
def test_a_score_that_isnt_a_real_number_is_refused(bad_score: object) -> None:
    reranker = FixedReranker([0.5, bad_score])

    with pytest.raises(TypeError):
        score_texts("question", ["a", "b"], reranker=reranker)
