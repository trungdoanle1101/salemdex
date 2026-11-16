"""The interface every reranker adapter implements, and how its scores are checked.

A reranker judges how well each passage answers a question, reading both together,
which ranks better than comparing vectors but costs a model call per passage. The app
supplies it as an adapter, a client for a hosted model or a wrapper around a local one,
speaking plain data only. The framework checks every score it returns
(`score_texts`), so an adapter needn't.
"""

from collections.abc import Sequence
from typing import Protocol

from salemdex_retrieval.checks import to_finite_float


class Reranker(Protocol):
    """A model scoring how well each text answers a question, one score per text.

    The app implements it, for a hosted model (a cross-encoder service, an LLM judge)
    or a local one. It returns scores, not an order: the executor sorts the candidates
    and records each score in their trail.

    Higher is better. Scores are comparable only within one call, never across
    questions or models.

    Nothing here names the model, for the same reason as `DenseEmbedder`: the
    framework couldn't verify it, and a mismatch costs ranking, not a guarantee.
    """

    def score(self, question: str, texts: Sequence[str]) -> Sequence[float]:
        """Return one score per text in `texts`, in the same order.

        Args:
            question: The question, as asked.
            texts: The passages to judge against it.

        Returns:
            One score per text; higher means a better answer.
        """
        ...


def score_texts(
    question: str, texts: Sequence[str], *, reranker: Reranker
) -> list[float]:
    """Return the reranker's score for each text, checked.

    The reranker receives every text in one call. No texts give no scores without
    calling it, since some services refuse an empty batch.

    Args:
        question: The question to judge the texts against.
        texts: The passages to score.
        reranker: The model scoring them.

    Returns:
        One finite score per text, in the same order.

    Raises:
        ValueError: The reranker returned a different number of scores than texts,
            or a score isn't finite.
        TypeError: A score isn't a real number (a `str` or a `bool`, say).

    Example:
        scores = score_texts(question, [c.embedded_text for c in kept], reranker=tei)
    """
    if not texts:
        return []

    scores = reranker.score(question, texts)
    # Scores pair with texts by position: a wrong count would swap candidates' scores
    if len(texts) != len(scores):
        raise ValueError(
            f"the reranker returned {len(scores)} scores for {len(texts)} texts"
        )
    # A nan compares false with everything, so sorting would go wrong silently
    return [
        to_finite_float(score, name=f"the rerank score for texts[{position}]")
        for position, score in enumerate(scores)
    ]
