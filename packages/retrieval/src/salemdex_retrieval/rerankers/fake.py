"""A deterministic reranker for tests: each score is the share of question words a text holds.

It needs no model, network or torch, so the engine's tests and the server's can run a
rerank stage anywhere. It deliberately ranks differently from `FakeDenseEmbedder`:
cosine there falls as a text grows longer, coverage here doesn't, so a test can show
that a rerank stage changed the order.
"""

import re
import unicodedata
from collections.abc import Sequence


class FakeReranker:
    """A reranker scoring each text by how many of the question's words it holds.

    Words are compared as sets, normalised to NFC and case-folded. Vietnamese is split
    into syllables ("hồ sơ" is two words). Tests can rely on:

    - the same question and text get the same score, from any instance;
    - the score is the share of the question's distinct words found in the text, from
      0.0 to 1.0;
    - repeating a word, or making the text longer, doesn't change the score;
    - case, punctuation and the Unicode form of a word don't change the score;
    - a question without words scores every text 0.0.

    Scores tie often (0.5, 1.0), so a test asserting an order should use texts whose
    scores differ.
    """

    def score(self, question: str, texts: Sequence[str]) -> list[float]:
        """Return one score per text in `texts`, in the same order.

        Args:
            question: The question, as asked.
            texts: The passages to judge against it.

        Returns:
            One score per text, from 0.0 (no question word) to 1.0 (every one).
        """
        question_words = set(_words(question))
        return [_score(question_words, set(_words(text))) for text in texts]


def _words(text: str) -> list[str]:
    """Return the words of `text`, normalised to NFC and case-folded."""
    # NFC first: a Vietnamese letter such as "ạ" can be one character or a letter
    # plus a combining mark, and both must compare alike
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def _score(question_words: set[str], text_words: set[str]) -> float:
    """Return the share of `question_words` found in `text_words`; 0.0 if none."""
    if not question_words:
        return 0.0
    return len(question_words & text_words) / len(question_words)
