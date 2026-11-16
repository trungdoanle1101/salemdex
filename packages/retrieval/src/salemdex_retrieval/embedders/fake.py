"""A deterministic embedder for tests: each row is built from the words of its text.

It needs no model, network or torch, so the engine's tests and the server's can embed
chunks and questions anywhere. The technique is feature hashing (the "hashing trick"):
each word is hashed to one position of the row, so texts sharing words point the same
way.
"""

import hashlib
import math
import re
import unicodedata
from collections.abc import Sequence
from typing import Final


class FakeDenseEmbedder:
    """A dense embedder whose rows are bags of hashed words, for tests.

    Each word, normalised to NFC and case-folded, adds +1 or -1 at a position chosen
    by its hash; the row is then scaled to unit length. Tests can rely on:

    - the same text gets the same row, from any instance and in any process;
    - a question gets the same row as an identical passage;
    - the more words two texts share, the higher they score, up to 1.0 for the same
      words;
    - case, punctuation and the Unicode form of a word don't change the row;
    - a text without words gets a zero row, which scores 0.0 under cosine.

    Different words can share a position. Their signs make such collisions cancel on
    average, but a test asserting a ranking should use a large `size` (e.g. 256) so
    that few words collide.

    Attributes:
        size: How many numbers each row holds; the same as the bound vector's `size`.
    """

    def __init__(self, *, size: int) -> None:
        """Build an embedder producing rows of `size` numbers.

        Args:
            size: The row length, the same as the bound `Dense` vector's `size`.

        Raises:
            ValueError: `size` isn't a positive `int` (a `bool` doesn't count).
        """
        if not (type(size) is int and size > 0):
            raise ValueError(f"size must be a positive integer, got {size!r}")
        self.size: Final = size

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one row per question in `texts`, in the same order.

        Questions are embedded exactly like passages.

        Args:
            texts: The questions.

        Returns:
            One row of `size` numbers per text, of unit length or all zeros.
        """
        return [self._embed_text(text) for text in texts]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one row per passage in `texts`, in the same order.

        Args:
            texts: The passages: each chunk's `embedded_text`.

        Returns:
            One row of `size` numbers per text, of unit length or all zeros.
        """
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> list[float]:
        """Return the row for `text`: its hashed words, scaled to unit length."""
        row = [0.0] * self.size
        for word in _words(text):
            position, sign = _hash_word(word, size=self.size)
            row[position] += sign
        # A text without words has no direction, so its row stays all zeros
        norm = math.hypot(*row)
        return [x / norm for x in row] if norm > 0 else row


def _words(text: str) -> list[str]:
    """Return the words of `text`, normalised to NFC and case-folded."""
    # NFC first: a Vietnamese letter such as "ạ" can be one character or a letter
    # plus a combining mark, and both must hash alike
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def _hash_word(word: str, *, size: int) -> tuple[int, float]:
    """Return the position in a row of `size` that `word` adds to, and its sign."""
    # hashlib, not hash(): the built-in is salted per process
    digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
    number = int.from_bytes(digest, "little")
    position = number % size
    # The top bit picks the sign, so collisions between unrelated words cancel on
    # average instead of always adding similarity
    sign = 1.0 if number >> 63 == 0 else -1.0
    return position, sign
