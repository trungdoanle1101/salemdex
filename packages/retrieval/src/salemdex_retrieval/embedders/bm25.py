"""BM25 as a sparse embedder: term weights counted from the text, with no model.

BM25 scores a passage by the question's terms it holds, each by how often it appears,
damped by repetition and by the passage's length, and by how rare the term is across
the index. The embedder gives the first part; the index gives the rarity, as IDF, so
bind it to a `Sparse(modifier="idf")` vector:

    bm25 = BM25Embedder(average_length=measure_average_length(chunk_texts))
    SparseBinding(vector=KbChunks.sparse, embedder=bm25)

It needs no model, network or extra, and gives the same weights for the same text in
every process, so tests can use it as it is. Changing its settings or tokenizer changes
the weights stored in the index: re-index, as for a new model.
"""

import hashlib
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from salemdex_retrieval.checks import to_finite_float
from salemdex_retrieval.embedders.words import split_words


@dataclass(frozen=True, kw_only=True)
class BM25Embedder:
    """A sparse embedder weighing terms as BM25 does, for a vector with IDF.

    A passage weighs each distinct term `t` by BM25's term-frequency part,
    `tf × (k1 + 1) / (tf + k1 × (1 − b + b × length / average_length))`, where `tf` is
    how often `t` appears and `length` how many terms the passage has. A question weighs
    each distinct term 1, however often it appears. The index multiplies in each term's
    IDF, so the two together score as BM25.

    A term's id is a 32-bit blake2b hash of it: the same everywhere, with no vocabulary
    to keep. Two terms sharing an id are rare (about one pair in 100,000 terms) and
    only merge two terms, usually rare ones.

    Frozen: the settings shape the stored weights, so they can't change under an index.

    Attributes:
        average_length: The mean number of terms in a passage of the corpus, as
            `tokenize` splits them; `measure_average_length` gives it. A wrong one
            shifts every passage's length normalization alike, so it matters little.
        k1: How slowly a repeated term stops adding weight; 0 counts a term once
            however often it appears.
        b: How much a passage's length counts, from 0 (not at all) to 1 (fully).
        tokenize: How a text splits into terms; `split_words` by default. Not in the
            repr: a function's repr changes from run to run.

    Example:
        bm25 = BM25Embedder(average_length=402.5)
        bm25.embed_passages(["Annual leave is twelve days"])
    """

    average_length: float = 256.0
    k1: float = 1.2
    b: float = 0.75
    tokenize: Callable[[str], Sequence[str]] = field(default=split_words, repr=False)

    def __post_init__(self) -> None:
        """Check each setting; nothing is converted, so nothing is stored.

        Raises:
            TypeError: A setting isn't a real number (a `str` or a `bool`, say).
            ValueError: A setting isn't finite, `average_length` isn't above 0, `k1`
                is below 0, or `b` is outside 0 to 1.
        """
        if to_finite_float(self.average_length, name="average_length") <= 0:
            raise ValueError(
                f"average_length must be above 0, got {self.average_length}"
            )
        if to_finite_float(self.k1, name="k1") < 0:
            raise ValueError(f"k1 must be at least 0, got {self.k1}")
        if not (0 <= to_finite_float(self.b, name="b") <= 1):
            raise ValueError(f"b must be from 0 to 1, got {self.b}")

    def embed_passages(self, texts: Sequence[str]) -> list[dict[int, float]]:
        """Return each passage's weight per term id, by BM25's term-frequency part.

        Args:
            texts: The passages: each chunk's `embedded_text`.

        Returns:
            One mapping per text, in order; empty for a text without terms.
        """
        result: list[dict[int, float]] = []
        for text in texts:
            terms = self.tokenize(text)
            norm = self.k1 * (1 - self.b + self.b * len(terms) / self.average_length)
            term_frequencies = Counter(_term_id(term) for term in terms)
            weights = {
                term_id: frequency * (self.k1 + 1) / (frequency + norm)
                for term_id, frequency in term_frequencies.items()
            }
            result.append(weights)
        return result

    def embed_queries(self, texts: Sequence[str]) -> list[dict[int, float]]:
        """Return each question's terms, each weighing 1, by term id.

        Args:
            texts: The questions, as asked.

        Returns:
            One mapping per text, in order; empty for a text without terms.
        """
        result: list[dict[int, float]] = []
        for text in texts:
            term_ids = {_term_id(term) for term in self.tokenize(text)}
            weights = {term_id: 1.0 for term_id in term_ids}
            result.append(weights)
        return result


def measure_average_length(
    texts: Sequence[str], *, tokenize: Callable[[str], Sequence[str]] = split_words
) -> float:
    """Return the mean number of terms in `texts`, for `BM25Embedder.average_length`.

    Measure it once over the corpus's chunks, with the tokenizer the embedder will use.

    Args:
        texts: The passages: each chunk's `embedded_text`.
        tokenize: How a text splits into terms; the embedder's.

    Returns:
        The mean, unrounded.

    Raises:
        ValueError: `texts` is empty: no mean exists.

    Example:
        average_length = measure_average_length(
            [chunk.embedded_text for chunk in chunks]
        )
    """
    if not texts:
        raise ValueError("no text is provided for averaging")
    return sum(len(tokenize(text)) for text in texts) / len(texts)


def _term_id(term: str) -> int:
    """Return `term`'s id: its 4-byte blake2b digest, read as an unsigned int."""
    # hashlib, not hash(): the built-in is salted per process, and ids must match
    # between the process that indexed and the one that searches
    digest = hashlib.blake2b(term.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little")
