"""The quote judge: a chunk answers when it holds one of a key point's quotes.

It reads no model and gives the same grades every time, in microseconds, so it judges
everything it's given. A chunk holds a quote when it matches the quote's `where` and its
text, never its embedded text, contains the quote's once both are normalised
(`normalize_text`). It misses a point stated in other words, or in a place no quote
names: it can undercount but never overcount, which makes it the floor, and what an LLM
judge is checked against.

`QuoteMatcher` and `normalize_text` are shared with the facts check, so the two can't
disagree about whether a chunk holds a quote.
"""

import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Self

from salemdex_retrieval.chunks import Chunk
from salemdex_retrieval.evaluation.judges.protocol import Judgment
from salemdex_retrieval.evaluation.questions import Question, Quote
from salemdex_retrieval.filters import build_chunk_matcher


class QuoteJudge:
    """A judge grading a chunk "answered" when it holds a quote of some key point.

    A key point's quotes are alternatives, so any one supplies it; a chunk answers when it
    supplies any key point, and "unrelated" otherwise. It never gives "related".

    Example:
        judgments = QuoteJudge().judge(question, chunks)
    """

    @property
    def name(self) -> str:
        """The judge's name on its judgments: "quote"."""
        return "quote"

    def judge(self, question: Question, chunks: Iterable[Chunk]) -> list[Judgment]:
        """Return one judgment for each chunk, naming the key points it supplies.

        Each quote is prepared once, and each chunk's text normalised once, however many
        quotes it's checked against. An unanswerable question has no key points, so
        every chunk is "unrelated" to it.

        Args:
            question: The question the chunks were returned for; each quote's `where`
                already checked against the chunks' schema.
            chunks: The chunks to grade.

        Returns:
            The judgments, one per chunk, in the order given.

        Raises:
            SchemaMismatchError: A quote's `where` names a field a chunk holds no value
                for: it wasn't checked against the chunks' schema.
        """
        # Shaped like the question, so each position is the key point's in the question
        quote_matchers_per_key_points = tuple(
            tuple(QuoteMatcher.from_quote(quote) for quote in key_point.quotes)
            for key_point in question.key_points
        )

        judgments: list[Judgment] = []
        for chunk in chunks:
            normalized_text = normalize_text(chunk.text)
            found_key_points = tuple(
                key_point_position
                for key_point_position, quote_matchers in enumerate(
                    quote_matchers_per_key_points
                )
                if any(
                    quote.found_in_chunk(chunk, normalized_text=normalized_text)
                    for quote in quote_matchers
                )
            )

            judgments.append(
                Judgment(
                    question_id=question.id,
                    chunk_id=chunk.id,
                    grade="answered" if len(found_key_points) > 0 else "unrelated",
                    key_points=found_key_points,
                    judge=self.name,
                )
            )

        return judgments


@dataclass(frozen=True, kw_only=True)
class QuoteMatcher:
    """A quote prepared once for matching many chunks.

    Its text is normalised and its `where` built into a matcher once, not for every chunk.

    Attributes:
        text: The quote's text, normalised.
        chunk_matcher: Tells whether a chunk sits where the quote may (its `where`).

    Example:
        matcher = QuoteMatcher.from_quote(quote)
        matcher.found_in_chunk(chunk, normalized_text=normalize_text(chunk.text))
    """

    text: str
    chunk_matcher: Callable[[Chunk], bool]

    @classmethod
    def from_quote(cls, quote: Quote) -> Self:
        """Return the matcher for `quote`.

        Args:
            quote: The quote, its `where` already checked against the chunks' schema.

        Returns:
            The matcher, ready for any number of chunks.
        """
        return cls(
            text=normalize_text(quote.text),
            chunk_matcher=build_chunk_matcher(quote.where),
        )

    def found_in_chunk(self, chunk: Chunk, *, normalized_text: str) -> bool:
        """Return whether `chunk` holds the quote: in its place, and in its text.

        Args:
            chunk: The chunk to look in.
            normalized_text: The chunk's text, normalised with `normalize_text` once by
                the caller, so it isn't repeated for every quote.

        Returns:
            Whether the chunk matches the quote's `where` and its text contains the
            quote's.

        Raises:
            SchemaMismatchError: The quote's `where` names a field the chunk holds no
                value for.
        """
        # The place first: reading a field is cheaper than searching the text
        chunk_matched = self.chunk_matcher(chunk)
        return chunk_matched and self.text in normalized_text


def normalize_text(text: str) -> str:
    """Return `text` as quotes and chunks are compared: composed, folded, spaced.

    Composed into NFC, case-folded, and every run of whitespace a single space, so a
    quote typed with decomposed accents, in another case, or across a line break still
    matches. Diacritics and punctuation are kept: they change meaning.

    Args:
        text: A quote's or a chunk's text.

    Returns:
        The text, normalised.

    Example:
        normalize_text("Được  CHO   phép")   # "được cho phép"
    """
    text = unicodedata.normalize("NFC", text).casefold()
    # `split()` also splits on non-breaking spaces, common in documents
    text = " ".join(text.split())
    # Case-folding can undo NFC (Turkish "İ" folds to "i" and a combining dot)
    return unicodedata.normalize("NFC", text)
