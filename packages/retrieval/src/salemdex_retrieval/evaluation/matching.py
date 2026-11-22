"""What "a chunk holds a quote" means, in one place.

A chunk holds a quote when it matches the quote's `where` and its text, never its
embedded text, contains the quote's once both are normalised (`normalize_text`). The
quote judge and the quote occurrences both match through `QuoteMatcher`, so they can't
disagree about whether a chunk holds a quote.
"""

import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

from salemdex_retrieval.chunks import Chunk
from salemdex_retrieval.evaluation.questions import Quote
from salemdex_retrieval.filters import build_chunk_matcher


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
