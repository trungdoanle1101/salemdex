"""The quote judge: a chunk answers when it holds one of a key point's quotes.

It reads no model and gives the same grades every time, in microseconds, so it judges
everything it's given. A chunk holds a quote when it matches the quote's `where` and its
text, never its embedded text, contains the quote's once both are normalised
(`normalize_text`). It misses a point stated in other words, or in a place no quote
names: it can undercount but never overcount, which makes it the floor, and what an LLM
judge is checked against.

What a match is lives in `matching`, shared with the quote occurrences.
"""

from collections.abc import Iterable

from salemdex_retrieval.chunks import Chunk
from salemdex_retrieval.evaluation.judges.protocol import Judgment
from salemdex_retrieval.evaluation.matching import QuoteMatcher, normalize_text
from salemdex_retrieval.evaluation.questions import Question


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
