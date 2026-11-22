"""Where each gold quote occurs in the corpus, found before any statement runs.

A quote that no chunk holds makes its question's score meaningless: every passage would
be judged unrelated, and a label problem would count as a retrieval miss. So each quote
is looked for in every chunk first, in its place and elsewhere:

- found in place: fine;
- found only elsewhere: its `where` is wrong, most likely the app's mapping from its
  documents to their ids;
- found nowhere: its text isn't the document's, it's split across two chunks, or its
  document isn't in the corpus.

A record is per quote, not per chunk, so it says which alternative of a key point is
broken; a judgment can't. Records hold ids and positions, never text, so they can be
logged.
"""

import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from salemdex_retrieval.chunks import Chunk
from salemdex_retrieval.evaluation.matching import QuoteMatcher, normalize_text
from salemdex_retrieval.evaluation.questions import Question


@dataclass(frozen=True, kw_only=True)
class QuoteOccurrences:
    """Where one gold quote occurs among the corpus's chunks.

    The quote is named by its question and its positions, not held, since its text is
    the document's.

    Attributes:
        question_id: The id of the question the quote belongs to.
        key_point: The position of the quote's key point in the question's
            `key_points`.
        quote: The quote's position in that key point's `quotes`.
        chunk_ids_in_place: The chunks matching the quote's `where` whose text holds
            it, in corpus order; empty when it isn't found.
        chunk_ids_elsewhere: The chunks outside its `where` whose text holds it, in
            corpus order: copies of the text elsewhere, or a sign the `where` is wrong.
    """

    question_id: str
    key_point: int
    quote: int
    chunk_ids_in_place: tuple[uuid.UUID, ...]
    chunk_ids_elsewhere: tuple[uuid.UUID, ...]

    @property
    def is_found(self) -> bool:
        """Whether some chunk in the quote's place holds it."""
        return bool(self.chunk_ids_in_place)


def find_quote_occurrences(
    questions: Iterable[Question], *, chunks: Iterable[Chunk]
) -> list[QuoteOccurrences]:
    """Return where every quote of `questions` occurs among `chunks`.

    Each quote is prepared once and each chunk's text normalised once, then every quote
    is looked for in every chunk. It trusts the evaluator's checks: question ids are
    unique (a second question with an id would replace the first's quotes), and each
    quote's `where` fits the chunks' schema.

    Args:
        questions: The questions whose quotes to look for; an unanswerable question has
            none.
        chunks: The corpus, e.g. `index.find(where=Everything())`.

    Returns:
        One record per quote, in question order, then key point, then quote.

    Raises:
        SchemaMismatchError: A quote's `where` names a field a chunk holding its text
            has no value for: it wasn't checked against the chunks' schema.

    Example:
        occurrences = find_quote_occurrences(questions, chunks=index.find(where=Everything()))
        broken = [found for found in occurrences if not found.is_found]
    """
    quote_matchers = {
        (question.id, key_point_pos, quote_pos): QuoteMatcher.from_quote(quote)
        for question in questions
        for key_point_pos, key_point in enumerate(question.key_points)
        for quote_pos, quote in enumerate(key_point.quotes)
    }
    occurrences: list[QuoteOccurrences] = []

    normalized_chunks = [(chunk, normalize_text(chunk.text)) for chunk in chunks]
    for (question_id, key_point, quote_position), matcher in quote_matchers.items():
        in_place: list[uuid.UUID] = []
        elsewhere: list[uuid.UUID] = []
        for chunk, normalized_text in normalized_chunks:
            # The text first: a chunk without it is in neither list, wherever it sits
            if matcher.text not in normalized_text:
                continue
            if matcher.chunk_matcher(chunk):
                in_place.append(chunk.id)
            else:
                elsewhere.append(chunk.id)
        occurrences.append(
            QuoteOccurrences(
                question_id=question_id,
                key_point=key_point,
                quote=quote_position,
                chunk_ids_in_place=tuple(in_place),
                chunk_ids_elsewhere=tuple(elsewhere),
            )
        )
    return occurrences
