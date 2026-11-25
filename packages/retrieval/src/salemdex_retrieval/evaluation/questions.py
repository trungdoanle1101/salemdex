"""Questions with known answers, for measuring how well a statement finds them.

A question needs all its key points; a key point needs any one of its quotes. A key
point is one thing a full answer says, a rule or its exception, say; its quotes are the
places stating it, alternatives to one another. (TREC calls key points nuggets.) A
question with no key points is one the documents can't answer, judged for abstention
rather than for ranks.

Answers are found by quote, never by article or section: a quote survives re-chunking
and a new outline, where a section label doesn't. Each quote says where it may sit with a
filter over the chunks' fields, so the app decides what a document is, and one question
may draw on several documents.

These are types, not a file format: the app's loader builds them from its own files.
They hold no location or other description of where an answer sits, since that would be
document text; a miss is explained by the ids of the chunks holding its quotes.
"""

from __future__ import annotations

from dataclasses import dataclass

from salemdex_retrieval.checks import to_stripped_str
from salemdex_retrieval.filters import Filter, check_is_filter


@dataclass(frozen=True, kw_only=True)
class Question:
    """A question, and the key points a full answer to it makes.

    Attributes:
        id: Names the question in runs, judgments and reports, e.g. "kt-07"; stripped
            of whitespace at either end. Unique within a question set.
        text: The question as a person would ask it; stripped of whitespace at either
            end.
        tags: The kinds of question it belongs to, e.g. "paraphrase", so a report can
            show which kinds a change helped; each stripped, no two alike, kept sorted
            so the order they're given in doesn't matter.
        key_points: Every key point the answer needs, all of them, in order; empty
            when the documents can't answer it.

    Example:
        Question(
            id="kt-07",
            text="…",
            tags=("several_places",),
            key_points=(
                KeyPoint(quotes=(Quote(text="…", where=Chunks.document_id.eq(doc)),)),
                KeyPoint(quotes=(Quote(text="…", where=Chunks.document_id.eq(doc)),)),
            ),
        )
    """

    id: str
    text: str
    tags: tuple[str, ...]
    key_points: tuple[KeyPoint, ...]

    def __post_init__(self) -> None:
        """Strip the id, the text and the tags, check them, and sort the tags.

        Raises:
            TypeError: The id, the text or a tag isn't a `str`.
            ValueError: The id, the text or a tag is empty or only whitespace, or two
                tags are alike once stripped.
        """
        # The dataclass is frozen; this is the one place its fields are set
        object.__setattr__(self, "id", to_stripped_str(self.id, name="id"))
        object.__setattr__(self, "text", to_stripped_str(self.text, name="text"))

        tags = sorted(
            to_stripped_str(tag, name=f"tags[{position}]")
            for position, tag in enumerate(self.tags)
        )

        seen_tags: set[str] = set()
        for tag in tags:
            if tag in seen_tags:
                raise ValueError(f"duplicate tag {tag!r}")
            seen_tags.add(tag)

        object.__setattr__(self, "tags", tuple(tags))

    @property
    def is_answerable(self) -> bool:
        """Whether the documents hold an answer: the question names some key point."""
        return bool(self.key_points)


@dataclass(frozen=True, kw_only=True)
class KeyPoint:
    """One thing a full answer says, and the quotes that would each supply it.

    The same point may be stated in several places, a regulation and its guidance, say:
    its quotes are alternatives, and a chunk holding any one of them supplies it.

    Attributes:
        quotes: The quotes supplying it; at least one.
    """

    quotes: tuple[Quote, ...]

    def __post_init__(self) -> None:
        """Check there's a quote.

        Raises:
            ValueError: There are no quotes, so the key point could never be found.
        """
        if not self.quotes:
            raise ValueError("KeyPoint must have at least one Quote")


@dataclass(frozen=True, kw_only=True)
class Quote:
    """A short verbatim passage of a document, and which chunks may hold it.

    A chunk holds the quote when it matches `where` and its text contains the quote's,
    compared after normalising both.

    Attributes:
        text: Words copied from the document, as a passage would show them; stripped
            of whitespace at either end.
        where: The chunks that may hold it, e.g. one document's; `Everything()` for
            anywhere, though a short quote may then match more chunks than meant.
    """

    text: str
    where: Filter

    def __post_init__(self) -> None:
        """Check the filter, and strip the text and check it.

        Raises:
            TypeError: `where` isn't a filter, or `text` isn't a `str`.
            ValueError: `text` is empty or only whitespace.
        """
        check_is_filter(self.where, owner="Quote")
        # The dataclass is frozen; this is the one place its text is set
        object.__setattr__(self, "text", to_stripped_str(self.text, name="text"))
