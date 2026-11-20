import uuid
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class DocumentSearched:
    """The document a passage comes from, as search shows it for citing.

    Attributes:
        id: The document's id.
        title: Its title now, read from the database, not from the index.
        version: The version the passage's text comes from: the one search serves.
    """

    id: uuid.UUID
    title: str
    version: int


@dataclass(frozen=True, kw_only=True)
class Passage:
    """A piece of a document that search found for a question.

    Attributes:
        text: The document's own words.
        headings: Where the text sits, outermost first, for citing it; empty for a
            document without structure.
        document: The document it comes from.
    """

    text: str
    headings: tuple[str, ...]
    document: DocumentSearched
