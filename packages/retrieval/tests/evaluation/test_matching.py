import unicodedata
import uuid

import pytest

from salemdex_retrieval.chunks import Chunk
from salemdex_retrieval.evaluation.matching import QuoteMatcher, normalize_text
from salemdex_retrieval.evaluation.questions import Quote
from salemdex_retrieval.filters import Everything
from salemdex_retrieval.schema import One, Schema

# Invented text only: the real question sets are restricted and never appear in tests


class Notes(Schema):
    document = One(str, indexed=True)


IN_HANDBOOK = Notes.document.eq("handbook")


def chunk(text: str, *, document: str = "handbook") -> Chunk:
    return Chunk(
        id=uuid.uuid4(), text=text, field_values=[Notes.document.value(document)]
    )


# QuoteMatcher


def test_a_quote_matcher_holds_the_quotes_text_normalized() -> None:
    matcher = QuoteMatcher.from_quote(Quote(text="Staff  MAY work", where=Everything()))

    assert matcher.text == "staff may work"


def test_a_quote_matcher_needs_both_the_place_and_the_text() -> None:
    matcher = QuoteMatcher.from_quote(Quote(text="staff may work", where=IN_HANDBOOK))
    text = normalize_text("Staff may work.")

    assert matcher.found_in_chunk(chunk("Staff may work."), normalized_text=text)
    assert not matcher.found_in_chunk(
        chunk("Staff may work.", document="guide"), normalized_text=text
    )
    assert not matcher.found_in_chunk(
        chunk("Staff may rest."), normalized_text=normalize_text("Staff may rest.")
    )


# normalize_text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("Staff May", "staff may", id="case-folded"),
        pytest.param("  staff \n\t may  ", "staff may", id="whitespace collapsed"),
        pytest.param("staff may", "staff may", id="a non-breaking space"),
        pytest.param("phép", "phép", id="composed (NFC)"),
        pytest.param("Được", "được", id="Vietnamese diacritics kept"),
        pytest.param("Straße", "strasse", id="ß folded to ss"),
    ],
)
def test_normalize_text(text: str, expected: str) -> None:
    assert normalize_text(text) == expected


def test_normalize_text_recomposes_after_case_folding() -> None:
    # Folding "İ" gives "i" and a combining dot; the result is NFC again
    folded = normalize_text("İ")

    assert folded == unicodedata.normalize("NFC", folded)


def test_normalize_text_leaves_punctuation() -> None:
    assert normalize_text("work, from home.") == "work, from home."
