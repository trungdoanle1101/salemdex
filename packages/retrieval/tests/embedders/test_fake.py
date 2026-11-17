import math
import unicodedata
import uuid

import pytest

from salemdex_retrieval.chunks import Chunk
from salemdex_retrieval.embedders.fake import FakeDenseEmbedder
from salemdex_retrieval.embedders.protocol import (
    DenseBinding,
    DenseEmbedder,
    embed_chunks,
)
from salemdex_retrieval.filters import Everything
from salemdex_retrieval.index import InMemoryIndex
from salemdex_retrieval.schema import One, Schema
from salemdex_retrieval.vectors import Dense


class Chunks(Schema):
    position = One(int)
    dense = Dense(size=16, distance="cosine")


TEXTS = [
    "Điều 1. Phạm vi điều chỉnh",
    "Điều 2. Đối tượng áp dụng",
    "Điều 3. Giải thích",
]


def test_the_same_text_gets_the_same_row_from_any_instance() -> None:
    first = FakeDenseEmbedder(size=16).embed_passages(TEXTS)
    second = FakeDenseEmbedder(size=16).embed_passages(TEXTS)

    assert first == second


def test_different_texts_get_different_rows() -> None:
    rows = FakeDenseEmbedder(size=16).embed_passages(TEXTS)

    assert len({tuple(row) for row in rows}) == len(TEXTS)


def test_each_row_has_the_given_size_and_unit_length() -> None:
    rows = FakeDenseEmbedder(size=16).embed_passages(TEXTS)

    for row in rows:
        assert len(row) == 16
        assert math.hypot(*row) == pytest.approx(1.0)


def test_a_question_gets_the_same_row_as_the_identical_passage() -> None:
    embedder = FakeDenseEmbedder(size=16)

    assert embedder.embed_queries(TEXTS) == embedder.embed_passages(TEXTS)


def test_no_texts_give_no_rows() -> None:
    embedder = FakeDenseEmbedder(size=16)

    assert embedder.embed_queries([]) == []
    assert embedder.embed_passages([]) == []


def test_a_question_finds_the_identical_passage_first() -> None:
    embedder: DenseEmbedder = FakeDenseEmbedder(size=16)
    chunks = [
        Chunk(id=uuid.uuid4(), text=text, field_values=[Chunks.position.value(index)])
        for index, text in enumerate(TEXTS)
    ]
    index = InMemoryIndex(schema=Chunks)
    index.put(
        embed_chunks(
            chunks, bindings=[DenseBinding(vector=Chunks.dense, embedder=embedder)]
        )
    )

    [question_row] = embedder.embed_queries([TEXTS[1]])
    hits = index.search(Chunks.dense.query(question_row), where=Everything(), limit=3)

    assert hits[0].chunk == chunks[1]
    assert hits[0].score == pytest.approx(1.0)


@pytest.mark.parametrize("size", [0, -1, True, 2.0])
def test_a_size_that_isnt_a_positive_integer_is_refused(size: object) -> None:
    with pytest.raises(ValueError, match="size"):
        FakeDenseEmbedder(size=size)  # type: ignore[arg-type]


def test_the_size_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        FakeDenseEmbedder(16)  # type: ignore[call-arg]


def dot(first: list[float], second: list[float]) -> float:
    """Return the dot product of two rows: their cosine, since rows have unit length."""
    return math.sumprod(first, second)


def test_texts_sharing_more_words_score_higher() -> None:
    embedder = FakeDenseEmbedder(size=256)
    [question] = embedder.embed_queries(["phạm vi điều chỉnh"])
    most, some, none = embedder.embed_passages(
        [
            "Điều 1. Phạm vi điều chỉnh của luật",
            "Điều 2. Đối tượng áp dụng",
            "Giải thích từ ngữ",
        ]
    )

    assert dot(question, most) > dot(question, some) > dot(question, none)


def test_words_add_with_either_sign() -> None:
    # Unsigned words would make every collision add similarity between unrelated texts
    [row] = FakeDenseEmbedder(size=256).embed_passages(
        ["điều phạm vi chỉnh đối tượng áp dụng giải thích từ ngữ"]
    )

    assert any(value > 0 for value in row)
    assert any(value < 0 for value in row)


def test_a_word_gets_the_same_row_in_either_unicode_form() -> None:
    composed = unicodedata.normalize("NFC", "Phạm vi điều chỉnh")
    decomposed = unicodedata.normalize("NFD", composed)
    embedder = FakeDenseEmbedder(size=256)

    assert composed != decomposed
    assert embedder.embed_passages([composed]) == embedder.embed_passages([decomposed])


def test_case_and_punctuation_dont_change_the_row() -> None:
    embedder = FakeDenseEmbedder(size=256)

    assert embedder.embed_passages(["ĐIỀU 1. Phạm vi!"]) == embedder.embed_passages(
        ["điều 1 phạm vi"]
    )


def test_a_text_without_words_gets_a_zero_row() -> None:
    [row] = FakeDenseEmbedder(size=16).embed_passages(["... !!"])

    assert row == [0.0] * 16
