import dataclasses
import uuid
from typing import Any

import pytest

from salemdex_retrieval.chunks import Chunk, EmbeddedChunk
from salemdex_retrieval.errors import SchemaMismatchError
from salemdex_retrieval.schema import FieldValue, Many, One, Schema
from salemdex_retrieval.vectors import Dense


class Chunks(Schema):
    document_id = One(uuid.UUID, indexed=True)
    position = One(int)
    headings = Many(str)


class Entries(Schema):
    document_id = One(uuid.UUID)


CHUNK_ID = uuid.UUID("00000000-0000-0000-0000-0000000000c1")
DOCUMENT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
TEXT = "Scope of application"


def build(**parts: Any) -> Chunk:
    """Return a chunk with every field of Chunks, overriding any part given."""
    defaults: dict[str, Any] = {
        "id": CHUNK_ID,
        "text": TEXT,
        "embedded_text": None,
        "field_values": [
            Chunks.document_id.value(DOCUMENT_ID),
            Chunks.position.value(3),
            Chunks.headings.value(["Part 1"]),
        ],
    }
    return Chunk(**(defaults | parts))


def test_a_chunk_keeps_its_id_and_text() -> None:
    chunk = build()

    assert chunk.id == CHUNK_ID
    assert chunk.text == TEXT


def test_the_embedded_text_is_the_text_unless_one_is_given() -> None:
    assert Chunk(id=CHUNK_ID, text=TEXT, field_values=[]).embedded_text == TEXT
    assert build(embedded_text=None).embedded_text == TEXT


def test_a_given_embedded_text_is_kept() -> None:
    chunk = build(embedded_text=f"Part 1\n{TEXT}")

    assert chunk.embedded_text == f"Part 1\n{TEXT}"
    assert chunk.text == TEXT


def test_a_chunk_takes_its_parts_by_keyword_only() -> None:
    with pytest.raises(TypeError):
        Chunk(CHUNK_ID, TEXT, None, [])  # type: ignore[call-arg]


def test_a_chunk_keys_its_field_values_by_field() -> None:
    chunk = build()

    assert dict(chunk.field_values) == {
        Chunks.document_id: DOCUMENT_ID,
        Chunks.position: 3,
        Chunks.headings: ("Part 1",),
    }


def test_an_empty_many_is_kept_as_no_values() -> None:
    chunk = build(field_values=[Chunks.headings.value([])])

    assert chunk.field_values[Chunks.headings] == ()


def test_the_same_field_given_twice_is_rejected() -> None:
    with pytest.raises(ValueError, match="position"):
        build(field_values=[Chunks.position.value(1), Chunks.position.value(2)])


def test_two_fields_with_one_name_are_rejected() -> None:
    # No schema has two fields with one name, so the chunk can't be valid anywhere.
    with pytest.raises(ValueError, match="document_id"):
        build(
            field_values=[
                Chunks.document_id.value(DOCUMENT_ID),
                Entries.document_id.value(DOCUMENT_ID),
            ]
        )


def test_a_chunk_rejects_an_id_that_is_not_a_uuid() -> None:
    with pytest.raises(TypeError, match="UUID"):
        build(id=str(CHUNK_ID))


def test_a_chunk_keeps_its_own_copy_of_the_field_values() -> None:
    values: list[FieldValue] = [Chunks.position.value(3)]
    chunk = build(field_values=values)

    values.append(Chunks.headings.value(["Part 1"]))

    assert list(chunk.field_values) == [Chunks.position]


def test_the_field_values_of_a_chunk_cannot_be_changed() -> None:
    chunk = build()

    with pytest.raises(TypeError):
        chunk.field_values[Chunks.position] = 4  # type: ignore[index]


def test_a_chunk_cannot_be_changed_after_it_is_built() -> None:
    chunk = build()

    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.text = "Other"  # type: ignore[misc]


def test_chunks_with_the_same_parts_are_equal() -> None:
    assert build() == build()
    assert build() != build(text="Other")


def test_a_chunk_does_not_show_its_text_in_its_repr() -> None:
    # Document text must never reach logs or error messages.
    shown = repr(build(embedded_text=f"Part 1\n{TEXT}"))

    assert TEXT not in shown
    assert str(CHUNK_ID) in shown


class Standard(Schema):
    document_id = One(uuid.UUID)


class Extended(Standard):
    position = One(int)


def test_a_chunk_with_a_value_for_every_field_fits() -> None:
    build().check_fits(Chunks)


def test_an_empty_many_counts_as_a_value() -> None:
    chunk = build(
        field_values=[
            Chunks.document_id.value(DOCUMENT_ID),
            Chunks.position.value(3),
            Chunks.headings.value([]),
        ]
    )

    chunk.check_fits(Chunks)


def test_a_field_from_a_preset_fits_the_schema_that_inherits_it() -> None:
    chunk = build(
        field_values=[
            Standard.document_id.value(DOCUMENT_ID),
            Extended.position.value(3),
        ]
    )

    chunk.check_fits(Extended)


def test_a_chunk_missing_a_field_does_not_fit_and_names_it_with_the_chunk_id() -> None:
    chunk = build(
        field_values=[Chunks.document_id.value(DOCUMENT_ID), Chunks.position.value(3)]
    )

    with pytest.raises(SchemaMismatchError, match="headings") as caught:
        chunk.check_fits(Chunks)

    assert str(CHUNK_ID) in str(caught.value)


def test_a_field_the_schema_does_not_have_does_not_fit() -> None:
    chunk = build(
        field_values=[
            Standard.document_id.value(DOCUMENT_ID),
            Extended.position.value(3),
        ]
    )

    with pytest.raises(SchemaMismatchError, match="position"):
        chunk.check_fits(Standard)


def test_a_field_from_another_schema_with_the_same_name_does_not_fit() -> None:
    chunk = build(field_values=[Entries.document_id.value(DOCUMENT_ID)])

    with pytest.raises(SchemaMismatchError, match="document_id"):
        chunk.check_fits(Standard)


def test_every_missing_field_is_named_at_once_in_schema_order() -> None:
    chunk = build(field_values=[Chunks.position.value(3)])

    with pytest.raises(SchemaMismatchError, match="missing: document_id, headings$"):
        chunk.check_fits(Chunks)


def test_extra_and_missing_fields_are_named_in_one_error() -> None:
    chunk = build(
        field_values=[
            Entries.document_id.value(DOCUMENT_ID),
            Extended.position.value(3),
        ]
    )

    with pytest.raises(SchemaMismatchError) as caught:
        chunk.check_fits(Standard)

    assert str(caught.value) == (
        f"chunk {CHUNK_ID}: "
        "fields not in Standard: document_id (another schema's field), position; "
        "fields missing: document_id"
    )


def test_a_mismatch_is_a_value_error() -> None:
    assert issubclass(SchemaMismatchError, ValueError)


def test_a_mismatch_does_not_show_the_text() -> None:
    chunk = build(field_values=[])

    with pytest.raises(SchemaMismatchError) as caught:
        chunk.check_fits(Chunks)

    assert TEXT not in str(caught.value)


# Embedded chunks


class Embedded(Schema):
    document_id = One(uuid.UUID)
    dense = Dense(size=2, distance="cosine")
    sentence = Dense(size=3, distance="dot")


class OtherEmbedded(Schema):
    dense = Dense(size=2, distance="cosine")


def plain_chunk(schema: type[Schema] = Embedded) -> Chunk:
    return Chunk(
        id=CHUNK_ID,
        text=TEXT,
        field_values=[Embedded.document_id.value(DOCUMENT_ID)]
        if schema is Embedded
        else [],
    )


def embedded(**parts: Any) -> EmbeddedChunk:
    """Return an embedded chunk with every field and vector of Embedded."""
    defaults: dict[str, Any] = {
        "chunk": plain_chunk(),
        "vector_values": [
            Embedded.dense.value([0.6, 0.8]),
            Embedded.sentence.value([1.0, 0.0, 0.0]),
        ],
    }
    return EmbeddedChunk(**(defaults | parts))


def test_an_embedded_chunk_keeps_its_chunk_and_keys_its_values_by_vector() -> None:
    built = embedded()

    assert built.chunk == plain_chunk()
    assert built.vector_values == {
        Embedded.dense: Embedded.dense.value([0.6, 0.8]),
        Embedded.sentence: Embedded.sentence.value([1.0, 0.0, 0.0]),
    }


def test_an_embedded_chunk_takes_its_parts_by_keyword_only() -> None:
    with pytest.raises(TypeError):
        EmbeddedChunk(plain_chunk(), [])  # type: ignore[call-arg]


def test_the_same_vector_given_twice_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate vector: dense"):
        embedded(
            vector_values=[
                Embedded.dense.value([0.6, 0.8]),
                Embedded.dense.value([0.8, 0.6]),
            ]
        )


def test_two_vectors_with_one_name_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate vector: dense"):
        embedded(
            vector_values=[
                Embedded.dense.value([0.6, 0.8]),
                OtherEmbedded.dense.value([0.6, 0.8]),
            ]
        )


def test_the_vector_values_of_an_embedded_chunk_cannot_be_changed() -> None:
    built = embedded()

    with pytest.raises(TypeError):
        built.vector_values[Embedded.dense] = Embedded.dense.value([1.0, 0.0])  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        built.chunk = plain_chunk()  # type: ignore[misc]


def test_an_embedded_chunk_does_not_show_its_text_or_numbers_in_its_repr() -> None:
    shown = repr(embedded())

    assert TEXT not in shown
    assert "0.6" not in shown
    assert "dense" in shown


def test_an_embedded_chunk_with_every_field_and_vector_fits() -> None:
    embedded().check_fits(Embedded)


def test_a_chunk_of_a_schema_without_vectors_fits_with_no_vector_values() -> None:
    EmbeddedChunk(chunk=build(), vector_values=[]).check_fits(Chunks)


def test_an_embedded_chunk_missing_a_vector_does_not_fit_naming_it() -> None:
    built = embedded(vector_values=[Embedded.dense.value([0.6, 0.8])])

    with pytest.raises(SchemaMismatchError, match="vectors missing: sentence$"):
        built.check_fits(Embedded)


def test_a_vector_from_another_schema_with_the_same_name_does_not_fit() -> None:
    built = embedded(
        vector_values=[
            OtherEmbedded.dense.value([0.6, 0.8]),
            Embedded.sentence.value([1.0, 0.0, 0.0]),
        ]
    )

    with pytest.raises(SchemaMismatchError, match=r"dense \(another schema's vector\)"):
        built.check_fits(Embedded)


def test_field_and_vector_problems_are_named_in_one_error() -> None:
    built = EmbeddedChunk(
        chunk=Chunk(id=CHUNK_ID, text=TEXT, field_values=[]),
        vector_values=[OtherEmbedded.dense.value([0.6, 0.8])],
    )

    with pytest.raises(SchemaMismatchError) as caught:
        built.check_fits(Embedded)

    assert str(caught.value) == (
        f"embedded chunk {CHUNK_ID}: "
        "fields missing: document_id; "
        "vectors not in Embedded: dense (another schema's vector); "
        "vectors missing: dense, sentence"
    )
