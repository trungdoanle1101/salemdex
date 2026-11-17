import dataclasses
import re
import uuid
from collections.abc import Sequence
from typing import Any

import pytest

from salemdex_retrieval.chunks import Chunk
from salemdex_retrieval.embedders.fake import FakeDenseEmbedder
from salemdex_retrieval.embedders.protocol import DenseBinding, embed_chunks
from salemdex_retrieval.engine.session import Engine
from salemdex_retrieval.engine.stages import Source
from salemdex_retrieval.engine.statements import Search, search
from salemdex_retrieval.engine.validity import AlwaysValid
from salemdex_retrieval.filters import Everything
from salemdex_retrieval.index.in_memory import InMemoryIndex
from salemdex_retrieval.schema import Many, Schema
from salemdex_retrieval.vectors import Dense

SIZE = 16
HR = uuid.UUID("00000000-0000-0000-0000-00000000000a")
LEGAL = uuid.UUID("00000000-0000-0000-0000-00000000000b")


class Chunks(Schema):
    groups = Many(uuid.UUID, indexed=True)
    dense = Dense(size=SIZE, distance="cosine")


BINDING = DenseBinding(vector=Chunks.dense, embedder=FakeDenseEmbedder(size=SIZE))


def build_chunk(text: str, *, groups: Sequence[uuid.UUID]) -> Chunk:
    """Return a chunk of Chunks readable by `groups`."""
    return Chunk(id=uuid.uuid4(), text=text, field_values=[Chunks.groups.value(groups)])


def statement_over(*chunks: Chunk) -> Search:
    """Return a statement searching an index holding `chunks`, keeping all of them."""
    index = InMemoryIndex(schema=Chunks)
    index.put(embed_chunks(chunks, bindings=[BINDING]))
    source = Source(name="dense", index=index, binding=BINDING, depth=10)
    return search("phạm vi điều chỉnh").retrieve(source).take(10)


def ids(passages: Sequence[Any]) -> set[uuid.UUID]:
    """Return the chunk id of each passage."""
    return {passage.chunk.id for passage in passages}


class Dropping:
    """A validity check keeping every chunk but the given ones."""

    def __init__(self, *dropped: Chunk) -> None:
        self.dropped = {chunk.id for chunk in dropped}

    def valid_ids(self, chunks: Sequence[Chunk]) -> set[uuid.UUID]:
        return {chunk.id for chunk in chunks} - self.dropped


HR_ONLY = build_chunk("Phạm vi điều chỉnh của quy chế nhân sự", groups=[HR])
LEGAL_ONLY = build_chunk("Phạm vi điều chỉnh của quy chế pháp chế", groups=[LEGAL])


# Access: the session's, in every search it runs


def test_a_session_sees_only_what_its_access_filter_allows() -> None:
    statement = statement_over(HR_ONLY, LEGAL_ONLY)
    session = Engine(validity=AlwaysValid()).session(access=Chunks.groups.any_of([HR]))

    assert ids(session.execute(statement)) == {HR_ONLY.id}


def test_two_sessions_of_one_engine_keep_their_own_access() -> None:
    statement = statement_over(HR_ONLY, LEGAL_ONLY)
    engine = Engine(validity=AlwaysValid())
    hr = engine.session(access=Chunks.groups.any_of([HR]))
    legal = engine.session(access=Chunks.groups.any_of([LEGAL]))

    assert ids(hr.execute(statement)) == {HR_ONLY.id}
    assert ids(legal.execute(statement)) == {LEGAL_ONLY.id}
    assert ids(hr.execute(statement)) == {HR_ONLY.id}


def test_a_session_with_everything_as_access_sees_every_chunk() -> None:
    statement = statement_over(HR_ONLY, LEGAL_ONLY)
    session = Engine(validity=AlwaysValid()).session(access=Everything())

    assert ids(session.execute(statement)) == {HR_ONLY.id, LEGAL_ONLY.id}


@pytest.mark.parametrize(
    ("wrong", "type_name"),
    [
        # What comparing a field with == gives, since fields don't define __eq__
        pytest.param(
            Chunks.groups == [HR],  # type: ignore[comparison-overlap]
            "bool",
            id="a comparison",
        ),
        pytest.param(None, "NoneType", id="None"),
        pytest.param("groups any of hr", "str", id="a string"),
    ],
)
def test_opening_a_session_with_access_that_is_not_a_filter_is_refused(
    wrong: Any, type_name: str
) -> None:
    engine = Engine(validity=AlwaysValid())

    with pytest.raises(
        TypeError, match=re.escape(f"Session: {type_name} is not a valid Filter")
    ):
        engine.session(access=wrong)


def test_a_session_cannot_change_its_access() -> None:
    session = Engine(validity=AlwaysValid()).session(access=Chunks.groups.any_of([HR]))

    with pytest.raises(dataclasses.FrozenInstanceError):
        session.access = Everything()  # type: ignore[misc]


# Validity: the engine's, in every session


def test_every_session_runs_its_engines_validity_check() -> None:
    statement = statement_over(HR_ONLY, LEGAL_ONLY)
    engine = Engine(validity=Dropping(LEGAL_ONLY))

    everyone = engine.session(access=Everything())
    legal = engine.session(access=Chunks.groups.any_of([LEGAL]))

    assert ids(everyone.execute(statement)) == {HR_ONLY.id}
    assert ids(legal.execute(statement)) == set()


def test_an_engine_cannot_change_its_validity_check() -> None:
    engine = Engine(validity=Dropping(LEGAL_ONLY))

    with pytest.raises(dataclasses.FrozenInstanceError):
        engine.validity = AlwaysValid()  # type: ignore[misc]
