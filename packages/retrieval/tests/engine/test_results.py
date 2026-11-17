import uuid
from typing import Any

import pytest

from salemdex_retrieval.chunks import Chunk
from salemdex_retrieval.engine.results import (
    Candidate,
    Found,
    Fused,
    Passage,
    Reranked,
)

TEXT = "Hồ sơ nộp trong thời hạn 15 ngày."
CHUNK = Chunk(id=uuid.uuid4(), text=TEXT, field_values=[])
FOUND = Found(source="dense", rank=3, score=0.81)


# Trail entries


def test_a_found_entry_keeps_its_source_rank_and_score() -> None:
    assert (FOUND.source, FOUND.rank, FOUND.score) == ("dense", 3, 0.81)


@pytest.mark.parametrize(
    "rank",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(True, id="a bool"),
        pytest.param(1.0, id="a float"),
    ],
)
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda rank: Found(source="dense", rank=rank, score=0.5), id="found"
        ),
        pytest.param(lambda rank: Fused(rank=rank, score=0.5), id="fused"),
        pytest.param(lambda rank: Reranked(rank=rank, score=0.5), id="reranked"),
    ],
)
def test_every_entry_refuses_a_rank_that_isnt_a_positive_int(
    build: Any, rank: Any
) -> None:
    with pytest.raises(ValueError, match="rank must be a positive integer"):
        build(rank)


# Candidate


def test_a_candidate_refuses_an_empty_trail() -> None:
    with pytest.raises(ValueError, match="trail"):
        Candidate(chunk=CHUNK, trail=())


def test_a_candidate_just_found_scores_as_its_source_scored_it() -> None:
    assert Candidate(chunk=CHUNK, trail=(FOUND,)).score == 0.81


def test_following_adds_the_entry_to_a_new_candidate() -> None:
    found = Candidate(chunk=CHUNK, trail=(FOUND,))
    reranked = Reranked(rank=1, score=0.97)

    followed = found.followed_by(reranked)

    assert followed.trail == (FOUND, reranked)
    assert followed.chunk is CHUNK
    assert found.trail == (FOUND,)


def test_a_candidate_scores_as_the_last_stage_scored_it() -> None:
    candidate = (
        Candidate(chunk=CHUNK, trail=(FOUND,))
        .followed_by(Found(source="sparse", rank=1, score=12.4))
        .followed_by(Fused(rank=2, score=0.032))
    )

    assert candidate.score == 0.032


# Passage


def test_a_passage_refuses_an_empty_trail() -> None:
    with pytest.raises(ValueError, match="trail"):
        Passage(chunk=CHUNK, trail=())


def test_a_passage_from_a_candidate_keeps_its_chunk_trail_and_score() -> None:
    candidate = Candidate(chunk=CHUNK, trail=(FOUND,)).followed_by(
        Reranked(rank=1, score=0.97)
    )

    passage = Passage.from_candidate(candidate)

    assert type(passage) is Passage
    assert passage.chunk is CHUNK
    assert passage.trail == candidate.trail
    assert passage.score == 0.97


# Logging


def test_a_passage_and_a_candidate_never_show_the_text_in_repr() -> None:
    candidate = Candidate(chunk=CHUNK, trail=(FOUND, Fused(rank=1, score=0.03)))

    for result in [candidate, Passage.from_candidate(candidate)]:
        assert TEXT not in repr(result)
        assert "'dense'" in repr(result)
