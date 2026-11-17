"""Running a search statement: from a question to passages, with the guarantees applied.

The statement says what to do; the executor adds what no statement may leave out:

- the session's access filter, inside every source's search, so hidden chunks never
  leave the index;
- the validity gate, where candidates enter (after a retrieve) and again before
  `Take`, so only chunks that may be served now become passages.

The stages run in order, each turning the candidates so far into the next ones. They
are kept as a list of ranked lists: one per source after a retrieve, a single one after
a merge, which every later stage works on.
"""

import uuid
from collections import defaultdict
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from functools import partial
from typing import assert_never

from salemdex_retrieval.chunks import Chunk
from salemdex_retrieval.embedders.protocol import DenseBinding, VectorBinding
from salemdex_retrieval.engine.results import Candidate, Found, Fused, Passage, Reranked
from salemdex_retrieval.engine.stages import (
    Fusion,
    Merge,
    Rerank,
    Retrieve,
    Rrf,
    Source,
    Take,
)
from salemdex_retrieval.engine.statements import Search, replay
from salemdex_retrieval.engine.validity import Validity
from salemdex_retrieval.filters import Filter
from salemdex_retrieval.rerankers.protocol import Reranker, score_texts
from salemdex_retrieval.vectors import VectorQuery


def execute(statement: Search, *, access: Filter, validity: Validity) -> list[Passage]:
    """Run `statement` and return its passages, best first.

    The statement is replayed first, so one built at run time is checked before any
    model is called or any index is read, and the replayed one is what runs.

    Args:
        statement: The search to run.
        access: What the asker may see, added to every source's search.
        validity: Which chunks may be served now, asked at every gate.

    Returns:
        At most the statement's `Take` limit of passages, each with its trail.

    Raises:
        StatementError: The stages come in an order the steps don't allow.
        ValueError: An embedder or the reranker returned the wrong number of rows or
            scores, or a number that doesn't fit.
        TypeError: A score or a vector number isn't a real number.
    """
    statement = replay(statement)
    lists: list[list[Candidate]] = []
    for stage in statement.stages:
        match stage:
            case Retrieve(sources=sources):
                lists = _retrieve(
                    statement.question, sources, access=access, validity=validity
                )
            case Merge(fusion=fusion):
                lists = [_fuse(lists, fusion=fusion)]
            case Rerank(reranker=reranker):
                lists = _rerank(lists, question=statement.question, reranker=reranker)
            case Take(limit=limit):
                [candidates] = lists
                return _take(candidates, validity=validity, limit=limit)
            case _:
                assert_never(stage)
    raise AssertionError("replay guarantees a Take ends every Search statement")


def _retrieve(
    question: str,
    sources: Sequence[Source],
    *,
    access: Filter,
    validity: Validity,
) -> list[list[Candidate]]:
    """Return one ranked list of valid candidates per source, each found by it.

    Every source searches with the access filter; the chunks of all lists are checked
    in one call, then each list keeps its valid ones. A candidate's `Found` rank is its
    place in what the source returned, so a dropped chunk doesn't move the others up.

    Raises:
        ValueError: An embedder returned other than one row for the question.
    """
    hits_by_source = [
        (
            source,
            source.index.search(
                _embed_question(question, binding=source.binding),
                where=access,
                limit=source.depth,
            ),
        )
        for source in sources
    ]
    valid_ids = _valid_ids(
        [hit.chunk for _, hits in hits_by_source for hit in hits], validity=validity
    )
    return [
        [
            Candidate(
                chunk=hit.chunk,
                trail=(Found(source=source.name, rank=rank, score=hit.score),),
            )
            for rank, hit in enumerate(hits, start=1)
            if hit.chunk.id in valid_ids
        ]
        for source, hits in hits_by_source
    ]


def _embed_question(question: str, *, binding: VectorBinding) -> VectorQuery:
    """Return the query searching `binding`'s vector for `question`.

    Raises:
        ValueError: The embedder returned other than one row, or the row doesn't fit
            the vector.
    """
    match binding:
        case DenseBinding(vector=vector, embedder=embedder):
            rows = embedder.embed_queries([question])
            if len(rows) != 1:
                raise ValueError(
                    f"the embedder for vector {vector.name!r} "
                    f"returned {len(rows)} rows for one question"
                )
            return vector.query(rows[0])
        case _:
            assert_never(binding)


def _fuse(lists: list[list[Candidate]], *, fusion: Fusion) -> list[Candidate]:
    """Return the lists fused into one, best first, each candidate's trail joined.

    Candidates are the same chunk when their ids are (a retrieve searches one index).
    A fused candidate's trail holds the entries of every list that has it, in list
    order, then its `Fused` entry. Ties keep the order of first appearance.
    """
    groups: dict[uuid.UUID, list[Candidate]] = defaultdict(list)
    for candidates in lists:
        for candidate in candidates:
            groups[candidate.chunk.id].append(candidate)

    match fusion:
        case Rrf(k=k):
            scorer = partial(_rrf, k=k)
        case _:
            assert_never(fusion)

    scores = {chunk_id: scorer(group) for chunk_id, group in groups.items()}

    sorted_groups = sorted(
        groups.values(), key=lambda group: scores[group[0].chunk.id], reverse=True
    )

    return [
        Candidate(
            chunk=group[0].chunk,
            trail=tuple(entry for candidate in group for entry in candidate.trail),
        ).followed_by(Fused(rank=rank, score=scores[group[0].chunk.id]))
        for rank, group in enumerate(sorted_groups, start=1)
    ]


def _rrf(group: list[Candidate], *, k: int) -> float:
    """Return the RRF score of one chunk's candidates, one from each list that has it.

    Each counts its current rank in its own list; a list without the chunk counts 0.
    """
    return sum(1 / (k + candidate.trail[-1].rank) for candidate in group)


def _rerank(
    lists: list[list[Candidate]], *, question: str, reranker: Reranker
) -> list[list[Candidate]]:
    """Return each list scored by `reranker` and sorted by its scores, best first.

    The reranker reads each chunk's `embedded_text`, the text the index embedded,
    context included. Ties keep their earlier order.

    Raises:
        ValueError: The reranker returned a score count other than the texts', or a
            score that isn't finite.
        TypeError: A score isn't a real number.
    """
    result: list[list[Candidate]] = []
    for candidates in lists:
        texts = [candidate.chunk.embedded_text for candidate in candidates]
        scores = score_texts(question, texts, reranker=reranker)

        reranked = sorted(
            zip(scores, candidates, strict=True), key=lambda pair: pair[0], reverse=True
        )
        result.append(
            [
                candidate.followed_by(Reranked(rank=rank, score=score))
                for rank, (score, candidate) in enumerate(reranked, start=1)
            ]
        )

    return result


def _take(
    candidates: list[Candidate], *, validity: Validity, limit: int
) -> list[Passage]:
    """Return the best `limit` candidates that pass the final gate, as passages.

    The gate runs before the cut, so the limit still fills when it can; this is the
    one place passages are built.
    """
    valid_ids = _valid_ids(
        [candidate.chunk for candidate in candidates], validity=validity
    )
    valid_candidates = [
        candidate for candidate in candidates if candidate.chunk.id in valid_ids
    ]
    return [Passage.from_candidate(candidate) for candidate in valid_candidates[:limit]]


def _valid_ids(
    chunks: Sequence[Chunk], *, validity: Validity
) -> AbstractSet[uuid.UUID]:
    """Return the ids of `chunks` that may be served, asking once, and only if needed."""
    # One chunk found by several sources is asked about once
    unique = list({chunk.id: chunk for chunk in chunks}.values())
    if not unique:
        return frozenset()
    return validity.valid_ids(unique)
