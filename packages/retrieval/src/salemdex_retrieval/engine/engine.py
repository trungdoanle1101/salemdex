"""The engine and its prepared statements: the one way to run a search statement.

The app builds one `Engine` for its whole life, registering under a name each index
a statement may search, with the scans it may run on it, and each reranker, with the
validity check every run applies. `Engine.prepare` checks a statement against them
once, and returns a `PreparedStatement`, which runs it for any question:

    engine = Engine(
        validity=validity,
        indexes={"chunks": SearchableIndex(index=index, scans={"dense": dense})},
        rerankers={"bge": bge},
    )
    prepared = engine.prepare(
        search("chunks").retrieve(scan("dense", depth=50)).rerank("bge").take()
    )
    passages = prepared.execute(question, limit=10, where=where)

Each run is given what changes from one search to the next: the question, how many
passages, and a filter (`where`) applied inside every read of the index, before
ranking. The filter is required (`Everything()` matches every chunk), so no search
runs unfiltered by accident. Validity applies at every gate: where candidates enter,
after each retrieve, and again before `Take`; a window's chunks pass both too.

What's registered is private and read-only, and only `prepare` builds a prepared
statement, so a statement runs only with what the app registered, and never
unchecked. Like a prepared statement in SQL, with the slots fixed: the question, the
limit and the filter.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence, Sized
from collections.abc import Set as AbstractSet
from dataclasses import InitVar, dataclass, field
from functools import partial
from types import MappingProxyType
from typing import assert_never

from salemdex_retrieval.checks import check_positive_int, to_stripped_str
from salemdex_retrieval.chunks import Chunk, Hit
from salemdex_retrieval.embedders import DenseBinding, SparseBinding, VectorBinding
from salemdex_retrieval.engine.results import Candidate, Found, Fused, Passage, Reranked
from salemdex_retrieval.engine.sources import (
    LookupScan,
    RegisteredScan,
    SearchableIndex,
    VectorScan,
)
from salemdex_retrieval.engine.stages import (
    Fusion,
    Merge,
    Rerank,
    Retrieve,
    Rrf,
    Scan,
    Stage,
    Take,
    Window,
)
from salemdex_retrieval.engine.statements import Statement, replay
from salemdex_retrieval.engine.validity import Validity
from salemdex_retrieval.errors import StatementError
from salemdex_retrieval.filters import Filter, and_, check_is_filter, or_
from salemdex_retrieval.index import Index
from salemdex_retrieval.index.protocol import check_filter_matches_schema
from salemdex_retrieval.rerankers import Reranker
from salemdex_retrieval.rerankers.protocol import score_texts
from salemdex_retrieval.vectors import Vector, VectorQuery

# Every lookup hit's score: a chunk matches the parsed filter or it doesn't
_LOOKUP_SCORE = 1.0


@dataclass(frozen=True, kw_only=True, eq=False)
class Engine:
    """A search engine: the indexes, rerankers and validity check statements run with.

    Built once, for the whole life of the app. It's frozen and keeps read-only copies
    of the mappings it's given, so nothing registered can change under a statement
    already prepared. An app with nothing to check passes `AlwaysValid()`, which says
    so explicitly.

    Example:
        engine = Engine(
            validity=AlwaysValid(),
            indexes={
                "chunks": SearchableIndex(
                    index=index, scans={"dense": VectorScan(binding=binding)}
                ),
            },
            rerankers={"bge": reranker},
        )

    Args:
        validity: Which chunks may be served now, asked at every gate.
        indexes: What each index name a statement may use stands for: the index,
            with the scans a statement may run on it.
        rerankers: The reranker each name a statement may use stands for.
    """

    validity: InitVar[Validity]
    indexes: InitVar[Mapping[str, SearchableIndex]]
    rerankers: InitVar[Mapping[str, Reranker]]

    _validity: Validity = field(init=False, repr=False)
    _indexes: Mapping[str, SearchableIndex] = field(init=False, repr=False)
    _rerankers: Mapping[str, Reranker] = field(init=False, repr=False)

    def __post_init__(
        self,
        validity: Validity,
        indexes: Mapping[str, SearchableIndex],
        rerankers: Mapping[str, Reranker],
    ) -> None:
        """Check the registered names, and keep read-only copies of what's registered.

        Names must be given as statements hold them, stripped, or no statement could
        name them.

        Raises:
            TypeError: A name isn't a `str`.
            ValueError: A name is empty, or has whitespace at either end.
        """
        for name in indexes:
            if name != to_stripped_str(name, name="index's name"):
                raise ValueError(
                    f"index's name must be stripped and not empty, got {name!r}"
                )

        for name in rerankers:
            if name != to_stripped_str(name, name="reranker's name"):
                raise ValueError(
                    f"reranker's name must be stripped and not empty, got {name!r}"
                )

        object.__setattr__(self, "_validity", validity)
        object.__setattr__(self, "_indexes", MappingProxyType(dict(indexes)))
        object.__setattr__(self, "_rerankers", MappingProxyType(dict(rerankers)))

    def prepare(self, statement: Statement) -> PreparedStatement:
        """Return `statement` checked against this engine, ready to run for any question.

        Checked once, here, so a wrong statement fails before any model is called or
        any index is read, and each run checks only its own inputs. The statement is
        replayed first, so one built at run time is checked like one built by the
        steps, and the replayed one is what runs.

        Args:
            statement: The statement to prepare, however it was built.

        Returns:
            The prepared statement.

        Raises:
            StatementError: The stages come in an order the steps don't allow; the
                index isn't registered, a scan isn't registered on it, or a reranker
                isn't registered; or two scans of a retrieve search one vector.
            SchemaMismatchError: A window's field isn't one of the index's schema.
            ValueError: The index's name is empty, or a stage's settings are wrong
                (see `replay`).
            TypeError: A name isn't a `str`, or a scan isn't a `Scan`.
        """
        statement = replay(statement)
        if statement.index not in self._indexes:
            raise StatementError(
                f"index {statement.index!r} was not registered in the engine; "
                f"registered: {_names(self._indexes)}"
            )
        for stage in statement.stages:
            self._check_stage(
                stage,
                searchable_index=self._indexes[statement.index],
                index_name=statement.index,
            )
        return PreparedStatement._create(statement, engine=self)

    def _check_stage(
        self, stage: Stage, *, searchable_index: SearchableIndex, index_name: str
    ) -> None:
        """Check that `stage` names only what's registered, and fits what it names.

        Raises:
            StatementError: A scan or a reranker isn't registered, or a retrieve's
                scans search one vector twice.
            SchemaMismatchError: A window's field isn't one of the index's schema.
        """
        match stage:
            case Retrieve(scans=scans):
                seen_vectors: set[Vector] = set()
                for scan in scans:
                    if scan.name not in searchable_index.scans:
                        raise StatementError(
                            f"scan {scan.name!r} was not registered on index "
                            f"{index_name!r}; registered there: "
                            f"{_names(searchable_index.scans)}"
                        )
                    registered_scan = searchable_index.scans[scan.name]

                    match registered_scan:
                        case VectorScan(binding=binding):
                            vector = binding.vector
                            if vector in seen_vectors:
                                raise StatementError(
                                    "two scans in the same Retrieve stage search "
                                    f"vector {vector.name!r}; give it once, "
                                    "with the depth you need"
                                )
                            seen_vectors.add(vector)
                        case LookupScan():
                            pass
                        case _:
                            assert_never(registered_scan)
            case Rerank(reranker=name):
                if name not in self._rerankers:
                    raise StatementError(
                        f"reranker {name!r} was not registered in the engine; "
                        f"registered: {_names(self._rerankers)}"
                    )

            case Merge():
                pass
            case Take(window=window):
                if window is not None:
                    schema = searchable_index.index.schema
                    for declared_field in (window.order_by, *window.partition_by):
                        schema.check_declares(declared_field)
            case _:
                assert_never(stage)


class PreparedStatement:
    """A statement an engine checked, run for any question with `execute`.

    Only `Engine.prepare` builds one: calling the class raises `TypeError`, so holding
    one proves the checks ran. For the same reason it can't be changed: a statement
    swapped in later would run unchecked. Prepare once, when the app starts, and run it
    for every search.

    A plain class, not a dataclass: a dataclass's `__init__` is what's refused here,
    and `dataclasses.replace` would build a changed copy without the checks.

    Attributes:
        statement: The statement it runs, as replayed when it was prepared.
    """

    __slots__ = ("_engine", "_statement")

    _statement: Statement
    _engine: Engine

    def __init__(self) -> None:
        """Refuse to build one: `Engine.prepare` is the only way.

        Raises:
            TypeError: Always.
        """
        raise TypeError("build a PreparedStatement with engine.prepare(statement)")

    @property
    def statement(self) -> Statement:
        """The statement it runs, as replayed when it was prepared."""
        # A property, so mypy refuses an assignment as `__setattr__` does at run time
        return self._statement

    def execute(self, question: str, *, limit: int, where: Filter) -> list[Passage]:
        """Run the statement for `question`, and return its best passages, best first.

        The inputs are checked before any model is called or any index is read.

        Args:
            question: The question, as asked; stripped of whitespace at either end.
            limit: The most passages to return; a window adds chunks to each passage,
                never more passages.
            where: Which chunks may be found, applied inside every read of the index,
                windows included, before ranking; `Everything()` for every chunk. It
                must fit the schema of the statement's index.

        Returns:
            At most `limit` passages, each with its window and trail.

        Raises:
            TypeError: `question` isn't a `str`, `where` isn't a filter, a score or
                a vector number isn't a real number, or a lookup's parser returned
                something other than a filter or `None`.
            ValueError: `question` is empty or only whitespace, or `limit` isn't a
                positive `int`; or an embedder or a reranker returned the wrong number
                of rows or scores, or a number that doesn't fit.
            SchemaMismatchError: `where` uses a field that isn't in the schema of the
                statement's index, or a lookup's parsed filter does. An error from a
                lookup's parser passes through.
        """
        question = to_stripped_str(question, name="question")
        check_positive_int(limit, name="limit")
        self._check_filter(where)
        lists: list[list[Candidate]] = []
        searchable_index = self._engine._indexes[self._statement.index]
        for stage in self.statement.stages:
            match stage:
                case Retrieve(scans=scans):
                    lists = _retrieve(
                        question,
                        scans,
                        where=where,
                        searchable_index=searchable_index,
                        validity=self._engine._validity,
                    )

                case Merge(fusion=fusion):
                    lists = [_fuse(lists, fusion=fusion)]
                case Rerank(reranker=name):
                    lists = [
                        _rerank(
                            candidates,
                            question=question,
                            reranker=self._engine._rerankers[name],
                        )
                        for candidates in lists
                    ]
                case Take(window=window):
                    [candidates] = lists
                    return _take(
                        candidates,
                        where=where,
                        window=window,
                        index=searchable_index.index,
                        validity=self._engine._validity,
                        limit=limit,
                    )
                case _:
                    assert_never(stage)
        raise AssertionError("replay guarantees a Take ends every statement")

    def _check_filter(self, where: Filter) -> None:
        """Check that `where` is a filter fitting the statement's index.

        Raises:
            TypeError: `where` isn't a filter.
            SchemaMismatchError: `where` uses a field that isn't in the index's schema.
        """
        check_is_filter(where, owner="execute")
        schema = self._engine._indexes[self.statement.index].index.schema
        check_filter_matches_schema(where, schema=schema)

    @classmethod
    def _create(cls, statement: Statement, *, engine: Engine) -> PreparedStatement:
        """Return `statement` prepared for `engine`; only `Engine.prepare` calls it."""
        # `__new__` skips `__init__`, which refuses every other caller
        prepared = cls.__new__(cls)
        object.__setattr__(prepared, "_statement", statement)
        object.__setattr__(prepared, "_engine", engine)
        return prepared

    def __setattr__(self, name: str, value: object) -> None:
        """Refuse every change.

        Raises:
            AttributeError: Always.
        """
        raise AttributeError(f"a PreparedStatement can't be changed ({name})")


def _retrieve(
    question: str,
    scans: Sequence[Scan],
    *,
    where: Filter,
    searchable_index: SearchableIndex,
    validity: Validity,
) -> list[list[Candidate]]:
    """Return one ranked list of valid candidates per scan, each found by it.

    Every scan reads the index with the `where` filter; the chunks of all lists are
    checked in one call, then each list keeps its valid ones. A candidate's `Found`
    rank is its place in what the scan returned, so a dropped chunk doesn't move the
    others up.

    Raises:
        ValueError: An embedder returned other than one row for the question.
    """
    hits_by_scan = [
        (
            scan.name,
            _search(
                question,
                scan,
                where=where,
                index=searchable_index.index,
                registered=searchable_index.scans[scan.name],
            ),
        )
        for scan in scans
    ]
    valid_ids = _valid_ids(
        [hit.chunk for _, hits in hits_by_scan for hit in hits], validity=validity
    )

    return [
        [
            Candidate(
                chunk=hit.chunk,
                trail=(Found(scan=scan_name, rank=rank, score=hit.score),),
            )
            for rank, hit in enumerate(hits, start=1)
            if hit.chunk.id in valid_ids
        ]
        for scan_name, hits in hits_by_scan
    ]


def _search(
    question: str,
    scan: Scan,
    *,
    where: Filter,
    index: Index,
    registered: RegisteredScan,
) -> list[Hit]:
    """Return what `scan` reads from `index` for `question`, at most its depth.

    A vector scan embeds the question and searches; a lookup scan parses it into a
    filter and fetches what that and `where` both match (`LookupScan`).

    Raises:
        ValueError: The embedder returned other than one row for the question.
        TypeError: A lookup's parser returned something other than a filter or `None`.
        SchemaMismatchError: A lookup's parser filtered by a field the index's schema
            doesn't declare.
    """
    match registered:
        case VectorScan(binding=binding):
            query = _embed_question(question, binding=binding)
            return index.search(query, where=where, limit=scan.depth)
        case LookupScan(parse_filter=parse_filter):
            lookup_filter = parse_filter(question)
            if lookup_filter is None:
                return []
            # Checked on every run: the filter is built from each question
            check_is_filter(
                lookup_filter, owner=f"the parse_filter of scan {scan.name!r}"
            )
            check_filter_matches_schema(lookup_filter, schema=index.schema)
            found = index.find(where=and_(lookup_filter, where))
            # find promises no order; ids give one that's the same on every run
            kept = sorted(found, key=lambda chunk: chunk.id)[: scan.depth]
            return [Hit(chunk=chunk, score=_LOOKUP_SCORE) for chunk in kept]
        case _:
            assert_never(registered)


def _embed_question(question: str, *, binding: VectorBinding) -> VectorQuery:
    """Return the query searching `binding`'s vector for `question`.

    Raises:
        ValueError: The embedder returned other than one row, or the row doesn't fit
            the vector.
        TypeError: A number or weight isn't a real number, or a sparse id isn't an
            integer.
    """
    match binding:
        # One name per kind: mypy gives a variable one type in the whole function
        case DenseBinding(vector=vector, embedder=embedder):
            numbers = embedder.embed_queries([question])
            _check_one_row(numbers, vector=vector)
            return vector.query(numbers[0])
        case SparseBinding(vector=vector, embedder=embedder):
            weights = embedder.embed_queries([question])
            _check_one_row(weights, vector=vector)
            return vector.query(weights[0])
        case _:
            assert_never(binding)


def _fuse(lists: list[list[Candidate]], *, fusion: Fusion) -> list[Candidate]:
    """Return the lists fused into one, best first, each candidate's trail joined.

    Candidates are the same chunk when their ids are (a statement searches one index).
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
    candidates: list[Candidate], *, question: str, reranker: Reranker
) -> list[Candidate]:
    """Return `candidates` scored by `reranker` and sorted by its scores, best first.

    The reranker reads each chunk's `embedded_text`, the text the index embedded,
    context included. Ties keep their earlier order.

    Raises:
        ValueError: The reranker returned a score count other than the texts', or a
            score that isn't finite.
        TypeError: A score isn't a real number.
    """
    texts = [candidate.chunk.embedded_text for candidate in candidates]
    scores = score_texts(question, texts, reranker=reranker)

    reranked = sorted(
        zip(scores, candidates, strict=True), key=lambda pair: pair[0], reverse=True
    )

    return [
        candidate.followed_by(Reranked(rank=rank, score=score))
        for rank, (score, candidate) in enumerate(reranked, start=1)
    ]


def _take(
    candidates: list[Candidate],
    *,
    where: Filter,
    window: Window | None,
    index: Index,
    validity: Validity,
    limit: int,
) -> list[Passage]:
    """Return the best `limit` candidates that pass the final gate, as passages.

    The gate runs before the cut, so the limit still fills when it can. Windows are
    found after both, for the passages returned only. This is the one place passages
    are built.
    """
    valid_ids = _valid_ids(
        [candidate.chunk for candidate in candidates], validity=validity
    )
    hits = [c for c in candidates if c.chunk.id in valid_ids][:limit]
    hit_windows = _windows(
        [hit.chunk for hit in hits],
        window=window,
        where=where,
        index=index,
        validity=validity,
    )

    return [
        Passage.from_candidate(hit, window=hit_window)
        for hit, hit_window in zip(hits, hit_windows, strict=True)
    ]


def _windows(
    hits: list[Chunk],
    *,
    window: Window | None,
    where: Filter,
    index: Index,
    validity: Validity,
) -> list[tuple[Chunk, ...]]:
    """Return each hit's window, in the order of `hits`, reading the index once."""
    if window is None:
        return [(hit,) for hit in hits]

    # No hits need no read, and an empty or_() is refused
    if not hits:
        return []

    neighbors = _find_neighbors(
        hits, window=window, where=where, index=index, validity=validity
    )
    partitions: dict[tuple[object, ...], list[Chunk]] = defaultdict(list)
    for chunk in [*hits, *neighbors]:
        partitions[_partition_key(chunk, window=window)].append(chunk)
    return [
        _window_of(hit, partitions[_partition_key(hit, window=window)], window=window)
        for hit in hits
    ]


def _find_neighbors(
    hits: list[Chunk],
    *,
    window: Window,
    where: Filter,
    index: Index,
    validity: Validity,
) -> list[Chunk]:
    """Return the valid chunks near any hit that `where` lets through, hits left out."""
    near_any_hit = or_(*(_near(hit, window=window) for hit in hits))
    # The hits passed the gate already: asking again could drop one from its window
    hit_ids = {hit.id for hit in hits}
    found = [
        chunk
        for chunk in index.find(where=and_(where, near_any_hit))
        if chunk.id not in hit_ids
    ]
    valid_ids = _valid_ids(found, validity=validity)
    return [chunk for chunk in found if chunk.id in valid_ids]


def _near(hit: Chunk, *, window: Window) -> Filter:
    """Return a filter matching the chunks of `hit`'s window."""
    position = hit[window.order_by]
    each_side = window.chunks_each_side
    same_partition = [field.eq(hit[field]) for field in window.partition_by]
    in_range = window.order_by.in_(
        range(position - each_side, position + each_side + 1)
    )
    return and_(*same_partition, in_range)


def _window_of(
    hit: Chunk, partition: list[Chunk], *, window: Window
) -> tuple[Chunk, ...]:
    """Return the chunks of `partition` within range of `hit`, in `order_by` order."""
    position = hit[window.order_by]
    nearby = [
        chunk
        for chunk in partition
        if abs(chunk[window.order_by] - position) <= window.chunks_each_side
    ]
    # Repeated positions (allowed, if unusual) fall back to ids, the same every run
    return tuple(sorted(nearby, key=lambda chunk: (chunk[window.order_by], chunk.id)))


def _valid_ids(
    chunks: Sequence[Chunk], *, validity: Validity
) -> AbstractSet[uuid.UUID]:
    """Return the ids of `chunks` that may be served, asking once, and only if needed."""
    # One chunk found by several scans is asked about once
    unique = list({chunk.id: chunk for chunk in chunks}.values())
    if not unique:
        return frozenset()
    return validity.valid_ids(unique)


def _check_one_row(rows: Sized, *, vector: Vector) -> None:
    """Check the embedder for `vector` returned exactly one row for the question.

    Raises:
        ValueError: It returned none or several; the message names the vector, to find
            the embedder among several.
    """
    if len(rows) != 1:
        raise ValueError(
            f"the embedder for vector {vector.name!r} "
            f"returned {len(rows)} rows for one question"
        )


def _names(registered: Iterable[str]) -> str:
    """Return the names `registered`, quoted and sorted, for a message; "none" if none."""
    return ", ".join(repr(name) for name in sorted(registered)) or "none"


def _partition_key(chunk: Chunk, *, window: Window) -> tuple[object, ...]:
    """Return `chunk`'s values for the window's `partition_by` fields, in order."""
    return tuple(chunk[field] for field in window.partition_by)
