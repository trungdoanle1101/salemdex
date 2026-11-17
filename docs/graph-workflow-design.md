# Retrieval as a graph of stages

Status: build steps 1 to 3 done: the core, both index backends with `Index.search`
(section 10), embedders and rerankers (section 6, "Embedders"), and the engine with its
session (sections 5 and 6). Next: build step 4, in the server. Branch:
`features/retrieval-redesign`.

- Built, in `packages/retrieval` (`salemdex_retrieval`), with tests:
  - `schema.py`: `One` / `Many` fields, `Schema` with presets, `OneValue` / `ManyValues`,
    the value checks;
  - `filters.py`: `Equals` (`eq`), `AnyOf` (`any_of`), `and_` / `or_` / `not_`, and
    `Everything`, matching every chunk; `check_is_filter`, for filters arriving untyped;
  - `vectors.py`: `Dense` vectors, declared on the schema beside its fields, with checked
    values (`Dense.value`) and queries (`Dense.query`); the closed unions `VectorValue` and
    `VectorQuery`;
  - `chunks.py`: `Chunk`, and `Chunk.check_fits(schema)` raising `SchemaMismatchError`;
    `EmbeddedChunk`, a chunk with its vector values, which every write takes; `Hit`, a chunk
    with its score, which search returns;
  - `index/in_memory.py`: `InMemoryIndex` with `put`, `find`, `delete`, `replace` and
    `search` (exact cosine and dot, in plain Python: the core has no dependencies);
  - `index/protocol.py`: the `Index` protocol, `check_batch`, `check_filter`, `check_query`
    and `check_limit`, shared by every backend; `tests/index/test_contract.py` runs the
    behaviour tests, search included, against each backend: in memory, Qdrant's local mode,
    and a Qdrant server when `KB_TEST_QDRANT_URL` is set;
  - `index/qdrant/payload.py`: encoding a chunk as a Qdrant payload and decoding it strictly;
  - `index/qdrant/filters.py`: compiling filters to Qdrant's must / should / must_not;
  - `index/qdrant/index.py`: `QdrantIndex` with `put`, `find`, `delete`, `replace` and
    `search` (`query_points`, filtered inside Qdrant), behind the `qdrant` extra, and `QdrantIndex.open`, the startup check: it creates the collection
    with the schema's named vectors and the payload indexes it lacks (one per `indexed=True`
    field, plus `written_fields`), and refuses to open on vectors that differ from the
    schema's, on an index of the wrong type, or on points that don't fit the schema
    (section 6, "The schema"); `tests/index/test_qdrant_open.py` tests it, the payload
    indexes against a server only;
  - `embedders/protocol.py`: the `DenseEmbedder` protocol, `DenseBinding` and the closed
    union `VectorBinding`, and `embed_chunks(chunks, *, bindings)` (section 6,
    "Embedders"), tested in `tests/embedders/test_embed_chunks.py`;
  - `embedders/fake.py`: `FakeDenseEmbedder(size=…)`, a deterministic embedder by feature
    hashing (signed, NFC and case-folded words), so texts sharing words score higher;
    tested in `tests/embedders/test_fake.py`.
  - `checks.py`: `check_positive_int`, `to_finite_float`, `to_stripped_str`, shared by
    every layer;
  - `rerankers/`: the `Reranker` protocol, `score_texts`, and `FakeReranker`;
  - `engine/stages.py`: `Retrieve` (named sources, one index), `Merge` with `Rrf`,
    `Rerank`, `Take`, as data;
  - `engine/statements.py`: `search(question)`, the type-state steps, and `replay`;
  - `engine/results.py`: the trail entries (`Found`, `Fused`, `Reranked`), `Candidate`
    and `Passage`;
  - `engine/validity.py`: the `Validity` port (`valid_ids(chunks) -> ids to keep`) and
    `AlwaysValid`;
  - `engine/executor.py`: `execute(statement, *, access, validity)`. It replays the
    statement and checks the access filter against every source's schema, before any
    model is called; then it searches every source with the access filter, gates where
    candidates enter (one validity call per retrieve) and again inside `Take`, fuses by
    RRF, and reranks every candidate it receives. Tested in `tests/engine/test_executor.py`
    with a scripted index and spies that count embedder, validity and reranker calls.
  - `engine/session.py`: `Engine(validity=…)` and `Session`, from `engine.session(access=…)`,
    both frozen; `Session.execute(statement)` is the one public way to run a statement.
    Tested in `tests/engine/test_session.py`.
  - The re-exports, in three tiers (section 10, "Import paths"); `tests/test_public_api.py`
    pins the top level's `__all__`.
- **Pick up here** (as of 2026-11-18): build step 4, in the server (section 10):
  1. The server's schema: `KbChunks` on an access preset whose `access_group_ids` is
     `indexed=True`, as is every other field each search filters by; chunk ids
     `uuid5(document_id, version, position)` (section 6, "The chunk" and "One filter for
     every schema"). Settle which fields the standard preset holds first (section 11).
  2. Port `run_index_job` to `embed_chunks` and `index.put`, with the version switch:
     `documents.indexed_version` (one migration), set in the transaction that marks the
     index job done, then `delete` of the document's other versions after the commit
     (section 6, "Currency").
  3. The server's validity adapter: a hit is kept when its document is active and its
     version is `indexed_version`, one query per gate (section 6, "How the slots plug in").
  4. Then search on it (server build step 5): one service opens sessions, a test of the
     access filter it builds for each kind of asker, and the import-boundary check.
- Decided, built when needed: a limit on `merge` (when fusing deep lists leaves too many to
  rerank), resolving sources for generated entries, caller filters
  (`search(q).where(…)`), `expand`, query transforms (sections 5 and 6).
- Later, when a caller needs it: `chunk.get(field)` for typed reads (step 3 or 4), `in_` on
  `One`, filtering by chunk ids.

The project is named **Salemdex**. This engine is its first package under that name,
`salemdex-retrieval` (section 10).

## 1. The problem

The project is a knowledge base: a product whose answers must be **accurate** and **safe**.
Every answer is built from what search returns, so search is where both are won or lost:

- An answer can't be better than its evidence, so search caps accuracy.
- An answer can only reveal what search gave it. If search returns only what the asker may see,
  everything built on top is safe without checking again.
- Evidence must be checkable: each passage is the source's own words, traced to its source.

Retrieval patterns range from one vector search to agentic loops. Hard-wiring one of them means
rewriting search for the next. The current framework API also has these problems:

- A filter can't OR across fields ("my groups OR my managed collections").
- There are two filter mechanisms: fixed fields (`document_ids`, `chunk_keys`, `reference_keys`)
  and `attributes`.
- `search_chunks` takes 11 parameters, mixing tools and settings. `limit` and `rerank_depth` are
  coupled through `max()`.
- `ChunkFilter()` means everything. Removing a document is `replace_document(id, [], [])`.
- Ids are `str`, which already caused a UUID bug.
- Reading the index needs the private `_payload` module and an embedder identity.
- Indexed attributes are declared in two places (`QDRANT_INDEXED_ATTRIBUTES` and the chunks).

## 2. Search in its most basic form

```text
search(question, asker) → passages, best first
passage = the source's own words + where they come from (document, version, location) + rank
```

| # | Property | Kind |
|---|---|---|
| 1 | **Allowed:** only what the asker may see; hidden things leave no trace, not even in counts | guarantee |
| 2 | **Current:** nothing deleted; one whole version per document, never a mix | guarantee |
| 3 | **Faithful:** the text is exactly the source's, traceable to document, version, location | guarantee |
| 4 | **Ranked:** what most likely answers the question comes first | quality |

**Guarantees** always hold, are tested as yes/no, and are never traded for quality. **Quality**
is measured (hit rate, MRR) and tuned. This split drives the whole design.

Conceptually there are two steps: **narrow** (what this asker may see, now) and **order**
(relevance, keep top k). Everything else refines one of the two, or sits above search.

## 3. The basic pipeline

```text
request: question, asker, scope?, k
│
├─ 1. Encode         question                      → query vectors (dense, sparse…)
├─ 2. Allow          asker, scope?                 → filter
├─ 3. Retrieve       query vectors, filter, depth  → candidates: chunk + index score
├─ 4. Confirm        candidates                    → candidates still current
├─ 5. Rerank         question, candidate texts     → candidates reordered + rerank score
└─ 6. Cut & shape    candidates, k                 → passages: text, source, score
```

- 1 and 2 are independent.
- 4 runs before 5: no reranking of candidates about to be dropped.
- The filter runs inside the index. Why: filtering afterwards loses results and reveals hidden
  hits. The index applies access; the database confirms currency.
- **One score per passage:** from the last stage that ordered the candidates (rerank if present,
  else the index or fusion score). Comparable only within one response. Why: scores from
  different stages, or different questions, aren't on the same scale.

## 4. Fixed frame, pluggable stages

Only the quality stages are pluggable. Why: if every stage can be swapped, a plugin can break a
guarantee (e.g. an "expand to related chunks" stage that pulls in chunks the asker can't see).

```text
fixed frame (always runs, can't be removed)     pluggable inside it
─────────────────────────────────────────────    ──────────────────────────
allow    → every candidate source gets the filter  query transforms
confirm  → current check on whatever comes out     sources, merging
faithful → passages carry source text + origin     reranking, expansion, pruning
```

### Stage kinds

| Kind | In → out | Examples |
|---|---|---|
| query transform | question → queries | rewrite, multi-query, HyDE, decomposition |
| source | query, filter → candidates | dense, sparse, keyword, generated entries |
| merge | candidate lists → candidates | RRF, union |
| expand | candidates, filter → candidates | neighbours, parent section, linked chunks |
| rerank | question, candidates → candidates | cross-encoder, LLM judge |
| prune | candidates → candidates | dedupe, threshold, diversity |

Every stage that adds candidates (source, expand) receives the filter. The framework fixes
*where* the guarantees apply. The app decides *what* they mean (section 6).

## 5. Statements and the executor (the SQLAlchemy pattern)

**A statement describes a search; an executor runs it**, like `select()` and
`session.execute()`.

```text
stmt = search(question).retrieve(dense, sparse).merge(rrf).rerank(cross_encoder).take(10)
passages = engine.session(access=access).execute(stmt)
```

- **Access is bound to the session, not the statement.** Why: a statement can't forget the
  access filter. The executor adds it to every source and runs the validity gate where
  candidates enter and again before `Take`.
  (How the slots plug in: section 6.)
- **Type-state builder:** each method returns a type that offers only the steps allowed next, so
  mypy checks the order:

  ```text
  search(question)                                   → Question
  Question       → .retrieve(source)                 → Candidates
  Question       → .retrieve(a, b, …)                → CandidateLists
  CandidateLists → .merge(fusion)                    → Candidates
  Candidates     → .rerank() (later .expand() .prune()) → Candidates
  Candidates     → .take(limit)                      → Search   (complete)

  later, with query transforms:
  Question       → .rewrite()                        → Question
  Question       → .split()                          → Queries
  Queries        → .retrieve(sources…)               → CandidateLists (always)
  ```

  - **A state is named after what it guarantees,** so later steps only add. `Question` holds
    exactly one question, which is why one source gives one list; `Queries` (several) arrives
    with `split`, and its `retrieve` always gives lists to merge.
  - **Several lists must be merged before anything else.** `CandidateLists` offers only
    `merge`, so mypy refuses `retrieve(a, b).rerank(…)`.
  - **The complete state is `Search`, not `Passages`:** `execute(search: Search) ->
    list[Passage]` would otherwise put the plan and its result under nearly one name.
  - **States are siblings over a private base** holding `question` and `stages`. Why: a
    state subclassing another would be accepted where the earlier one is expected.

- **A statement built at run time is replayed through the steps.** `replay(statement)`
  starts again from `search(statement.question)` and takes one step per stage. Each
  (state, stage) pair must be a step that state offers, and the last state must be a
  `Search`; anything else raises `StatementError` naming the stage's position. The executor
  runs the replayed statement.
  - Why: the order rules then live once, in the steps. A separate check over the whole
    tuple ("first a retrieve, a merge exactly when…, one take at the end") would encode the
    same state machine twice, and the copies would drift. A new step is one method and one
    `case`.
  - Why not in `Search.__post_init__`: replaying ends with `take()`, which builds a
    `Search`, which would replay again.
  - **Every bound limits what a stage keeps, never what it reads:** a source's `depth`, a
    take's `limit` (later a merge's). Each is "at most" (a source can return fewer hits,
    access and validity remove more), and none is compared with another: a loose statement
    isn't a wrong one.
  - **A rerank scores every candidate it receives,** so its cost is set by the stages before
    it: the source's `depth`, or after a merge the union of the sources' lists. Why: the
    usual convention for rerankers (Cohere's and LlamaIndex's score everything they're given),
    and a number on the rerank would be the one bound on input, reading as "depth" beside
    the sources' output bound. Search engines' alternative, a window of the top N to rescore
    (Elasticsearch's `window_size`, Vespa's `rerank-count`), returns as an output bound on
    `merge` when fusing deep lists leaves too many to rerank.
- **Every state checks its question:** stripped, and refused when it's empty, in the base's
  `__post_init__`. Why: a state built by hand skips `search()`.
- **Immutable and generative:** each method returns a new statement. Statements are data: they
  can be inspected, logged as provenance and replayed.
- **The question stays out of `repr`.** Why: it can reveal what the asker is after, and a
  traceback or a debug line would print it. Auditing questions is the app's choice: it reads
  `statement.question` and stores it under its own rules (access, retention). The executor's
  trace, like the trail, holds no text.
- **Composite stages:** a stage can contain stages (a merge over several sources). That gives a
  tree, which covers fan-out and fan-in. There is no free-form graph of untyped nodes. Why: it
  would lose strict typing and make the guarantees hard to prove.

### Candidates: what flows between stages

A **passage** is what comes out at the end. A **candidate** is what flows between stages. Later
stages can only use what earlier stages kept, so a candidate keeps its history:

```text
Candidate
  chunk   the chunk (id, text, field values)
  score   the current score: the last trail entry's
  trail   one entry per stage that touched it:
          stage, score or rank, found through (which source, or which generated entry)
```

- Why a trail: it explains a hit ("why is this here?"), lets each stage be evaluated on its
  own, gives answers their provenance, and lets a prune use a specific stage's score ("rerank
  score ≥ x").
- "Found through" generalises today's `found_through` to every stage.
- The trail holds ids, scores and stage names, never text. Why: it can then be logged as
  provenance.
- The passage's single score is the last trail entry's (section 3).
- Rejected: only chunk + score (each stage overwrites the score, so history is lost); ids only
  with text loaded when needed (more complex loading, not needed at depth 50).

## 6. Data: generic chunks, a schema, fixed slots

**Flexible data, restricted structure.** The framework knows structure: what can follow what.
The app decides meaning: what a document, an access group or a reference is. A chunk has no
built-in concept of document, access or reference.

### The chunk

```text
Chunk
  id             UUID, unique across the whole index
  text           what is shown and returned; never changed
  embedded_text  what the model embeds; the text itself unless one is given
  field_values   {field: value}, keyed by the field object; read-only
```

- **Built with keywords only.** Why: `text` and `embedded_text` are both strings, and swapped
  by position, search would return the context as the source's words.
- **`embedded_text` is optional to give, always a string to read.** Left out, the constructor
  fills in the text. Why: no reader handles `None`, and nobody has to know what it means.
- **Text stays out of `repr`** (`field(repr=False)`). Why: a chunk in a log line or an error
  would otherwise print restricted text.
- **Values are keyed by the `Field` object, not its name.** Why: `KbChunks.document_id` and
  `Entries.document_id` share a name but are different fields; only identity tells them
  apart. It commits `Field` to never defining `__eq__` (so no `==` filter sugar): a dict
  would compare keys with it.
- **Two values with one name are refused when the chunk is built** (`ValueError`): the same
  field twice, or fields from two schemas. No schema has two fields with one name, so such a
  chunk can't fit anywhere.
- **`chunk.check_fits(schema)`** checks a value for every field of the schema and no other,
  comparing fields as objects; an empty `Many` counts as a value. It raises one
  `SchemaMismatchError` (a `ValueError`) naming the chunk id and every extra and missing
  field. Why a separate error from `SchemaError`: the schema is fine, the data doesn't fit it,
  so a caller (the worker) can treat it as permanent.

- **The id is unique across the index**, not only within a document. The app derives it
  deterministically, e.g. `uuid5(document_id, version, position)`. Why: Qdrant takes UUIDs
  natively, mypy can check the type, and re-indexing the same version gives the same ids, so a
  retried write is idempotent. The version is part of the id so that a new version's chunks
  never overwrite the old version's while both are in the index (see "Currency" below).
- **`embedded_text` is where context goes**, e.g. the headings path prefixed to the text. That
  is the reason it can differ from `text`.
- **Vectors aren't part of the chunk.** The schema declares them, and a write pairs a chunk with
  its vector values in an `EmbeddedChunk` ("Vectors" below).
- **Everything else is a field:** document id, version, position, headings, reference keys,
  access groups.
- **"Faithful" narrows.** The framework guarantees only that `text` comes back exactly as it
  was indexed. Where it came from is the app's fields.

### The schema

Declared like SQLAlchemy's `Table` / `Column`, but called fields: a chunk isn't a table row.
Fields are declared once, with their types and whether they are indexed. Each field is a
typed handle on one chunk value. Its type `T` is checked wherever the field is used.

```python
class KbChunks(Schema):
    document_id = One(UUID, indexed=True)  # One[UUID]
    version = One(int, indexed=True)  # One[int]
    position = One(int)
    headings = Many(str)  # Many[str]: several values, maybe none
    groups = Many(UUID, indexed=True)
```

**Two field kinds: `One[T]` (a single value) and `Many[T]` (several values).** Why not a
single `Field[T]`: mypy can't infer `T` from `tuple[str, ...]`, only from a real class. Also,
the operations differ: `eq` and ranges fit one value, `any_of` / `all_of` fit several.

- `T` is one of `str`, `int`, `float`, `bool`, `UUID`, `datetime`: what every backend can store.
  `Field.__init__` checks this and names the allowed types. It compares classes exactly, not
  with `issubclass`, because `bool` is a subclass of `int`.
- **Values are checked wherever they enter:** in `value()` (writes) and in `eq` / `any_of`
  (filters), with one shared check. Why filters too: statements built at run time (config, an
  agent) arrive as `Any`, and a `str` where a `UUID` belongs would match nothing instead of
  failing.
  - The type must match exactly; an `int` is widened to `float` for a `float` field.
  - A `datetime` must have a timezone.
  - A `float` must be finite, and an `int` must fit in 64 bits (`ValueError`). Why: JSON has
    no NaN or infinity, Qdrant stores int64, and `nan` never equals itself, so `eq(nan)` would
    match nothing without saying so.
  - `Many.value` and `any_of` refuse a `str`, `bytes` or `Mapping`: each is iterable, and
    would be taken apart into characters, ints or keys.
- `One(UUID)` is the call; `One[UUID]` is the type mypy infers from it. The type is passed as an
  argument because the run time needs it too (checking and encoding values, choosing backend
  index types), and generic parameters are erased at run time.

**Every field is required; there are no optional fields (for now).**

- Every chunk has a value for every field. A `Many` field may hold no values (`[]`), and that
  is how "none" is said for multi-valued data: no tags, no groups.
- Why: "missing" would need rules in every filter, like SQL's `NULL`, and both backends would
  have to apply them the same way. A `not_(…)` filter lets chunks without the field through (in
  Qdrant, `must_not` matches them), so a missing value could fail open. It also keeps `T` free
  of `None`, and `One` needs no overloads.
- **A missing or `null` value is a data error, caught at three points:**

  | Where | What happens |
  |---|---|
  | writes | every chunk is checked against the schema before it's sent; a missing value raises, naming the chunk id and field. An empty `Many` is written as `[]`, never left out |
  | the index opening | count, for each field, the chunks not written with it, and find entries the schema doesn't define (in Qdrant, one facet over `written_fields`, below); if any exist, refuse to start: "412 points lack `version:int:one`; re-index into a new collection" |
  | reads | a hit that still lacks a value raises a typed error naming the chunk id and field. No half-filled chunk, no guessed default |

  Why refuse to start: filters then only ever run over chunks that keep the promise. Why raise
  on reads rather than skip: skipping also fails safe, but data would vanish from results without
  anyone noticing.
- **No sentinel values** (`0`, `""`, an epoch date) to stand in for "missing". They quietly take
  part in filters: `page.lt(5)` would match every chunk whose page is the `0` sentinel. A field
  that would need one is the signal that optional fields are due.
- **When optional fields come:** with the first genuinely sparse, kind-specific field (a page,
  a sender, a sheet name). Adding them later breaks no existing declaration. Two rules come with
  them, so "missing" never matches by accident:
  - fields used by the guarantee slots (access, validity) must be required; the compile step
    checks it;
  - `not_` over an optional field is rejected unless the filter says what happens to missing
    values (`is_missing()` / `is_present()`).

**What `T` buys at each use:**

| Use | Example | What mypy catches |
|---|---|---|
| filter | `KbChunks.document_id.eq(doc_id)` | `eq("abc")`: a `str` where a `UUID` is expected |
| ordered filters | `KbChunks.version.ge(2)` | `ge` on a `UUID` field. It exists only for int, float, datetime (a self-type on the method) |
| multi-valued | `KbChunks.groups.any_of(ids)` | `eq` on a `Many`, `any_of` on a `One` |
| read | `chunk.get(KbChunks.document_id)` → `UUID` | no `str` or `Any` coming back. It's generic: `get(col: One[T]) -> T` |
| stage settings | `expand(order_by=KbChunks.position)` | `order_by` takes only ordered `One` fields |

**Writing chunks: typed fields plus a value builder.**

```python
Chunk(
    id=…,
    text=…,
    field_values=[KbChunks.document_id.value(doc_id), KbChunks.version.value(3), …],
)
```

- Every field-value pair is type-checked. A missing field is caught at run time, when the
  index calls `chunk.check_fits(schema)` on the write.
- **Two value classes, like the two field kinds:** `One.value` returns `OneValue[T]` (`value:
  T`) and `Many.value` returns `ManyValues[T]` (`values: tuple[T, ...]`).
  `type FieldValue = OneValue[Any] | ManyValues[Any]` names either, and a backend matches on
  the class to tell one value from several. Filters follow the same pattern: `Equals[T]`,
  `AnyOf[T]`, and `type Filter = Equals[Any] | AnyOf[Any]`.
  - Why not one generic `FieldValue[T]`: with `value: T`, a `Many` field's tuple isn't a `T`,
    and `value: T | tuple[T, ...]` wouldn't tie the shape to the kind of field.
  - Why the unions aren't generic: a chunk's values, and a filter combining fields, span
    several types, so every user would write `[Any]`. `T` is kept where it is known: `value()`,
    `eq` and `any_of` return the specific class.
- **Fields come only from the class body and from `Schema` bases.** A `Field` in a plain mixin
  isn't part of the schema; shared fields belong in a `Schema` preset.
- Why this level: chunks are built in one place (the index handler), while filters and reads are
  spread everywhere, and those are fully typed.
- Possible later: the SQLAlchemy 2.0 style. The schema class is also a typed record
  (`dataclass_transform` plus descriptors), so mypy catches missing fields too. It needs a small
  experiment to see how well mypy handles it. Filters and reads wouldn't change.
- Rejected: a plain dataclass with `Annotated` metadata. Writes would be typed, but filters
  would name fields by string and lose their typing.

**Each index owns its schema; the engine reads it from the index (`index.schema`).**

- Why: one source of truth. Two copies, one on the index and one on the engine, could drift.
  The compile step would then approve a filter on a field the index doesn't have.
- There is more than one schema (chunks, generated entries), each with its own index.
- **One index, one schema.** Why: `check_fits`, the filter check and the startup check all
  compare against a single schema; with two, a filter on one schema would run over the other's
  chunks, and `not_` would turn "matches nothing" into "matches everything". Several kinds of
  records are several indexes, combined by a merge stage. Two logical indexes sharing one Qdrant
  collection is a backend detail; each still has one schema. A schema that changes over time is
  a migration (backfill or re-index), not a second schema.
- The compile step checks that every field in a statement belongs to the schema of the index it
  runs against. A chunk field can't filter the entries index.
- **The index checks its stored data when it opens:** it compares the schema with the backend
  (e.g. Qdrant payload indexes), then creates what's missing or fails with a clear message.
  Why: otherwise filters run slowly or wrongly without saying so.
- Not now: statements generic over their schema (`Candidates[KbChunks]`), which would let mypy
  catch cross-schema fields. The compile check is enough to start with.

**Other uses of the schema:**

- Backend indexes follow from the declaration. That replaces the duplicated
  `QDRANT_INDEXED_ATTRIBUTES`.
- Values are encoded to the backend's format and decoded back (`UUID`, `datetime`). Why: that
  removes the need for the private `_payload` module.
- **The Qdrant payload layout:**

  ```python
  {
      "text": "…",
      "embedded_text": "…",
      "field_values": {"document_id": "3f2a…", "version": 2, "groups": ["…"], "headings": []},
      "written_fields": ["document_id:uuid:one", "groups:uuid:many", "headings:str:many", "version:int:one"],
  }
  ```

  - **Only the Qdrant backend knows it.** Encoding and decoding are functions in
    `index/qdrant/payload.py`, not methods on `Chunk`. Why: the layout and encodings are
    Qdrant's choices (another backend would encode differently), and decoding needs the schema
    to know each field's type, which a chunk doesn't carry.
  - **The chunk id is the point id**, not a payload key; Qdrant takes UUIDs natively.
  - **Field values sit under their own key, `field_values`**, apart from the chunk's own parts.
    The backend turns `KbChunks.document_id` into the key `field_values.document_id` for
    filters and payload indexes (Qdrant handles dotted keys). Why a separate key: in a flat
    payload, a field named `text` would overwrite the chunk's text and break "faithful"
    without an error. Reserving names instead would make every payload key the framework adds
    later a breaking change for existing schemas, and would put one backend's layout into the
    schema's rules. Why this name: it's the same idea as `Chunk.field_values` (values keyed by
    field), and "fields" already means the declarations (`Schema.fields()`).
  - **No layout number yet.** Strict decoding already turns a changed layout into a loud error
    (a renamed key is missing, a changed encoding fails the value checks), so old points can't
    be misread silently. What keeps the layout from changing by accident is a golden test
    (`tests/index/test_qdrant_payload.py`): it compares one chunk's payload with the stored
    layout written out in full, so any change fails CI, where a version number would rely on
    someone remembering to bump it. When the layout first really changes, add a `layout` key:
    a point without one is layout 1, so old points need no migration to be recognised.
    Changing the layout means re-indexing.
  - **`embedded_text` is always stored**, even when it equals the text. Why: re-embedding with
    a new model needs it without re-chunking, and a missing key always means a corrupt point,
    so decoding stays strict. The duplicated text is small next to the vectors.
  - **Encodings:** `str`, `int`, `float`, `bool` as they are; `UUID` as its string (Qdrant's
    `uuid` payload index); `datetime` as ISO 8601 with its offset (what Qdrant's `datetime`
    index parses); a `Many` as a list, `[]` when empty, never left out.
  - **`written_fields` records the fields a point was written with**, sorted, each as
    `name:type:one|many`, with the type from a fixed table of stored names (`uuid`, `str`,
    `bool`, `int`, `float`, `datetime`), not Qdrant's index types or Python's class names.
    Why: Qdrant can't tell a missing list from `[]` (`is_empty` and `values_count` match
    both), so the opening check couldn't find a `Many` that's missing without reading every
    point with an empty list. With the entries, `open` counts in one facet request over a
    keyword payload index on `written_fields`: an expected entry on fewer points than the
    collection holds is missing somewhere, and an entry the schema doesn't define comes from
    another schema or an older one. A changed type or `One`/`Many` changes the entry, so it's
    caught too. Why our types: the entry describes how the value was encoded, and Qdrant's
    index types would merge types the decoder keeps apart. `indexed=True` isn't part of the
    entry: turning an index on doesn't change what's stored. What it can't catch: a value
    removed by hand while its entry stays; strict decoding still raises on read.
  - **Decoding is strict** (the "reads" checkpoint): a malformed point id, a missing key, an
    unknown key under `field_values`, `written_fields` not recording exactly the schema's
    fields, a wrong type or a malformed value raises one error,
    `SchemaMismatchError`, naming the point id and the key, so callers (a search, the startup
    check) catch a single type. Every value goes back through `field.value()`, so the
    write-time checks run again, and `check_fits` confirms the chunk has exactly the schema's
    fields.
- **Writes are filter-based**, like SQL's `DELETE … WHERE`: `index.put(chunks)`,
  `index.delete(where=…)`, `index.replace(where=KbChunks.document_id.eq(x), with_=new_chunks)`.
  The framework needs no document concept even for index upkeep, and
  `replace_document(id, [], [])` disappears.
  - **`put` is an upsert:** a stored chunk with the same id is overwritten. Why: a retried job
    writes the same deterministic ids again and must succeed (idempotent consumer); Qdrant's
    upsert works the same way. Named `put`, not `add`, because "add" reads as an insert and
    hides the overwrite.
  - **Every write is checked in full before anything is stored:** each chunk fits the schema,
    and no two share an id. Why: a batch failing halfway would leave part of a version beside
    the old one, the mix "Current" rules out.
  - **`replace` is SQL's `DELETE … WHERE` then `INSERT`:** the new chunks needn't match `where`
    ("replace v1 with v2" doesn't). Retrying it is safe because the ids are deterministic.
  - **Every field in a filter must belong to the index's schema, checked before any chunk is
    read or changed.** Why: `delete(where=not_(Entries.version.eq(2)))` on the chunks index
    would otherwise match nothing inside the `not_`, so everything, and empty the index.
  - **Writes are methods, not statements.** They come from the indexing worker, which has no
    asker, so nothing needs to sit between the caller and the index. Searches are statements
    (section 5) because the session must add the access filter to every one. Revisit if writes
    need a unit of work, a trace or a dry run.
  - **`index.find(where=…)` reads without any access filter**, like a raw database
    connection: only indexing code, tests and the engine may call it, never anything serving a
    user.
- **The in-memory index** (`InMemoryIndex`) is the reference behaviour every backend must match,
  for tests and single-process local runs. It is single-threaded and promises no more than
  Qdrant: no locks, so `replace` isn't made atomic there either (in Qdrant it is two requests).
  Why: the API and the worker are separate processes, so production needs Qdrant anyway, and a
  stronger in-memory index would let tests pass on guarantees Qdrant doesn't give. It stores
  vector values already; ranking by them comes with `Index.search` (build step 3).
- **The Qdrant index** (`QdrantIndex`) passes the same contract tests. Each vector value is one
  of the point's named vectors.
  - **`put` sends one `upsert` request**, not `upload_points`. Why: `upload_points` splits a
    write into batches of 64, so a failure partway leaves part of a version stored. A
    document's chunks fit in one request (Qdrant's default limit is 32 MB). An empty write
    sends nothing, because Qdrant refuses an update with no points.
  - **`replace` writes first, then deletes what's left over:** the points `where` matches,
    minus the ids just written. Why: in the other order, a reader between the two requests
    finds none of the chunks; this way they may briefly see old and new together, never
    neither. A failure between the two leaves both stored, and repeating the call finishes it.
    The server's version switch doesn't rely on this: it splits `put` and `delete` around a
    commit ("Currency" below).
  - **`find` reads every page** of `scroll`, 1,000 points at a time. Why: one page would
    silently drop the chunks of a large document.
  - **A point with an integer id is refused when read** (`SchemaMismatchError`). Qdrant allows
    integer ids, but this index only writes UUIDs, so such a point was written by something
    else.
- **Stages that need meaning take fields as settings:**
  `expand(neighbours, order_by=KbChunks.position, within=KbChunks.document_id)`,
  `prune(dedupe, by=KbChunks.document_id)`. The compile step checks that those fields exist
  and have the right types.
- A **standard schema preset** (document id, position, headings) is offered as a convenience,
  never as a requirement. Why: without one, every app would rebuild the basics.

### Vectors

```python
class KbChunks(Schema):
    document_id = One(UUID, indexed=True)
    ...
    dense = Dense(size=1024, distance="cosine")


# writing: the caller embeds, then pairs each chunk with its values
index.put([EmbeddedChunk(chunk=chunk, vector_values=[KbChunks.dense.value(numbers)])])

# searching
hits = index.search(KbChunks.dense.query(question_numbers), where=access, limit=20)
```

- **Vectors are declared on the schema, beside its fields.** `Field` and `Vector` share a
  `Declaration` base, which takes its name from the class body (`__set_name__`). So one schema
  can't have a field and a vector with the same name. The declaration also carries what the
  backend needs to create the collection, such as `Dense`'s size and distance.
- **Vectors are named from the start (`dense`), even while there is only one.** Why: Qdrant
  keeps named and unnamed vectors in different layouts. Adding a second vector to an unnamed
  collection would mean re-indexing.
- **The caller embeds; the index only stores and compares.** Why:
  - the embedder serves both writes and searches;
  - the index stays testable without a model;
  - a backend never has to choose a model.
- **Values and queries are checked where they enter:** `Dense.value(numbers)` for writes and
  `Dense.query(numbers)` for searches. Each checks:
  - exactly `size` numbers;
  - real numbers only, with `bool` refused;
  - every number finite.

  For `"cosine"`, the caller needn't normalise: the backend does it.
- **`EmbeddedChunk(chunk, vector_values)` is what every write takes.**
  - Two values for vectors with one name are refused when it's built.
  - `check_fits(schema)` compares fields and vectors as objects, and raises one
    `SchemaMismatchError` naming the chunk and every extra or missing field and vector.
  - A value's size isn't checked again there: it was checked against its own vector when built.
- **Reads return the `Chunk` alone** (`find`, and `Hit` in search). Why:
  - vectors are large;
  - backends don't give them back as written: Qdrant stores float32 and normalises cosine
    vectors, so `[3, 4]` reads back as `[0.6, 0.8]`;
  - no caller needs them.

  Planned changes that need neither re-embedding nor reading vectors back:
  - `update_fields(*, where, values)` (Qdrant's `set_payload`) for `sync_attributes` (server
    build step 7), so a changed title or group isn't re-embedded;
  - later perhaps `export(where)`, for copying a collection.
- **Kinds:** only `Dense` exists. More kinds become further `Vector` subclasses when a source
  needs them:
  - `Sparse`: `Mapping[int, float]`, with a `modifier` (`"idf"` for BM25-style weights,
    `"none"` for learned ones such as bge-m3's);
  - `MultiDense`: many rows per chunk, one per token, compared by max-sim.

  Each is a sibling of `Dense`, not a subclass. Why: `isinstance(vector, Dense)` would
  otherwise match a `MultiDense` as well, and the order of the `case`s would decide
  correctness. Each kind adds a `…Value` and a `…Query` to the closed unions `VectorValue` and
  `VectorQuery`, so mypy points at every backend that hasn't learned it.
- **Max-sim scores grow with the question's length,** so they aren't on any dense score's
  scale. The executor merges sources by rank, and a score threshold can't be applied to a
  multi-vector source as it is.
- **Adding, dropping or changing a vector means a new collection:** Qdrant can't change a
  collection's vectors in place.

**In Qdrant:** each vector value is one of the point's named vectors. Qdrant has two slots:

| Kind | Slot | Settings compared at `open` |
|---|---|---|
| `Dense` | `vectors_config` | size, distance, one row |
| `MultiDense` (later) | `vectors_config`, with `multivector_config` | size, distance, many rows (max-sim) |
| `Sparse` (later) | `sparse_vectors_config` | modifier (none read back as `none`) |

- **`open` refuses a collection whose vectors differ from the schema's.** It lists every
  difference in one `SchemaMismatchError`, whose header gives the fix: re-index into a new
  collection. It also creates nothing, so a refused collection gets no payload indexes. It
  catches:
  - a vector missing, as in a collection created before vectors existed;
  - a vector the schema doesn't declare;
  - a vector with other settings;
  - a collection holding one unnamed vector.
- **An extra vector is refused, not ignored.** It's treated like an unknown `written_fields`
  entry, not like an extra payload index. Why: Qdrant accepts a point that lacks one of the
  collection's named vectors, and a search by that vector silently skips the point. Older code
  writing into a newer collection would quietly lose recall. An extra payload index, by
  contrast, only changes speed.
- **Settings are compared by describing each side, then comparing the descriptions.** Why:
  - the message can't disagree with the check;
  - adding a setting is one line;
  - defaults the server fills in (`on_disk`, `hnsw_config`, …) never enter the description,
    so they can't cause a false mismatch.

  The rule that comes with it: a setting the description leaves out isn't compared.
- **What `open` doesn't check:** points that lack a vector. The index's own writes can't
  produce them, since `check_fits` demands every vector, and a missing vector costs recall,
  not access. If that's ever needed, Qdrant's `HasVectorCondition` can count them.

### Embedders

```python
# the app's adapter: plain data in and out, no salemdex_retrieval types
class TeiDenseEmbedder:
    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...


# writing: a binding pairs a schema vector with the embedder that fills it
embedded = embed_chunks(
    chunks, bindings=[DenseBinding(vector=KbChunks.dense, embedder=TeiDenseEmbedder(…))]
)
index.put(embedded)

# searching: the retrieve stage embeds the question through the same binding, with
# embed_queries and Dense.query
```

- **An embedder is a port; the app plugs in an adapter** (ports and adapters). Models are often
  hosted elsewhere (TEI, a vendor API) or loaded locally, and the framework never creates or
  configures them. Why: otherwise the framework would have to know how every model is built.
- **The protocol speaks plain data:** strings in, rows of numbers out. No `Dense`, `DenseValue`
  or schema. Why: an adapter is then a thin client over a service, written without knowing the
  framework.
- **The protocol asks for as little as the framework needs:** `Sequence[str]` in,
  `Sequence[Sequence[float]]` out. Why: a protocol's return type is a demand on adapters, not a
  promise to callers, so "return concrete types" applies from the other side. The framework
  only needs `len` (the row check) and iteration (each row is copied by `Dense.value`). Each
  adapter still declares its own concrete type (`list[list[float]]`), which fits, since return
  types are covariant. `Sequence`, not `Iterable`, for the input: the row check needs
  `len(texts)`, and an adapter splitting a batch may go over it twice. A numpy array isn't a
  `Sequence` for mypy, so a local adapter returns `.tolist()`.
- **The framework pairs each embedder with its vector through a binding:**
  `DenseBinding(vector=…, embedder=…)`. `embed_chunks(chunks, *, bindings)` fills one vector
  per binding, and the retrieve stage uses the same binding for searches. The numbers are
  wrapped with `Dense.value` / `Dense.query`. Why:
  - the checks on the numbers (size, finite) run where they enter, whoever wrote the adapter;
  - which model fills which vector is stated in one place;
  - a binding ties the kinds together: a `Dense` vector only takes a `DenseEmbedder`. A
    `Mapping[Vector, Embedder]` couldn't, since a mapping can't tie each key's type to its
    value's.
- **`type VectorBinding = DenseBinding` is a closed union** (later `| SparseBinding`), matched
  with `assert_never` like `VectorValue`. The signatures take the whole union now, and each
  `match` handles the kinds that exist. Why not a union of embedder protocols: protocols are
  matched by structure, so `isinstance` (even with `@runtime_checkable`) checks method names
  only. A dense and a sparse embedder share their method names, so a `match` couldn't tell
  them apart, and each branch would need a `cast`. The vector classes are nominal, so the
  binding is where the kinds are told apart.
- **The embedder sees every chunk of a call in one batch,** once per binding. No chunks call no
  embedder (some services refuse an empty batch). An embedder returning a different number of
  rows than texts raises a `ValueError` naming the vector: rows would otherwise pair with the
  wrong chunks.
- **Two bindings for one vector name are refused before anything is embedded.** Why:
  `EmbeddedChunk` refuses them too, but only after every model call has been paid for.
- **No bindings are allowed,** giving chunks without vectors. Why: a schema may declare no
  vectors, and `embed_chunks` doesn't know the schema, so it can't tell "forgot the binding"
  from "nothing to bind". The check belongs where the schema is known: `put` runs
  `check_fits`, which refuses a chunk missing a vector before anything is stored. (Unlike an
  empty `and_()`, nothing dangerous slips through: an empty AND would match everything with no
  later check.)
- **`embed_chunks` embeds `embedded_text`, never `text`.** Why: it takes chunks, not strings, so
  no caller can pick the wrong one, and the headings context lives only in `embedded_text`.
- **Two operations, by role: queries and passages.** Why: many models encode them differently
  (e5's `query: ` and `passage: ` prefixes), and a question sent through the passage path ranks
  worse without any error.
- **Both take batches:** `embed_queries(texts)` and `embed_passages(texts)`, one row per text,
  in order. Why: with a remote model a round trip costs more than the work. The worker sends a
  document's chunks in one call, and query transforms (multi-query, decomposition) produce
  several questions at once.
- **One protocol per vector kind, named after it:** `DenseEmbedder` now, a sparse one
  (`Mapping[int, float]` per text) when a source needs it. Why:
  - each kind's output has its own shape, so a sparse embedder could never implement the dense
    protocol, and a plain `Embedder` would claim a generality it lacks;
  - services serve the kinds from separate endpoints (TEI's `/embed` and `/embed_sparse`);
  - adapters live in apps, so renaming `Embedder` once sparse arrives would break them all.
- **No identity in the protocol.** Why:
  - the framework can't verify it: for a hosted model, it's a string the adapter reports about
    itself, so a check on it would look like a guarantee without being one;
  - a model mismatch breaks ranking, not one of the three guarantees (section 2). The framework
    enforces the guarantees; quality is measured;
  - the app chooses the model in its config and already records it per version (the server's
    `provenance["index"]["embedder"]`).

  The framework checks what it can verify: the size and finite numbers (`Dense.value`,
  `Dense.query`). What that misses: another model with the same size (bge-m3 and
  multilingual-e5-large both give 1024 numbers), which silently ruins ranking until a re-index.
  **The app owns that check:** the server compares the configured embedder with the one recorded
  for the indexed versions (build step 4).

  Rejected: an `identity` on the protocol, stored in the collection and compared at `open`. It
  rests on the adapter's word, and it ties model configuration to the framework.
- **Synchronous.** Why:
  - the server is synchronous (`def` routes, synchronous SQLAlchemy), and the worker's embedding
    is CPU work, which async doesn't speed up;
  - synchronous code can be called from anywhere (async code uses `asyncio.to_thread`), async
    code only from inside an event loop (in Jupyter, the running loop makes `asyncio.run` fail);
  - an async protocol would make every adapter async, wrappers of synchronous-only libraries
    included;
  - on an event loop, one hidden blocking call stalls every request; in a thread pool it stalls
    only its own.

  The cost for agents is small (estimated, not yet measured): LLM calls dominate each round, and
  independent statements run in a thread pool as concurrently as with `asyncio.gather`. Two gaps
  remain:
  - inside one statement, sources run one after another (roughly 50–100 ms per extra source). A
    thread pool inside the executor can remove that without changing any API;
  - a thread can't be cancelled: a controller's timeout stops waiting, but the statement runs to
    its end. The executor checks a deadline between stages, and adapters set timeouts on their
    HTTP clients. An abandoned statement only reads, so only work is lost.

  **When to add async:** agents run many statements in parallel, or the trace shows fan-out
  taking a large share of a search. Then add async protocols (`AsyncDenseEmbedder`,
  `AsyncIndex`) and an async executor beside the synchronous ones. Statements and stages are
  data, so they stay unchanged.

### Filter expressions

- **AND / OR / NOT over schema fields and chunk ids.** One mechanism for every filter. Why:
  today's fixed fields plus `attributes` can't OR across fields.
- **Methods are the API:** `KbChunks.document_id.eq(x)`, `any_of(…)`. Why: they read plainly and
  type-check under strict mypy. Operator overloading (`==`, `|`) may come later as sugar over
  the methods, only if they prove ugly.
- **Combinators are functions, the nodes frozen dataclasses:** `and_(…)`, `or_(…)`, `not_(…)`
  build `And`, `Or`, `Not` (trailing underscores because `and`, `or`, `not` are keywords, as in
  SQLAlchemy). `type Filter = Equals[Any] | AnyOf[Any] | And | Or | Not | Everything`
  refers to itself.
  - **The checks live in the nodes' `__post_init__`**, so a node built directly is checked too.
    The functions only turn their arguments into a tuple.
  - **`and_` and `or_` need at least one filter.** Why: an empty AND matches everything, so
    `delete(where=and_(*conditions))` with an empty list would empty the index (the old
    `ChunkFilter()` problem). One filter is allowed: it's harmless, and lists of conditions
    vary in length.
  - **`Everything()` matches every chunk, and is accepted wherever a filter is,** writes and
    combinators included, like SQL with no `WHERE`: `delete(where=Everything())` empties the
    index as `DELETE FROM chunks` empties a table. Why that's safe when an empty `and_()`
    isn't: the danger is emptying the index by accident. An empty `and_()` comes from a list
    of conditions that turned out empty; `Everything()` has to be written out, so the intent
    is clear at the call site. It names no field, so it fits every schema. It's a frozen
    dataclass with no fields, like the other nodes, so two compare equal.
    - Its uses: an app without access control passes it as the session's access filter
      ("The guarantee slots"); tests and indexing code read or clear a whole index with it.
    - In Qdrant it becomes a filter with no conditions, which matches every point, also
      nested (`not_(Everything())` matches none).
    - Rejected: allowing it in reads only, with writes keeping `Filter` (a second filter type
      on every write signature, for a case the call site already makes plain); `where:
      Filter | None` on `search` (`None` meaning "everything" fails open when a variable is
      `None` by accident).
  - **Each argument must be a filter**, checked at run time. Why: fields don't define `__eq__`,
    so `KbChunks.version == 2` is `False`, not an error; filters built at run time arrive as
    `Any`. The check reads its classes from the `Filter` alias, so a new node can't be missed.
- **Each backend compiles the expression tree:** Qdrant into must / should / must_not, the
  in-memory index into a predicate. A construct the backend can't run, or a field outside the
  index's schema, fails at compile time, before any work. The children are compiled first, so
  the whole tree is checked even where `or_` would stop early.
  - **The schema check is shared:** `check_filter` walks the tree once, before either backend
    compiles it, and compares each field with the schema's as an object, not by name.
  - **Two Qdrant conditions don't map one to one.** Floats and datetimes match by a range
    from the value to itself. Why: Qdrant can't match a float exactly, and a datetime stored
    as a string would miss the same instant written with another offset. An `any_of` with no
    values becomes a filter matching nothing. Why: Qdrant reads an empty `should` as no
    constraint and matches every point.
- Not yet: filtering by chunk ids (first caller: re-checking earlier candidates, step 3) and
  `in_` on `One` (first caller: scoping a search to collections or documents).

### Generated entries

An entry's text isn't the source's words, so it can't be a chunk's `text`. Entries get their
own schema with a field that points to chunk ids. A source stage searches the entries and
returns the chunks they lead to, recording "found through" in each candidate's history. They
become a separate source, not a special case inside the index.

This is a **resolving source**: it searches one index (the entries) and returns chunks of
another (its *target*, the chunks).

- **It resolves through the target index's `find`, with the access filter.** What it returns is
  then checked on the chunks' own schema, whatever the entries hold. Validity runs on the
  resolved chunks as usual.
- **It keeps one entry per chunk, the best ranked.** Several entries can lead to one chunk, and
  RRF would otherwise count it several times in one list.
- **Its ranks are given after resolution,** counting only the chunks it returns. Why: a gap in
  the ranks would reveal that something hidden ranked above.

### Several indexes

**A `Retrieve` searches one index, for now.** Its sources may search different vectors of it,
never different indexes. Why: a candidate is identified by its chunk id, which is only unique
within one index, and one index means one schema, so one access filter fits every source.
When resolving sources arrive, the rule becomes "every source returns chunks of the same
target index", and the chunk id stays a sound merge key.

Patterns that use several indexes, and where each belongs:

| Pattern | What it does | Where it goes |
|---|---|---|
| Multi-representation indexing (generated entries: questions, summaries, keywords) | Searches another representation of the same chunks, fused with the chunk sources | A resolving source in the same `Retrieve` |
| Two embedding models side by side (a model migration) | Compares or fuses old and new vectors | Two vectors of one index, as now |
| Hierarchical, coarse-to-fine retrieval | Finds the top documents (summaries, titles), then searches chunks only within them | A step on `Question` before retrieve, producing a scope (`document_id ∈ top N`), not candidates |
| Small-to-big, parent document, neighbours | Fetches each candidate's parent section or adjacent chunks | `expand`, with the validity gate after it |
| Citation following | Follows a references field to the cited articles ("theo quy định tại Điều 5 …") | An `expand` variant over a field |
| Pseudo-relevance feedback, iterative retrieval | Reads the top hits and forms a better query | The controller, running two statements (section 8) |

**Rejected: federated search,** fusing indexes that hold different things (regulations and an
FAQ) in one `Retrieve`. Different schemas mean different access filters, a different meaning
of validity and different fields on passages. Corpora go in one index with a `collection`
field, or run as separate statements that the app shows separately.

### The guarantee slots

The executor can't enforce access or currency on data it doesn't understand. So the frame has
fixed **slots**, and the app fills them:

```text
framework (generic)                         app (server) fills
─────────────────────────────────────────   ─────────────────────────────────────
schema: declared fields, types, indexes  ← document_id, version, groups, collection…
chunk = id + text + field values
filter expressions over fields
access slot:   a filter, given per session ← "shares a group OR manages the collection"
validity slot: chunks → ids to keep        ← "document active, last indexed version"
stages, executor, plan, gates
```

- **Every slot must be filled.** An engine can't be built without an access policy. An app that
  wants no access control says so with an explicit `access=Everything()`, so the absence is
  visible. Why no separate `AllowEverything`: `Everything` is already a `Filter`, so the
  executor passes it to every source unchanged, and one idea keeps one word.
- The executor still decides *where* the slots apply: the access filter goes into every source
  and expand stage, and the validity gate runs where candidates enter and again before `Take`.
- **What access means is the app's design, not the framework's.** The framework only requires
  the access slot to be a filter expression. How the app builds it (which fields,
  combinators or a denormalised field) is decided in the app.

### Filters, gates, validity and cuts

One word per idea, since all four narrow the results:

| Word | Means | Applied |
|---|---|---|
| **filter** | a `Filter` expression over chunk fields, evaluated by the index; the access filter is one | inside the index, before ranking |
| **gate** | a point in the executor where every candidate must pass a check, whatever the statement says | in code, after search |
| **validity** | the check a gate runs: should anyone be served this chunk now? Its truth lives outside the index | at each gate |
| **cut** | keeping the top N by rank, judging no content (`depth`, `limit`) | in each stage |

- **Access is a filter, validity isn't.** Access's truth is on the chunks (their groups) and
  must apply before ranking, so `limit` fills with visible chunks and hidden ones never leave
  the index. Validity's truth is in the app's database (document active, `indexed_version`)
  and changes without the chunks being rewritten: the version switch is one row. As a filter
  it would list every document's served version, or need a flag on every chunk, rewritten at
  each switch, which would lose the one-row switch. Stale chunks are few and short-lived, so
  checking after search costs little recall.
- **A condition whose truth is kept on the chunks belongs in a filter,** not in validity: a
  filter applies before ranking. A regulation's legal effect (in force, replaced) is such a
  field: a filter ("in force only") or a marker on the passage, the app's choice, never hidden
  silently by validity.
- **Later:** caller filters (a collection, "in force only") are `Filter` expressions the executor
  ANDs with access; prunes (thresholds, dedupe) are stages that cut by score. Neither is a
  gate: a gate always runs, and a statement can leave a stage out.

### How the slots plug in

**Validity: a port the engine holds.**

```text
engine = Engine(validity=ServerValidity(session_factory))   # Protocol: chunks → ids to keep
```

- It is part of the frame, not a stage the app writes into statements. Why: a statement author
  or an agent could leave a stage out, and stale results would pass.
- It doesn't depend on the asker ("is the document active, is this its indexed version?"), so
  it belongs to the long-lived engine.
- **The port: `valid_ids(chunks: Sequence[Chunk]) -> Set[UUID]`, the ids to keep.** A port
  (ports and adapters): the package defines the shape, the server's adapter queries Postgres,
  and the package never imports a database. Why this shape:
  - **It can only narrow.** The executor keeps a candidate only if its id is returned, so an
    adapter can't add, swap or reorder chunks, and extra ids are ignored.
  - **It fails closed.** A buggy adapter returning nothing serves nothing, which is noticed at
    once. Returning the ids to drop would fail open: stale and deleted chunks served.
  - **It's batched:** one call, so one query, per gate. Never one per chunk (the N+1 problem).
  - **It takes chunks,** so the adapter reads its own schema's fields (`document_id`,
    `version`); the package needn't know which fields identify a version. Not candidates:
    the app never imports engine internals.
  - Rejected: returning chunks (could widen), one flag per chunk (positional, so a misaligned
    list keeps the wrong chunk), one chunk per call (N+1), verdicts with reasons (no need yet).
  - Synchronous for now; an async port comes with the async executor.
- **Where the executor runs it:**
  - **where candidates enter:** after `Retrieve` (later after `expand` too), so no stale chunk
    costs a reranker call and every later stage works on valid candidates. After a retrieve with
    several sources, the chunks of every list go in one call, then each list is filtered: the
    ranks fusion reads stay the sources' own.
  - **again inside `Take`, before the cut:** a safety net, since a document can be deleted or
    its version switched while a reranker runs (a second or more on CPU), and a future stage
    could add candidates without its gate. It costs one query, on the candidates left.
    `Take` cuts from what passes, so `limit` still fills when it can, and it is the one place
    passages are built: "only after the final gate" is then true in the code.
  - An app with nothing to check passes an explicit `AlwaysValid()`.

**Access: the filter is passed when a session opens.**

```text
access = documents.access_filter(principal)     # server service, with its own DB session
session = engine.session(access=access)
```

- Why this over a policy the engine holds (asker → filter): it is simpler. The framework has no
  asker type and no per-request plumbing. In the server, building the filter needs a DB lookup
  (the principal doesn't carry its groups or managed collections), and services already check
  permissions with a DB session.
- The cost: the guarantee rests on a convention, not on structure. A call site that passes a
  filter that's too broad breaks it. Kept robust by:
  - **one server service opens sessions**; `/search`, `/answer` and agents all go through it;
  - a test of the filter that service builds for each kind of asker;
  - an import-boundary check in CI: only that module imports `Engine`.
- **When to switch to an engine-held policy:** when agents or other entry points need to open
  sessions themselves. The engine sees a filter either way, so the switch stays cheap.

**One filter for every schema: an access preset.** A session holds one filter, but a statement
may search indexes with different schemas (chunks and generated entries), and a filter's
fields are checked against each schema by identity. So the access fields live in a schema
preset that every searchable schema inherits, and the app builds the filter over the
preset's fields:

```text
class Restricted(Schema):  access_group_ids = Many(UUID, indexed=True)
class KbChunks(Restricted): …
class KbEntries(Restricted): …
access = Restricted.access_group_ids.any_of(groups)   # valid on both schemas
```

- An inherited declaration is the same object in every subclass, so the existing identity
  check accepts it on both, and refuses it on a schema without the preset: it fails closed.
- The executor checks the filter against every source's schema before any model is called.
- The cost: entries copy their chunk's access fields when indexed, so a change to a document's
  groups updates its chunks and its entries. A missed update can't leak a passage: the
  resolving source fetches chunks through the chunks index with the filter. On the chunks the
  filter is the guarantee; on the entries it keeps recall (hidden entries don't fill the
  depth) and closes the rank side channel.
- Rejected: a filter per schema on the session (one rule built twice, which can drift; kept as
  the fallback for a schema whose access really uses other fields), a schema → filter function
  (the same, harder to inspect), entries stored in the chunks index with a `kind` field (an
  entry's text isn't the source's words, and every chunk search would need `kind = chunk`).

### Currency: show the last indexed version

How the server fills the validity slot. While a new version waits to be indexed, search keeps
showing the last indexed version. Why: hiding the document would make it disappear during every
re-index (minutes on CPU), and for good if the new version fails to index.

1. **The database records which version is in the index:** `documents.indexed_version` (one
   migration). The worker sets it in the transaction that marks the index job done. The validity
   check keeps a hit when the document is active and the hit's version is `indexed_version`, not
   `current_version`.
2. **Chunk ids include the version** (section 6, "The chunk"), so v2's chunks never overwrite
   v1's.
3. **The write is split around the commit.** This is the pointer-flip pattern: write the new
   copy, switch one pointer, remove the old copy.

   ```text
   handler   index.put(v2 chunks)                     v1 still served: indexed_version = 1
   worker    commit indexed_version = 2                the switch: one row, atomic
   after     index.delete(where=document is x AND version isn't 2)
   ```

   - From a reader's view the switch is atomic: all of v1, then all of v2, never a mix and never
     nothing.
   - A crash after the commit leaves v1's chunks behind. The validity check already hides them,
     and `gc` removes them. It's the same idea as "originals are deleted after the commit".

Consequences:

- **Access follows the document, not the version.** Old chunks stay in the index longer. A change
  to the document's groups must update every chunk of the document, whichever version
  (`where=KbChunks.document_id.eq(x)`). Otherwise the old version would be served with stale
  groups: a leak.
- **A failed new version keeps the old one served indefinitely.** That's acceptable, but it must
  not be silent:
  - every passage carries its version, so the API can mark a hit "a newer version is being
    processed" (`current_version > indexed_version`);
  - the failure is visible on the document (operations, build step 8).

## 7. Validation

| When | What | Why |
|---|---|---|
| Before any work ("compile") | stage order, by replaying the stages through the steps; one index per retrieve; the access filter fits every source's schema | nothing is spent on a statement that would fail |
| During the run | empty results, wrong score counts, a tool failing | only known as it runs |

mypy covers statements written in code. Statements built at run time (from config, or by an
agent) still need the compile check, so the executor always replays the statement (section 5)
and runs the replayed one. Limits aren't compared with depths: each is "at most" for its own
stage.

## 8. Loops

Loops are common in advanced retrieval (retrieve → grade → rewrite → retrieve again: CRAG,
Self-RAG, iterative retrieval). **Statements stay acyclic. A controller above the executor owns
the loop.**

```text
controller (the loop)       run → look at the candidates → next statement, or stop
  └─ statement (acyclic)    validated as a whole, finite, the same gates every round
```

- Every round stays checkable. A cyclic graph can't be fully validated up front.
- Termination is explicit: the controller owns the limits (most rounds, time or cost budget).
- The gates hold every round. The controller re-checks the final output, because a document can
  be deleted mid-loop.
- Possible later: a declarative `repeat(body, until=…, max_rounds=…)` that compiles to the
  controller. Its body is still an acyclic statement.

An agent step continues from earlier results, e.g. `from_candidates(previous).expand(...)`.
These still go through the same executor.

## 9. Agents that recommend decisions

Retrieval is the agent's evidence layer. The guarantees carry up to the recommendation unchanged.

```text
decision agent (controller)
  1. frame     the decision → what must be known (options, constraints, criteria)
  2. gather    one statement per need  ──→ a session with the user's access
  3. weigh     options against evidence (the LLM's work)
  4. check     is every claim backed by a passage? does any evidence conflict?
  5. answer    recommendation + cited evidence, or "not enough evidence"
```

| Concern | Provided by |
|---|---|
| No recommendation rests on, or hints at, hidden knowledge | every statement runs in a session with the user's access |
| No deciding on outdated knowledge | the validity gate on every statement |
| "Why did it recommend this?" | passages with origin; statements logged and replayable |
| Coverage: every constraint that applies, counter-evidence | different statements per need ("supports A", "argues against A", "everything in scope") |
| Knowing when not to recommend | scores and gates, so the agent can abstain |
| Is it improving? | fixed statements per step, evaluated one step at a time |

The framework doesn't make step 3 correct. It guarantees the reasoning starts from permitted,
current, traceable evidence. Step 4 checks the output against that evidence, and that check is
where most of the remaining safety comes from.

## 10. Scope of the first version

Build only the stage kinds that basic search needs: **retrieve, merge, rerank, take**, with the
filter expression and the executor's frame. Add rewrite, expand, prune and the controller when
something uses them. Why: the type-state structure is the costly part to get right; adding
stage kinds afterwards is cheap.

**Package:** the engine is a new uv workspace member in this repo: the distribution
`salemdex-retrieval`, imported as `salemdex_retrieval`. Why not a namespace package
(`salemdex.retrieval`): it needs special care in hatch and mypy, and no member could have a
`salemdex/__init__.py`. Pydantic does the same with `pydantic_core`.

- **Folder:** `packages/retrieval/` at the repo root. Libraries go in `packages/`, deployables
  (server, CLI) in `apps/`.
- **Scope: storing and finding chunks.**

  | In `salemdex_retrieval` | Why here |
  |---|---|
  | schema, chunks, filter expressions | the data being retrieved and the language for selecting it |
  | index backends, reads and writes (`put`, `replace`, `delete`) | the index is what retrieval reads; writing it is the other half (as SQLAlchemy has `insert` beside `select`) |
  | embedders, rerankers | used by both writes and searches |
  | statements, stages, executor, session, validity | retrieval itself |

  | In a later package (e.g. `salemdex_agents`) | Why separate |
  |---|---|
  | the controller, agent loops, LLM tools | the controller sits above the executor (section 8); a package boundary makes that layering real |
  | answer generation, LLM providers | separate dependencies; retrieval stays usable without an LLM |

  Why the split also helps safety: an agents package uses retrieval's public API, where
  the only way to run a statement is a session with the user's access. The boundary
  doesn't enforce it fully, though: a `Source` holds its `Index`, whose `find` and `search`
  apply no access filter of their own (section 11).

- It never imports `document_processor`. An import-boundary check in CI enforces it. Why: the
  engine is generic; turning `chunking.Chunk` into an index chunk (today's `ingest` and
  `address_keys`) is app code, in the server, which declares the schema anyway.
- Its extras hold the backends and models: `qdrant`, `huggingface` (embedder, reranker).
- A separate repo waits until the engine needs its own releases.
- The family follows later, one rename per commit: `salemdex-ingest` (today's
  `document_processor`, moving to `packages/ingest/`) and `salemdex-server` (today's
  `kb_server`). The GitHub repo is already `salemdex`.

**Folder layout.** The core modules stay at the top level. Each new kind of part gets a
subfolder when it first arrives.

```text
src/salemdex_retrieval/
  __init__.py              the top level's names ("Import paths" below)
  declarations.py  errors.py  schema.py  filters.py                       core
  checks.py                checks on plain values: check_positive_int, to_finite_float
  vectors.py               Dense, VectorValue, VectorQuery (closed unions)
  chunks.py                Chunk, EmbeddedChunk, Hit (chunk, score)

  index/                   storing and finding chunks
    __init__.py            Index, InMemoryIndex
    protocol.py            Index, check_batch, check_filter, check_query, check_limit
    in_memory.py           search: filter first, then score every stored value
    qdrant/
      index.py             search: query_points(using=…, query_filter=…, limit=…)
      filters.py  payload.py
      vector_config.py     when Sparse arrives: _VectorConfigs and the describers

  embedders/               used by both writes and searches
    __init__.py
    protocol.py            DenseEmbedder, DenseBinding, VectorBinding, and
                           embed_chunks(chunks, *, bindings)
    fake.py                a deterministic embedder for tests, the server's too
    huggingface.py         a local adapter (bge-m3), behind the `huggingface` extra

  engine/                  retrieval itself (build step 3)
    statements.py          search(question) and the type-state steps:
                           Question → CandidateLists → Candidates → Search
    stages.py              retrieve (named sources), merge (rrf), rerank, take, as data
    results.py             Candidate, the trail entries (Found, Fused, Reranked),
                           Passage
    executor.py            compiles a statement and runs it: adds the access filter to every
                           source, the validity gate where candidates enter and before Take
    session.py             Engine, Session(access=…)
    validity.py            the Validity port, AlwaysValid

  rerankers/               with the rerank stage
    protocol.py  fake.py  huggingface.py

tests/                     mirrors the source folders, each with its own __init__.py
  index/                   test_contract.py holds the search cases too
  embedders/               test_embed_chunks.py
  engine/                  test_statements.py  test_executor.py  test_session.py
  test_public_api.py       the top level's __all__, written out in full
```

- **`engine/`, not `search/`.** "Search" would otherwise name three things:
  `index.search(query, where=…)` (one source, raw hits), the package, and the `search(question)`
  builder, so a reader couldn't tell from the call site which one runs.
  - `Index.search` keeps the word every vector store uses for a single-source read.
  - The package is named after the `Engine` it exports.
  - `search(question)` reads as "a search for this question", which is what a statement is.
- **`Hit` lives in `chunks.py`, beside `EmbeddedChunk`.** One wraps a chunk on its way into the
  index, the other on its way out. The executor then imports it from the core, not from
  `index/`, to start each candidate's trail.
- **`embed_chunks` sits beside the embedder protocols,** as `check_batch` sits beside `Index`:
  it's the one helper every caller of the protocol shares.
- **The fake embedder lives in `src`,** as `InMemoryIndex` does. Why: the server's tests need
  one without loading torch.
- **Backends and models sit behind extras** (`qdrant`, `huggingface`). No package
  `__init__.py` imports them.
- **`engine/` modules depend only on those before them here:** stages, statements, results,
  validity, executor, session.
- **`validity.py`, not `gates.py`:** the module holds the check, not the place it runs. A
  *gate* is where the executor runs it; "the validity gate" stays a word in the executor's
  docstrings, with no module of its own.

**Import paths (built).** Each public name has exactly one path to import it from. The
question isn't what's public, but which path is the one to use:

| Tier | Holds | Names |
|---|---|---|
| top level | what callers build or call: the vocabulary at many call sites | `Schema`, `One`, `Many`, `Dense`, `Chunk`, `EmbeddedChunk`, `and_`, `or_`, `not_`, `Everything`, `search`, `Source`, `Rrf`, `Engine` |
| a core module, or a subpackage's `__init__` | what callers receive, annotate, catch or implement | `errors.SchemaMismatchError`, `filters.Filter`, `chunks.Hit`, `index.Index`, `index.InMemoryIndex`, `embedders.DenseEmbedder`, `embedders.VectorBinding`, `rerankers.Reranker`, `engine.Session`, `engine.Passage`, `engine.TrailEntry`, `engine.Validity`, `engine.AlwaysValid` |
| a module of its own | what needs an extra, or is only for tests | `index.qdrant.QdrantIndex`, `embedders.fake.FakeDenseEmbedder`, `rerankers.fake.FakeReranker` |
| not exported | what only the package uses | `execute`, `replay`, `Candidate`, the stage classes, the intermediate states, `score_texts`, the `check_*` helpers |

- **Why not everything at the top:** where a name lives tells the reader what it is.
  `errors.SchemaMismatchError` says more than a bare name, and the protocols adapter
  authors implement shouldn't crowd the namespace every app reads.
- **Why not only the deep modules:** paths like `engine.statements` would become public by
  use, and moving a module would break callers. Deep modules (`engine/statements.py`,
  `index/in_memory.py`) are never an import path; they can move freely.
- **Fakes stay in their own module,** not in their subpackage's `__init__`: the import line
  then says it's test code, and `.fake import` outside `tests/` is easy to find.
  `InMemoryIndex` isn't a fake: it's a real backend for single-process runs and the
  reference behaviour, so `index` exports it.
- **`execute` is exported nowhere,** so `Session.execute` is the one public way to run a
  statement, and no search runs without an access filter.
- **`AlwaysValid` isn't at the top,** though it's built: only apps with nothing to check
  and tests use it, and the top level shouldn't suggest it's the usual choice.
- **`tests/test_public_api.py` pins the top level's `__all__`,** written out in full, as
  the golden payload test does: a name joins or leaves the top level only by a change a
  reviewer sees.

**`Index.search` (built):**
`index.search(query, *, where, limit) -> list[Hit]`.

- **`query` is a `VectorQuery`:** `DenseQuery` now, later `| SparseQuery | MultiDenseQuery`.
  - It carries its vector, so the backend knows which named vector to search.
  - Its numbers were checked where they entered (`Dense.query`).
  - The union is closed, so each backend `match`es on it and ends with `assert_never`. mypy
    then points at every backend that hasn't learned a new kind.
- **One query per call.** Several sources are the executor's job: it calls `search` once per
  source and merges by rank.
- **`check_query(query, *, schema)`** in `index/protocol.py`: the query's vector must be the
  schema's own, compared as an object, as `check_filter` compares fields. Another schema's
  vector with the same name raises `SchemaMismatchError`. Both call
  `schema.check_declares(field_or_vector)`, which a statement's sources use too.
- **`where` is required, with no default.** Every search carries a filter, and a required
  argument can't be forgotten. Searching everything says so: `where=Everything()`.
- **`limit` must be a positive `int`,** checked where it enters (`check_limit`, a
  `ValueError`; a `bool` doesn't count), so no backend has to decide what 0 means.
- **Every backend checks before reading:** `check_limit`, `check_query`, then `check_filter`.
- **Ties come in no promised order.** The in-memory index keeps storage order; Qdrant
  promises nothing, so the contract tests never build a tie.
- **`Hit.score` is comparable only within one search's results:** higher is better for cosine
  and dot, but never compared across vectors or questions. That's why the executor merges by
  rank. A `Hit` holds the chunk without its vectors, as `find` does.
- **The in-memory index scores in plain Python** (`math.sumprod`, `math.hypot`), not numpy.
  Why: the core package has no dependencies, and numpy only arrived through
  `qdrant-client`, so the package failed to import without the `qdrant` extra. A zero vector
  scores 0.0 under cosine, as it does in Qdrant.
- **Qdrant converts by kind in two helpers,** `_to_qdrant_value` (writes) and
  `_to_qdrant_query` (searches), each a `match` ending in `assert_never`, returning
  `_QdrantVectorData`. That alias is `list[float]` for now and widens with each kind
  (`list[list[float]]` for multi-dense, `models.SparseVector` for sparse), so mypy catches a
  case returning the wrong shape.
- **Qdrant's search is exact only while the collection is small.** Below its indexing
  threshold, Qdrant compares every vector, as the in-memory index does, which is why the
  small contract tests can demand exact ranking. Once it builds the HNSW graph, ranking is
  approximate: a restrictive `where` (a reader's few groups) can return fewer than `limit`
  hits or miss the nearest. A payload index on each field the filter uses keeps it accurate,
  since Qdrant can then count the matching points and compare them all when they're few. So
  every field the engine filters every search by, `access_group_ids` above all, must be
  declared `indexed=True` (build step 4).
- **Contract tests:**
  - hits come ranked by score, with scores compared within a tolerance: Qdrant stores float32
    and normalises cosine vectors;
  - `where` applies before ranking: a hidden chunk that scores best never appears, and `limit`
    is still filled by visible chunks;
  - `limit` is respected, and an empty index gives no hits;
  - another schema's vector is refused, and so is a `limit` below 1.

**Build order.** Each layer is tested before the next one depends on it.

1. **Schema, chunk, filter expressions**, plus the in-memory index that translates them. All of
   it is testable without Qdrant. **(done)**
2. **The Qdrant backend:** translating filters, creating payload indexes from the schema, the
   startup check, `put` / `find` / `replace` / `delete`, with the in-memory index's tests run
   against it too. Extract the `Index` protocol the two share. **(done)**
3. **Statements and the executor** with retrieve, merge, rerank and take, plus the session, the
   access filter and the validity check. **(done)**
4. **Port the server's indexing handlers** to the new API, then build search on it. The
   server's schema declares `access_group_ids` and every other field each search filters by
   with `indexed=True` ("`Index.search`" above).

**Migration:** build the new API alongside the old one, port the callers (server indexing
handlers, search, evaluation, tests), then delete the old API.

**What the reranker reads is set by the stages before it,** never hard-coded: each source's
`depth`, and later a merge's limit (section 5). Their defaults are chosen later, by measurement:
latency per candidate count on the target hardware, and hit rate / MRR per depth on the
evaluation set (build step 9). Pick the smallest depths where quality stops improving, within
the waiting-time budget. Why it matters: a cross-encoder scores each question–passage pair
through a large model, so what reaches it trades recall (it can only lift what it sees) against
waiting time and throughput. For the measurement, the executor records each stage's duration in
the session's trace.

## 11. Open questions

- **The standard schema preset:** which fields it holds. The access fields belong in it, or
  in an access preset of their own (section 6, "One filter for every schema").
- **The agent step API** (later, with the controller):
  - How a statement starts from earlier candidates. A likely shape: the LLM sees short handles
    (`p2`), and the controller maps them back; an invented handle doesn't resolve. Earlier
    candidates pass the filter and the validity check again, since documents change between
    rounds.
  - What the controller, and so the LLM, sees after a tool call: passage text and handles, how
    much of the trail, how many passages per round (the context window limits this).
  - Everything the LLM writes is untrusted input. The session's access is what keeps that safe.
- **Keeping agents inside a session.** A `Source` holds its `Index`, and `Index` is
  public, so code holding a statement can call `source.index.find(where=Everything())`
  with no access filter. Python can't hide the attribute, so the package boundary alone
  doesn't keep agents inside a session (section 10). A likely shape, with the agents
  package: agents never hold `Source` objects. Statements name their sources, and the
  engine resolves the names from sources the app registered with it, so an agent gets a
  session and source names, never an index. Until then, the rule rests on code review
  and on agents being given sessions only.
