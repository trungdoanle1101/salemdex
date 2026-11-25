# Retrieval as a graph of stages

Status: build steps 1 to 3 done: the core, both index backends with `Index.search`
(section 10), embedders and rerankers (section 6, "Embedders") with Hugging Face adapters,
and the engine with its prepared statements (sections 5 and 6). Build step 4 is under way
in the server, as part of replacing the old retrieval: its index and search run on the
engine ("Pick up here" below). Branch: `features/retrieval-redesign`.

- Built, in `packages/retrieval` (`salemdex_retrieval`), with tests:
  - `schema.py`: `One` / `Many` fields, `Schema` with presets, `OneValue` / `ManyValues`,
    the value checks;
  - `filters.py`: `Equals` (`eq`), `In` (`in_`, on a `One`), `AnyOf` (`any_of`, on a
    `Many`), `and_` / `or_` / `not_`, and `Everything`, matching every chunk;
    `check_is_filter`, for filters arriving untyped;
  - `vectors.py`: `Dense` and `Sparse` vectors, declared on the schema beside its fields,
    with checked values (`Dense.value`, `Sparse.value`) and queries (`Dense.query`,
    `Sparse.query`); the closed unions `VectorValue` and `VectorQuery` (section 6,
    "Sparse vectors");
  - `chunks.py`: `Chunk`, with typed reads (`chunk[field]`), and
    `Chunk.check_fits(schema)` raising `SchemaMismatchError`;
    `EmbeddedChunk`, a chunk with its vector values, which every write takes; `Hit`, a chunk
    with its score, which search returns;
  - `index/in_memory.py`: `InMemoryIndex` with `put`, `find`, `delete`, `replace` and
    `search` (exact cosine and dot, in plain Python: the core has no dependencies; sparse
    by the dot product over shared ids, with IDF as a Qdrant server computes it), the
    sparse rules tested on every backend in `tests/index/test_sparse_search.py`;
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
  - `embedders/protocol.py`: the `DenseEmbedder` and `SparseEmbedder` protocols,
    `DenseBinding`, `SparseBinding` and the closed union `VectorBinding`, and
    `embed_chunks(chunks, *, bindings)` (section 6, "Embedders"), tested in
    `tests/embedders/test_embed_chunks.py`; the engine embeds questions through either
    binding, and `tests/engine/test_engine.py` runs a hybrid statement (dense and sparse,
    fused by RRF) on an in-memory index;
  - `embedders/fake.py`: `FakeDenseEmbedder(size=…)`, a deterministic embedder by feature
    hashing (signed, NFC and case-folded words), so texts sharing words score higher;
    tested in `tests/embedders/test_fake.py`.
  - `checks.py`: `check_positive_int`, `to_finite_float`, `to_stripped_str`, shared by
    every layer;
  - `rerankers/`: the `Reranker` protocol, `score_texts`, and `FakeReranker`;
  - `engine/stages.py`: `Retrieve` (sources by name, `source(name, depth=…)`), `Merge`
    with `Rrf`, `Rerank` (a reranker by name), `Take` (no limit), as plain data naming
    what the engine registered;
  - `engine/statements.py`: `search()`, the type-state steps ending in `Statement`, and
    `replay`;
  - `engine/sources.py`: `VectorSource(index=…, binding=…)`, what a registered source
    name stands for, refusing a vector its index's schema doesn't declare;
  - `engine/results.py`: the trail entries (`Found`, `Fused`, `Reranked`), `Candidate`
    and `Passage`;
  - `engine/validity.py`: the `Validity` port (`valid_ids(chunks) -> ids to keep`) and
    `AlwaysValid`;
  - `engine/engine.py`: `Engine(validity=…, sources=…, rerankers=…)`, frozen, keeping
    read-only copies of what it registers, and `Engine.prepare(statement)`. Preparing
    replays the statement and checks that every name is registered, each retrieve searches
    one index, and no vector is searched twice. It returns a `PreparedStatement`, which
    only `prepare` builds and which can't be changed. `PreparedStatement.execute(*,
    question, limit, where)` checks its inputs (the filter against every searched schema)
    before any model is called; then it searches every source with `where`, gates where
    candidates enter (one validity call per retrieve) and again inside `Take`, fuses by
    RRF, and reranks every candidate it receives. Tested in `tests/engine/test_engine.py`
    with a scripted index and spies that count embedder, validity and reranker calls.
  - The re-exports, in three tiers (section 10, "Import paths"); `tests/test_public_api.py`
    pins the top level's `__all__`.
  - `chunks.py`: typed reads, `chunk[field]`; `filters.py`: `In`, from `One.in_`.
  - Behind the `huggingface` extra: `embedders/huggingface.py`
    (`HuggingFaceDenseEmbedder`, `encode_query` / `encode_document`, so the model's own
    prompts) and `rerankers/huggingface.py` (`HuggingFaceReranker`, a one-label
    `CrossEncoder`, refusing a classifier at load). Both take the model as Hugging Face
    names it, a hub id or a folder, and keep nothing about it; `_huggingface.py` holds the
    device choice and `load_model`, which raises `errors.ModelLoadError`. Tested in
    `tests/test_huggingface.py` (no model), `tests/rerankers/test_huggingface.py` (a tiny
    model built in the test) and, with a real model set, the embedder's and reranker's
    tests; `tests/test_extras.py` checks no core import loads an extra's library.
- **In progress** (as of 2026-11-25): **hybrid search** (section 6, "Sparse vectors"),
  with evaluation paused at the old/new gate (step 5 below). Built: the `Sparse` vector,
  both indexes, `SparseEmbedder` and `SparseBinding`, and the engine's sparse sources, all
  general, with no model's details. Next:
  1. a fake sparse embedder in `embedders/fake.py`, so the server's tests can run a hybrid
     statement without torch;
  2. the adapters: BM25 in the core, taking a `tokenize` function from the app, and
     `HuggingFaceSparseEmbedder` over sentence-transformers' `SparseEncoder` (SPLADE,
     inference-free models), behind the `huggingface` extra. A bge-m3 adapter only if the
     question set shows it beats them;
  3. in the server: a sparse vector on `KbChunks` (a new collection), a hybrid statement,
     then `evaluator.compare(hybrid, baseline=dense, …)` on the real question set.
- **Pick up here** (as of 2026-11-18): replace the old retrieval in the server, then delete
  it (expand, migrate, contract; section 10, "Migration"). Nothing else in the framework
  imports `document_processor.retrieval` or `integrations.{huggingface,qdrant}`: only the
  server, `scripts/try_search.py`, `scripts/pin_model.py` and their own tests do.
  1. **Index on salemdex** (build step 4):
     - `Guarded` and `KbChunks` (`backend-design.md` section 5), and a pure function
       turning one version's prepared chunks into salemdex `Chunk`s: ids
       `uuid5(document_id, f"{version}:{position}")`, `embedded_text` with the headings
       path. It replaces `to_indexed_chunks` and `index_document`;
     - `run_index_job` on `embed_chunks` and `index.replace` within its version, with the
       version switch (built):
       `documents.indexed_version` (one migration), set in the transaction that marks the
       index job done, then `delete` of the document's other versions after the commit
       (section 6, "Currency");
     - `run_delete_job` on `index.delete(where=KbChunks.document_id.eq(…))`;
     - wiring: `QdrantIndex.open(schema=KbChunks, …)`, whose payload indexes replace
       `QDRANT_INDEXED_ATTRIBUTES` and its test, and `HuggingFaceDenseEmbedder`; settings
       name the model as Hugging Face does, replacing `KB_EMBEDDER_SPEC` and
       `KB_MODELS_FOLDER`;
     - port `test_handlers.py`, `test_worker.py` and the smoke script
       (`scripts/smoke_test.py`, `test_smoke_script.py`) to `InMemoryIndex` and
       `FakeDenseEmbedder`.
  2. **The old preparation sizes in words** with `WordTokenizer` (section 6, "Chunk length
     and the models"), so it needs no model's tokenizer:
     - `KB_CHUNK_TOKENS` becomes `KB_CHUNK_WORDS` (a new name, since the unit changes);
       about 450 words keeps today's size (512 bge-m3 tokens at 1.10 tokens per word);
     - `PreparationSettings` drops the embedder identity, and the index job drops its
       "chunks sized for this embedder" check;
     - the recorded chunker settings name the tokenizer (`"tokenizer": "words"`), so a
       version sized in tokens and one sized in words differ for `reprepare --stale`;
     - `WordTokenizer`'s docstring no longer says "not for production".
  3. **Move the index to a new collection:** `QdrantIndex.open` refuses the old payload
     layout, so the new code writes to a new `KB_QDRANT_COLLECTION` and every active
     document needs an index job. Prepared output is unchanged: only re-embedding.
  4. **Search** (build step 4, server build step 5; built): the validity adapter (a hit is
     kept when its document is active and its version is `indexed_version`, one query per
     gate), `SearchService` holding the statement prepared at startup and passing each
     asker's access filter as `where`, a test of that filter for each kind of asker,
     `POST /search`, and `HuggingFaceReranker` from a reranker setting. Still to do: the
     import-boundary check (only the wiring builds an `Engine`).
  5. **Evaluation** (designed, section 13; next): hit rate and MRR on a labelled question
     set, which only `document_processor.retrieval.evaluation` does today. It must exist
     before the old package goes, or the two can't be compared. The question types
     (`evaluation/questions.py`), the filter matcher (`filters.build_chunk_matcher`), the
     quote judge (`evaluation/judges/`), the quote occurrences
     (`evaluation/occurrences.py`), the runs (`evaluation/runs.py`), `Assessment` and the
     evaluator's pooling (`judge_top_chunks`), the scorecard (`evaluator.score`, its
     metrics with a cut-off, `for_tag` and `tags`) and the comparison
     (`evaluator.compare`) are built; next, the old/new gate on the real question set,
     abstention, then the stage metrics.
  6. **Delete** `document_processor.retrieval`, `integrations/huggingface`,
     `integrations/qdrant`, their tests (`tests/retrieval`,
     `tests/integrations/{huggingface,qdrant}`) and extras, `scripts/try_search.py` (or
     rewrite it on the engine) and `scripts/pin_model.py`. The server's `worker` extra
     becomes `salemdex-retrieval[huggingface,qdrant]`. Update `backend-design.md`
     sections 3 (settings, the `KB_MODELS_FOLDER` paragraph), 8 (the index job, the
     Qdrant indexes paragraph) and 15.

  Decisions needed on the way:
  - **What identifies a model in provenance** (blocks step 1's provenance). The old
    identity held a pinned commit; a hub name alone can't tell an upstream update from the
    indexed model. Passing Hugging Face's own `revision` through is one answer.
  - **How to re-index into the new collection** (step 3): a one-off command now, or
    server build step 8's re-index early.
  - **Reference keys:** the old index stored `reference_keys` (`address_keys` in
    `retrieval/reference.py`), and `ChunkFilter` could filter by them; no server caller
    searches by them, and evaluation grades by quote, so it doesn't need them either
    (section 13). Drop them, or add a `Many(str)` field to `KbChunks` and move
    `address_keys` out of the old package first.
- Decided: evaluation lives in salemdex, and the app supplies the questions and the
  corpus (section 13).
- Decided, built when needed: a limit on `merge` (when fusing deep lists leaves too many to
  rerank), resolving sources for generated entries, a statement-level filter
  (`.where(…)`, ANDed with each run's), `expand`, query transforms, the trace (sections 5
  and 6).
- Later, when a caller needs it: filtering by chunk ids.

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
- Step 2 is the app's: it builds the filter from the asker. The framework requires a filter
  on every run and applies it inside every source, without knowing it holds access
  (section 6, "The guarantee slots").
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
*where* the guarantees apply. The app decides *what* they mean (section 6): to the framework,
"allow" is a filter each run must be given; that it holds access is the app's meaning.

## 5. Statements and prepared statements

**A statement describes a search; the engine prepares it once; the prepared statement runs
it for each question.** Like a prepared statement in SQL, with three fixed slots: the
question, the limit and the filter.

```text
engine   = Engine(validity=validity,
                  sources={"dense": dense, "sparse": sparse},
                  rerankers={"cross_encoder": cross_encoder})
stmt     = (search()
            .retrieve(source("dense", depth=50), source("sparse", depth=50))
            .merge(Rrf()).rerank("cross_encoder").take())
prepared = engine.prepare(stmt)                                     # once, at startup
passages = prepared.execute(question=q, limit=10, where=access)    # each search
```

- **A statement holds no question and no limit.** They change with every search, while how
  to search doesn't. Why: one statement then serves every question, can be named, logged and
  compared, and isn't rebuilt per request. Anything else that must vary per search is
  another statement, not another slot.
- **A statement names what it uses; the engine holds it.** The app registers its sources
  (`VectorSource(index=…, binding=…)`) and rerankers on the engine under names, and stages
  name them: `source("dense", depth=50)`, `.rerank("cross_encoder")`. Names are stripped,
  and a source's name is also what each candidate's trail calls it. Why:
  - a statement holds no objects, so it can be logged, compared and replayed;
  - one built at run time (from config, by an agent) can name only what the app registered;
  - code holding a statement can't reach an index through it (section 11).
  - Name by role, not model (`"cross_encoder"`, not `"bge"`): changing the model then
    leaves every statement, and every log of one, true.
  - Rejected: a resolved twin of each stage holding the objects. It doubles the stage
    classes only to hide what a frozen, private registry already hides.
- **Preparing checks once, against the engine.** `prepare` replays the statement (below),
  then checks that every source and reranker it names is registered, that each retrieve
  searches one index, and that no vector is searched twice, even under two names. Every
  failure is a `StatementError`. Why once: a wrong statement fails at startup, before any
  model is called, and each run then checks only its own inputs.
- **Only `prepare` builds a `PreparedStatement`** (parse, don't validate: holding one proves
  the checks ran). Calling the class raises `TypeError`; `prepare` uses a private
  `_create`. It's a plain class with slots, not a dataclass, refusing changes, and its
  `statement` is a read-only property. Why: an assignment or `dataclasses.replace` would
  give a "prepared" statement that `prepare` never saw. `object.__new__` can still get
  around it; that is deliberate abuse, plain in review.
- **The engine is frozen and keeps read-only copies of what it's given**
  (`MappingProxyType(dict(…))`), so nothing registered changes under a prepared statement.
  The registry is private, and nothing exposes it.
- **Every run is given a filter** (`where=`; `Everything()` searches every chunk), checked
  against the schema of every index the statement searches before any model is called.
  The framework calls it a filter, not access: what it holds is the app's (section 6).
- **Rejected: a `Session`** binding an access filter for one asker, opened from the engine
  (`engine.session(access=…).execute(statement)`, the first design). The session's "access"
  was only a filter that conventionally held access; nothing in the framework depended on
  that, so the idea leaked from the app into the framework. A required `where` on each run
  gives the same guarantee, a filter in every source, with one concept fewer.
- **Type-state builder:** each method returns a type that offers only the steps allowed
  next, so mypy checks the order:

  ```text
  search()                                           → Question
  Question       → .retrieve(source)                 → Candidates
  Question       → .retrieve(a, b, …)                → CandidateLists
  CandidateLists → .merge(fusion)                    → Candidates
  Candidates     → .rerank(name) (later .expand() .prune()) → Candidates
  Candidates     → .take()                           → Statement   (complete)

  later, with query transforms:
  Question       → .rewrite()                        → Question
  Question       → .split()                          → Queries
  Queries        → .retrieve(sources…)               → CandidateLists (always)
  ```

  - **A state is named after what it guarantees,** so later steps only add. `Question` means
    exactly one question flows, which is why one source gives one list; `Queries` (several)
    arrives with `split`, and its `retrieve` always gives lists to merge.
  - **Several lists must be merged before anything else.** `CandidateLists` offers only
    `merge`, so mypy refuses `retrieve(a, b).rerank(…)`.
  - **The complete state is `Statement`, the only state `prepare` accepts.** SQL's word for
    a complete command one prepares. Why not `Search` (the first name): beside the
    `search()` builder and `Index.search`, a third "search" would read poorly.
  - **States are siblings over a private base** holding `stages`. Why: a state subclassing
    another would be accepted where the earlier one is expected.

- **A statement built at run time is replayed through the steps.** `replay(statement)`
  starts again from `search()` and takes one step per stage. Each (state, stage) pair must
  be a step that state offers, and the last state must be a `Statement`; anything else
  raises `StatementError` naming the stage's position. `prepare` keeps the replayed
  statement, and that is what runs.
  - Why: the order rules then live once, in the steps. A separate check over the whole
    tuple ("first a retrieve, a merge exactly when…, one take at the end") would encode the
    same state machine twice, and the copies would drift. A new step is one method and one
    `case`.
  - Why not in `Statement.__post_init__`: replaying ends with `take()`, which builds a
    `Statement`, which would replay again.
  - **Every bound limits what a stage keeps, never what it reads:** a source's `depth`, a
    run's `limit` (later a merge's). Each is "at most" (a source can return fewer hits, the
    filter and validity remove more), and none is compared with another: a loose statement
    isn't a wrong one.
  - **A rerank scores every candidate it receives,** so its cost is set by the stages before
    it: the source's `depth`, or after a merge the union of the sources' lists. Why: the
    usual convention for rerankers (Cohere's and LlamaIndex's score everything they're given),
    and a number on the rerank would be the one bound on input, reading as "depth" beside
    the sources' output bound. Search engines' alternative, a window of the top N to rescore
    (Elasticsearch's `window_size`, Vespa's `rerank-count`), returns as an output bound on
    `merge` when fusing deep lists leaves too many to rerank.
- **Each run checks its inputs where they enter:** the question is stripped and refused when
  empty, the limit must be a positive `int`, and the filter must be a `Filter` fitting every
  searched schema; all before any model is called or any index is read.
- **Query transforms keep the same split (later):** the statement holds the transforms
  (`rewrite`, `split`), the run holds the queries they produce, and the result returns those
  queries, each `Found` naming which one found the candidate. Why: a rewritten question is
  per-run data, like the question itself.
- **Immutable and generative:** each method returns a new statement. Statements are data: they
  can be inspected, logged as provenance and replayed.
- **The question never enters a statement,** so a logged statement can't reveal what an
  asker is after. Auditing questions is the app's choice: it stores them under its own rules
  (access, retention). The trace (planned: plain data per stage, ranked ids and scores, drop
  counts, durations), like the trail, holds no text.
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
- **Values are read with `chunk[field]`, typed by the field:** a `One[T]` gives `T`, a
  `Many[T]` gives `tuple[T, ...]`. A field the chunk doesn't hold raises
  `SchemaMismatchError` naming the field and the chunk id, and says when it's another
  schema's field with the same name. Why `[]` and not `get`: `dict.get` returns `None` on a
  miss, so `chunk.get(f)` would suggest a miss is normal. Every chunk read from an index fits
  its schema, so a miss is a bug. Why not `value`: `One.value` already builds a value for a
  write.

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
  - fields the guarantee slots read (the app's filter, validity) must be required; checked
    before any work, with the filter;
  - `not_` over an optional field is rejected unless the filter says what happens to missing
    values (`is_missing()` / `is_present()`).

**What `T` buys at each use:**

| Use | Example | What mypy catches |
|---|---|---|
| filter | `KbChunks.document_id.eq(doc_id)` | `eq("abc")`: a `str` where a `UUID` is expected |
| ordered filters | `KbChunks.version.ge(2)` | `ge` on a `UUID` field. It exists only for int, float, datetime (a self-type on the method) |
| one of several | `KbChunks.collection_id.in_(ids)` | `in_` on a `Many` |
| multi-valued | `KbChunks.groups.any_of(ids)` | `eq` on a `Many`, `any_of` on a `One` |
| read | `chunk[KbChunks.document_id]` → `UUID`, `chunk[KbChunks.headings]` → `tuple[str, ...]` | no `str` or `Any` coming back. Two overloads: `One[T]` gives `T`, `Many[T]` gives `tuple[T, ...]` |
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
  The checks before a run would then approve a filter on a field the index doesn't have.
- There is more than one schema (chunks, generated entries), each with its own index.
- **One index, one schema.** Why: `check_fits`, the filter check and the startup check all
  compare against a single schema; with two, a filter on one schema would run over the other's
  chunks, and `not_` would turn "matches nothing" into "matches everything". Several kinds of
  records are several indexes, combined by a merge stage. Two logical indexes sharing one Qdrant
  collection is a backend detail; each still has one schema. A schema that changes over time is
  a migration (backfill or re-index), not a second schema.
- Before any work, every field a run uses (its filter, later a stage's settings) is checked
  against the schema of the index it runs against. A chunk field can't filter the entries
  index.
- **The index checks its stored data when it opens:** it compares the schema with the backend
  (e.g. Qdrant payload indexes), then creates what's missing or fails with a clear message.
  Why: otherwise filters run slowly or wrongly without saying so.
- Not now: statements generic over their schema (`Candidates[KbChunks]`), which would let mypy
  catch cross-schema fields. The check before each run is enough to start with.

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
    (section 5) because each must carry the filter into every source and pass the validity
    gates. Revisit if writes
    need a unit of work, a trace or a dry run.
  - **`index.find(where=…)` reads with only the filter its caller gives**, like a raw database
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
  `prune(dedupe, by=KbChunks.document_id)`. Preparing checks that those fields exist
  and have the right types.
- **The package ships no schema preset.** Why: nothing in it reads a field by its role
  (stages take fields as settings), so a preset would be a guess at an app's schema, and a
  public name to keep. Apps declare their own; the server's is in `backend-design.md`,
  section 5. Extract a shared one only when a second app repeats the same fields.

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
    build step 7), so a changed group isn't re-embedded;
  - later perhaps `export(where)`, for copying a collection.
- **Kinds:** `Dense` and `Sparse` (built; "Sparse vectors" below). More kinds become further
  `Vector` subclasses when a source needs them:
  - `MultiDense`: many rows per chunk, one per token, compared by max-sim.

  Each is a sibling of `Dense`, not a subclass. Why: `isinstance(vector, Dense)` would
  otherwise match a `MultiDense` as well, and the order of the `case`s would decide
  correctness. Each kind adds a `…Value` and a `…Query` to the closed unions `VectorValue` and
  `VectorQuery`, so mypy points at every backend that hasn't learned it.
- **Max-sim scores grow with the question's length,** so they aren't on any dense score's
  scale. The engine merges sources by rank, and a score threshold can't be applied to a
  multi-vector source as it is.
- **Adding, dropping or changing a vector means a new collection:** Qdrant can't change a
  collection's vectors in place.

**In Qdrant:** each vector value is one of the point's named vectors. Qdrant has two slots:

| Kind | Slot | Settings compared at `open` |
|---|---|---|
| `Dense` | `vectors_config` | size, distance, one row |
| `MultiDense` (later) | `vectors_config`, with `multivector_config` | size, distance, many rows (max-sim) |
| `Sparse` (built) | `sparse_vectors_config` | modifier (none read back as `none`) |

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

**Known about IDF (`modifier="idf"`), not worth handling.** Both were found while matching
the in-memory index to a Qdrant server. Neither changes rankings in a way that matters, so
neither needs work. They're written down only so they aren't rediscovered as bugs.

- **IDF counts chunks the asker can't see.** It's computed over the whole collection
  (Qdrant's default, `IdfScope.GLOBAL`), so a rare term in hidden documents slightly
  lowers its weight for everyone. In principle that's a side channel, but no sparse score
  leaves the engine: RRF keeps only ranks, and the reranker sets the final scores. Only
  an API returning raw sparse scores would make it worth a look; Qdrant's
  `IdfCorpusParams` can then count only the chunks the asker's filter matches.
- **The server counts deleted points until it cleans its segments up,** so right after a
  delete or an overwrite (every index job) IDF is slightly off. It's transient, and one
  document's chunks barely move a logarithm over the collection. The only consequence is
  for tests: none reads IDF after a delete (`tests/index/test_sparse_search.py`).

### Sparse vectors (built)

```python
class KbChunks(Guarded):
    dense = Dense(size=1024, distance="cosine")
    sparse = Sparse(modifier="none")  # learned weights; "idf" for BM25-style counts

KbChunks.sparse.value({4865: 0.21, 12: 0.08})  # an id → its weight
```

Sparse search scores chunks by the terms they share with the question, so it catches what
dense search blurs: exact numbers ("Điều 12"), abbreviations, rare terms. **Nothing in it
belongs to one model:** BM25, SPLADE, inference-free models and bge-m3's lexical weights
all fit.

- **A value is `Mapping[int, float]`: an id to its weight.** An id is a dimension of the
  embedder's own space (a vocabulary token, a word's hash); turning terms into ids is the
  embedder's job, so the framework never owns a vocabulary. Ints, not strings: Qdrant
  stores ids, not terms.
- **`Sparse.value` and `Sparse.query` check what every model must meet:** each id an
  integer (numpy's too; a `bool` or a `str` refused, as some libraries give ids as
  strings), from 0 to 2**32 - 1 (Qdrant stores u32); each weight finite. They store a
  read-only copy with plain `int` ids and `float` weights. Negative and zero weights and
  an empty mapping are kept: they're a model's business, and harmless.
- **No `size`:** a hashed vocabulary has no fixed width.
- **`modifier` is declared on the vector,** like `Dense.distance`: it's how a backend
  compares, and `open` compares it. `"none"` takes the weights as they are; `"idf"`
  multiplies each query weight by its id's IDF over the index, so an embedder giving BM25's
  term-frequency part gets BM25. Why the index and not the embedder: corpus statistics
  change with every write.
- **The score is the dot product over the ids the query and the chunk share.** A chunk
  sharing none isn't a hit, so `limit` may go unfilled: Qdrant finds sparse hits through an
  inverted index. One sharing an id with weight 0 is a hit scoring 0.
- **IDF is Qdrant's:** `ln((N - n + 0.5) / (n + 0.5) + 1)`, with N the chunks with at least
  one weight and n those holding the id (IR's document frequency: a document there is a
  chunk here), counted over the whole collection, whatever `where` matches. A Qdrant server
  is the reference: the client's local mode also counts chunks with no weights, and one
  test marks it.
- **Scores have no fixed range** (no cosine; they grow with shared terms and with the
  model's scale), so a sparse and a dense score can't be compared: the engine fuses by rank.

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
- **`type VectorBinding = DenseBinding | SparseBinding` is a closed union,** matched
  with `assert_never` like `VectorValue`. Each kind has its own `case`, never an
  or-pattern: one `case` for both would type `vector` and the rows as unions, losing which
  rows belong to which vector (and a variable assigned in both cases needs a name per
  kind, as mypy gives it one type in the whole function). Why not a union of embedder protocols: protocols are
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
- **One protocol per vector kind, named after it:** `DenseEmbedder` and `SparseEmbedder`
  (`Mapping[int, float]` per text, built). Why:
  - each kind's output has its own shape, so a sparse embedder could never implement the dense
    protocol, and a plain `Embedder` would claim a generality it lacks;
  - services serve the kinds from separate endpoints (TEI's `/embed` and `/embed_sparse`);
  - adapters live in apps, so renaming `Embedder` once sparse arrives would break them all.
- **A sparse embedder's two sides matter more than a dense one's:** BM25 weighs a passage by
  its term counts and length but a question only by its terms, and inference-free models
  run only on passages.
- **Nothing checks that the vector's `modifier` suits the embedder.** BM25 needs `"idf"`
  (without it, no IDF); a learned model needs `"none"` (with it, its weights are scaled
  twice). As with the model itself, the app chooses the pairing; `SparseEmbedder`'s
  docstring says which suits which, and a wrong one shows in `evaluator.compare`. Why not a
  `modifier` on the protocol: a general adapter (`HuggingFaceSparseEmbedder`) can't know
  which its model needs, and dense has no such check either.
- **One model call filling two vectors stays out of the protocols.** bge-m3 gives dense and
  sparse weights from one forward pass; an adapter can share the work (one object exposing
  a dense and a sparse view over a shared result). A protocol question only if measuring
  shows the double call costs too much.
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
    thread pool inside `execute` can remove that without changing any API;
  - a thread can't be cancelled: a controller's timeout stops waiting, but the statement runs to
    its end. `execute` will check a deadline between stages, and adapters set timeouts on their
    HTTP clients. An abandoned statement only reads, so only work is lost.

  **When to add async:** agents run many statements in parallel, or the trace shows fan-out
  taking a large share of a search. Then add async protocols (`AsyncDenseEmbedder`,
  `AsyncIndex`) and an async `execute` beside the synchronous one. Statements and stages are
  data, so they stay unchanged.

### Chunk length and the models

The old preparation sized chunks in its one embedder's tokens. That assumed one dense
vector, so one tokenizer and one way to count. A schema may now declare no dense vector or
several, and the reranker reads every chunk too.

- **Chunks are sized in words** (whitespace-separated; in Vietnamese, syllables), at sentence
  boundaries, as Elasticsearch's `semantic_text` and Haystack's `DocumentSplitter` do. Why:
  the size is chosen for retrieval quality (how much a hit holds), which no tokenizer
  decides; a model change doesn't mean re-preparing; and words vary less than characters
  between languages and tokenizers (measured once on Vietnamese and English text: XLM-R
  1.10 and 1.31 tokens per word, MiniLM 1.72 and 1.16). The old preparation keeps counting
  in `KB_CHUNK_TOKENS` until it is reworked.
- **Two questions, kept apart:**

  | Question | Can it be checked? | How |
  |---|---|---|
  | Does a model's pipeline cut a chunk off? | yes, exactly | run the model's own tokenization, with its truncation, on what that model reads, and compare the tokens kept with the tokens there were |
  | Does a model read long input well? | no | evaluation (build step 9); a window can be longer than the text a model was trained on |

  So if fit is checked, it is checked by the model's own tokenization, never by a declared
  limit (MiniLM's tokenizer says 512, sentence-transformers cuts at 256) and never by our
  word count. What each model reads counts: an embedder reads `embedded_text`, headings
  included; a reranker reads the question with the text.
- **Not built:** a fit check. With bge-m3 and bge-reranker-v2-m3 (8192 tokens each), a
  250-word chunk is about 275 tokens, so it would always report nothing. Build it when a
  model with a small window is configured. Counting stays outside the protocols: a local
  adapter can offer it, a hosted model returns an error or truncates on its own (TEI refuses
  long input unless `truncate=true`). It would be a measurement, not a refusal: a cut-off
  chunk still returns its exact text, so ranking suffers, not a guarantee (section 2).
- **The chat LLM is a different constraint:** it reads many passages at once, so its limit
  decides how many passages fit its context when an answer is assembled, not how long a
  chunk is.

### Filter expressions

- **AND / OR / NOT over schema fields and chunk ids.** One mechanism for every filter. Why:
  today's fixed fields plus `attributes` can't OR across fields.
- **Methods are the API:** `KbChunks.document_id.eq(x)`, `in_(…)`, `any_of(…)`. Why: they read plainly and
  type-check under strict mypy. Operator overloading (`==`, `|`) may come later as sugar over
  the methods, only if they prove ugly.
- **Combinators are functions, the nodes frozen dataclasses:** `and_(…)`, `or_(…)`, `not_(…)`
  build `And`, `Or`, `Not` (trailing underscores because `and`, `or`, `not` are keywords, as in
  SQLAlchemy). `type Filter = Equals[Any] | In[Any] | AnyOf[Any] | And | Or | Not | Everything`
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
    - Its uses: an app without access control passes it as every run's `where` ("The
      guarantee slots"); tests and indexing code read or clear a whole index with it.
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
    as a string would miss the same instant written with another offset. An `in_` or
    `any_of` with no values becomes a filter matching nothing. Why: Qdrant reads an empty `should` as no
    constraint and matches every point.
- **`in_` on `One` is its own node, `In`, not an `AnyOf`.** Why: `AnyOf` treats the chunk's
  value as a collection, so on a `str` field it would match by characters. Its first caller
  is the access filter for a manager (`collection_id.in_(…)`); an empty `in_` matches
  nothing, so a manager of no collection sees only what their groups allow.
- Not yet: filtering by chunk ids (first caller: re-checking earlier candidates).

### Generated entries

An entry's text isn't the source's words, so it can't be a chunk's `text`. Entries get their
own schema with a field that points to chunk ids. A source stage searches the entries and
returns the chunks they lead to, recording "found through" in each candidate's history. They
become a separate source, not a special case inside the index.

This is a **resolving source**: it searches one index (the entries) and returns chunks of
another (its *target*, the chunks).

- **It resolves through the target index's `find`, with the run's filter.** What it returns is
  then checked on the chunks' own schema, whatever the entries hold. Validity runs on the
  resolved chunks as usual.
- **It keeps one entry per chunk, the best ranked.** Several entries can lead to one chunk, and
  RRF would otherwise count it several times in one list.
- **Its ranks are given after resolution,** counting only the chunks it returns. Why: a gap in
  the ranks would reveal that something hidden ranked above.

### Several indexes

**A `Retrieve` searches one index, for now.** Its sources may search different vectors of it,
never different indexes. Why: a candidate is identified by its chunk id, which is only unique
within one index, and one index means one schema, so one filter fits every source.
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

The engine can't enforce access or currency on data it doesn't understand. So the frame has
fixed **slots**, and the app fills them:

```text
framework (generic)                          app (server) fills
──────────────────────────────────────────   ─────────────────────────────────────
schema: declared fields, types, indexes   ← document_id, version, groups, collection…
chunk = id + text + field values
filter expressions over fields
filter slot:   a filter, required each run ← access: "shares a group OR manages the collection"
validity slot: chunks → ids to keep         ← "document active, last indexed version"
stages, the engine, prepared statements, gates
```

- **Every slot must be filled.** No run starts without a filter, and no engine is built
  without a validity check. An app with no access control says so with an explicit
  `where=Everything()`, and one with nothing to check passes `AlwaysValid()`, so each absence
  is visible. Why no separate `AllowEverything`: `Everything` is already a `Filter`, so the
  engine passes it to every source unchanged, and one idea keeps one word.
- The engine decides *where* the slots apply: the filter goes into every source (later every
  expand), and the validity gate runs where candidates enter and again before `Take`.
- **What the filter means is the app's design, not the framework's.** The framework calls the
  slot a filter, not access: the server's holds access, but nothing in the framework depends
  on that, and another app may filter by anything. How the app builds it (which fields,
  combinators or a denormalised field) is decided in the app.

### Filters, gates, validity and cuts

One word per idea, since all four narrow the results:

| Word | Means | Applied |
|---|---|---|
| **filter** | a `Filter` expression over chunk fields, evaluated by the index; each run's `where` is one, and the server's holds access | inside the index, before ranking |
| **gate** | a point in the engine where every candidate must pass a check, whatever the statement says | in code, after search |
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
- **Later:** a statement-level filter (`.where(…)`: a collection, "in force only") is a
  `Filter` the engine ANDs with each run's; prunes (thresholds, dedupe) are stages that cut
  by score. Neither is a
  gate: a gate always runs, and a statement can leave a stage out.

### How the slots plug in

**Validity: a port the engine holds.**

```text
engine = Engine(validity=ServedVersionsOnly(session_factory), sources=…, rerankers=…)
# validity's Protocol: chunks → ids to keep
```

- It is part of the frame, not a stage the app writes into statements. Why: a statement author
  or an agent could leave a stage out, and stale results would pass.
- It doesn't depend on the asker ("is the document active, is this its indexed version?"), so
  it belongs to the long-lived engine.
- **The port: `valid_ids(chunks: Sequence[Chunk]) -> Set[UUID]`, the ids to keep.** A port
  (ports and adapters): the package defines the shape, the server's adapter queries Postgres,
  and the package never imports a database. Why this shape:
  - **It can only narrow.** The engine keeps a candidate only if its id is returned, so an
    adapter can't add, swap or reorder chunks, and extra ids are ignored.
  - **It fails closed.** A buggy adapter returning nothing serves nothing, which is noticed at
    once. Returning the ids to drop would fail open: stale and deleted chunks served.
  - **It's batched:** one call, so one query, per gate. Never one per chunk (the N+1 problem).
  - **It takes chunks,** so the adapter reads its own schema's fields (`document_id`,
    `version`); the package needn't know which fields identify a version. Not candidates:
    the app never imports engine internals.
  - Rejected: returning chunks (could widen), one flag per chunk (positional, so a misaligned
    list keeps the wrong chunk), one chunk per call (N+1), verdicts with reasons (no need yet).
  - Synchronous for now; an async port comes with an async `execute`.
- **Where a run calls it:**
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

**The filter: given to each run.**

```text
access = self._read_access_filter(principal)   # server service, with its own DB session
passages = prepared.execute(question=question, limit=limit, where=access)
```

- Why each run, over a policy the engine holds (asker → filter): it is simpler. The framework
  has no asker type and no per-request plumbing. In the server, building the filter needs a
  DB lookup (the principal doesn't carry its groups or managed collections), and services
  already check permissions with a DB session.
- The cost: that the filter holds the right access rests on the app, not on structure. A call
  site that passes a filter that's too broad breaks it. The server keeps it robust by:
  - **one service runs searches** (`SearchService`); `/search`, `/answer` and agents all go
    through it. It holds the statement prepared at startup, never the engine, so it can run
    that checked statement and nothing else (least authority);
  - a test of the filter that service builds for each kind of asker;
  - planned: an import-boundary check in CI: only the wiring builds an `Engine`.
- **When to switch to an engine-held policy:** when agents or other entry points need to run
  statements themselves. The engine sees a filter either way, so the switch stays cheap.
- Rejected: a `Session` holding the filter (section 5).

**One filter for every schema: an access preset.** A run gives one filter, but a statement
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
- The preset holds every field the access filter names, not only groups: an `or_` across
  fields (my groups, or the collections I manage) must be valid on every schema too. Fields
  the validity adapter reads belong there for the same reason: it is called on candidates
  from every source. Which fields those are is the app's decision.
- `execute` checks the filter against every source's schema before any model is called.
- The cost: entries copy their chunk's access fields when indexed, so a change to a document's
  groups updates its chunks and its entries. A missed update can't leak a passage: the
  resolving source fetches chunks through the chunks index with the filter. On the chunks the
  filter is the guarantee; on the entries it keeps recall (hidden entries don't fill the
  depth) and closes the rank side channel.
- Rejected: a filter per schema in each run (one rule built twice, which can drift; kept as
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
   handler   index.replace(where=document is x AND version is 2, with_=v2 chunks)
                                                      v1 still served: indexed_version = 1
   worker    commit indexed_version = 2                the switch: one row, atomic
   after     index.delete(where=document is x AND version isn't 2)
   ```

   - `replace` within the version, not `put`: a version prepared again with fewer chunks
     leaves none of its old ones, which validity would otherwise show.

   - From a reader's view the switch is atomic: all of v1, then all of v2, never a mix and never
     nothing.
   - A crash after the commit leaves v1's chunks behind. The validity check already hides them,
     and the document's next index job or `gc` removes them. It's the same idea as "originals are deleted after the commit".

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
| Preparing (once, at startup) | stage order, by replaying the stages through the steps; every source and reranker registered; one index per retrieve; no vector searched twice | a wrong statement stops the app before it serves anyone |
| Each run, before any work | the question has text; the limit is a positive `int`; the filter is a `Filter` fitting every searched schema | nothing is spent on a run that would fail |
| During the run | empty results, wrong score counts, a tool failing | only known as it runs |

mypy covers statements written in code. Statements built at run time (from config, or by an
agent) still need the checks, so `prepare` always replays the statement (section 5) and keeps
the replayed one. Limits aren't compared with depths: each is "at most" for its own stage.

## 8. Loops

Loops are common in advanced retrieval (retrieve → grade → rewrite → retrieve again: CRAG,
Self-RAG, iterative retrieval). **Statements stay acyclic. A controller above the engine owns
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
These still run as prepared statements, through the same gates.

## 9. Agents that recommend decisions

Retrieval is the agent's evidence layer. The guarantees carry up to the recommendation unchanged.

```text
decision agent (controller)
  1. frame     the decision → what must be known (options, constraints, criteria)
  2. gather    one statement per need  ──→ prepared, run with the user's filter
  3. weigh     options against evidence (the LLM's work)
  4. check     is every claim backed by a passage? does any evidence conflict?
  5. answer    recommendation + cited evidence, or "not enough evidence"
```

| Concern | Provided by |
|---|---|
| No recommendation rests on, or hints at, hidden knowledge | every run is given the user's access filter; the agent holds prepared statements, never the engine or an index |
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
filter expression and the engine's frame. Add rewrite, expand, prune and the controller when
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
  | statements, stages, the engine and prepared statements, validity | retrieval itself |

  | In a later package (e.g. `salemdex_agents`) | Why separate |
  |---|---|
  | the controller, agent loops, LLM tools | the controller sits above the engine (section 8); a package boundary makes that layering real |
  | answer generation, LLM providers | separate dependencies; retrieval stays usable without an LLM |

  Why the split also helps safety: an agents package uses retrieval's public API, where
  the only way to run a statement is a prepared statement given a filter. Statements hold
  names, never indexes, so an agent given prepared statements can't reach an index through
  them. The rest is a convention: agents are never handed the `Engine` or an `Index`
  (section 11).

- **Nothing domain-specific.** Every technique in the packages works for any knowledge
  base; regulations, Vietnamese and other domains are the app's (section 12).
- It never imports `document_processor`. An import-boundary check in CI enforces it. Why: the
  engine is generic; turning `chunking.Chunk` into an index chunk (today's `ingest` and
  `address_keys`) is app code, in the server, which declares the schema anyway.
- Its extras hold the backends and models: `qdrant`, `huggingface` (embedder, reranker).
- A separate repo waits until the engine needs its own releases.
- The family follows later, one rename per commit: `salemdex-ingest` (today's
  `document_processor`, moving to `packages/ingest/`) and `salemdex-server` (today's
  `kb_server`). The GitHub repo is already `salemdex`.
- **The `vn_regulation` profile moves to the app** with the `salemdex-ingest` rename, for
  the rule in section 12. It began as an example profile that's easy to use. The framework
  keeps `plain`, the profile machinery and its counter styles (`lower_vn` beside
  `lower_latin`: a numbering style, like CSS's, not a domain rule), and lets the app register
  its own profiles beside the built-in ones. The framework's tests that use `vn_regulation`
  today switch to a neutral example profile of their own.

**Folder layout.** The core modules stay at the top level. Each new kind of part gets a
subfolder when it first arrives.

```text
src/salemdex_retrieval/
  __init__.py              the top level's names ("Import paths" below)
  declarations.py  errors.py  schema.py  filters.py                       core
  checks.py                checks on plain values: check_positive_int, to_finite_float
  vectors.py               Dense, Sparse, VectorValue, VectorQuery (closed unions)
  chunks.py                Chunk, EmbeddedChunk, Hit (chunk, score)

  index/                   storing and finding chunks
    __init__.py            Index, InMemoryIndex
    protocol.py            Index, check_batch, check_filter, check_query, check_limit
    in_memory.py           search: filter first, then score every stored value; sparse
                           IDF counted over every stored chunk
    qdrant/
      index.py             search: query_points(using=…, query_filter=…, limit=…)
      filters.py  payload.py
      (index.py also holds _VectorConfigs and the describers; vector_config.py if it grows)

  embedders/               used by both writes and searches
    __init__.py
    protocol.py            DenseEmbedder, SparseEmbedder, DenseBinding, SparseBinding,
                           VectorBinding, and embed_chunks(chunks, *, bindings)
    fake.py                a deterministic embedder for tests, the server's too
    huggingface.py         HuggingFaceDenseEmbedder: any sentence-transformers model, as
                           Hugging Face names it, behind the `huggingface` extra

  engine/                  retrieval itself (build step 3)
    stages.py              retrieve (sources by name), merge (rrf), rerank (a reranker by
                           name), take, as plain data
    statements.py          search() and the type-state steps:
                           Question → CandidateLists → Candidates → Statement; replay
    sources.py             VectorSource: what a registered source name stands for
    results.py             Candidate, the trail entries (Found, Fused, Reranked),
                           Passage
    validity.py            the Validity port, AlwaysValid
    engine.py              Engine (the registry, prepare) and PreparedStatement (execute):
                           the filter in every source, the validity gate where candidates
                           enter and before Take

  rerankers/               with the rerank stage
    protocol.py  fake.py
    huggingface.py         HuggingFaceReranker: a one-label CrossEncoder, same extra

  _huggingface.py          what both adapters share: the device, load_model

tests/                     mirrors the source folders, each with its own __init__.py
  index/                   test_contract.py holds the search cases too
  embedders/               test_embed_chunks.py
  engine/                  test_stages.py  test_statements.py  test_sources.py
                           test_results.py  test_engine.py
  test_public_api.py       the top level's __all__, written out in full
```

- **`engine/`, not `search/`.** "Search" would otherwise name three things:
  `index.search(query, where=…)` (one source, raw hits), the package, and the `search()`
  builder, so a reader couldn't tell from the call site which one runs.
  - `Index.search` keeps the word every vector store uses for a single-source read.
  - The package is named after the `Engine` it exports.
  - `search()` reads as "a search", which is what a statement is: a way of searching, run
    for any question.
- **`Hit` lives in `chunks.py`, beside `EmbeddedChunk`.** One wraps a chunk on its way into the
  index, the other on its way out. The engine then imports it from the core, not from
  `index/`, to start each candidate's trail.
- **`embed_chunks` sits beside the embedder protocols,** as `check_batch` sits beside `Index`:
  it's the one helper every caller of the protocol shares.
- **The fake embedder lives in `src`,** as `InMemoryIndex` does. Why: the server's tests need
  one without loading torch.
- **Backends and models sit behind extras** (`qdrant`, `huggingface`). No package
  `__init__.py` imports them.
- **`engine/` modules depend only on those before them here:** stages, statements, sources,
  results, validity, engine.
- **`engine.py` holds both `Engine` and `PreparedStatement`:** each reaches the other's
  private parts (`prepare` calls `_create`, `execute` reads the registry), so one module keeps
  that access inside it. Its run helpers move to a private `_run.py` if it grows too long.
- **`validity.py`, not `gates.py`:** the module holds the check, not the place it runs. A
  *gate* is where a run calls it; "the validity gate" stays a word in the engine's
  docstrings, with no module of its own.

**Import paths (built).** Each public name has exactly one path to import it from. The
question isn't what's public, but which path is the one to use:

| Tier | Holds | Names |
|---|---|---|
| top level | what callers build or call: the vocabulary at many call sites | `Schema`, `One`, `Many`, `Dense`, `Sparse`, `Chunk`, `EmbeddedChunk`, `and_`, `or_`, `not_`, `Everything`, `search`, `source`, `Rrf`, `Engine`, `VectorSource` |
| a core module, or a subpackage's `__init__` | what callers receive, annotate, catch or implement | `errors.SchemaMismatchError`, `errors.StatementError`, `filters.Filter`, `chunks.Hit`, `index.Index`, `index.InMemoryIndex`, `embedders.DenseEmbedder`, `embedders.SparseEmbedder`, `embedders.DenseBinding`, `embedders.SparseBinding`, `embedders.VectorBinding`, `rerankers.Reranker`, `engine.Statement`, `engine.PreparedStatement`, `engine.Passage`, `engine.TrailEntry`, `engine.Validity`, `engine.AlwaysValid` |
| a module of its own | what needs an extra, or is only for tests | `index.qdrant.QdrantIndex`, `embedders.fake.FakeDenseEmbedder`, `rerankers.fake.FakeReranker` |
| not exported | what only the package uses | `replay`, `Candidate`, the stage classes, the intermediate states, `score_texts`, the `check_*` helpers |

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
- **There is no `execute` function:** `PreparedStatement.execute` is the one way to run a
  statement, so none runs unchecked, and none without a filter.
- **`AlwaysValid` isn't at the top,** though it's built: only apps with nothing to check
  and tests use it, and the top level shouldn't suggest it's the usual choice.
- **`tests/test_public_api.py` pins the top level's `__all__`,** written out in full, as
  the golden payload test does: a name joins or leaves the top level only by a change a
  reviewer sees.

**`Index.search` (built):**
`index.search(query, *, where, limit) -> list[Hit]`.

- **`query` is a `VectorQuery`:** `DenseQuery | SparseQuery`, later `| MultiDenseQuery`.
  - It carries its vector, so the backend knows which named vector to search.
  - Its numbers were checked where they entered (`Dense.query`, `Sparse.query`).
  - The union is closed, so each backend `match`es on it and ends with `assert_never`. mypy
    then points at every backend that hasn't learned a new kind.
- **One query per call.** Several sources are the engine's job: it calls `search` once per
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
  and dot, but never compared across vectors or questions. That's why the engine merges by
  rank. A `Hit` holds the chunk without its vectors, as `find` does.
- **The in-memory index scores in plain Python** (`math.sumprod`, `math.hypot`), not numpy.
  Why: the core package has no dependencies, and numpy only arrived through
  `qdrant-client`, so the package failed to import without the `qdrant` extra. A zero vector
  scores 0.0 under cosine, as it does in Qdrant.
- **Qdrant converts by kind in two helpers,** `_to_qdrant_value` (writes) and
  `_to_qdrant_query` (searches), each a `match` ending in `assert_never`, returning
  `_QdrantVectorData`. That alias is `list[float] | models.SparseVector` and widens with
  each kind (`list[list[float]]` for multi-dense), so mypy catches a case returning the
  wrong shape. A plain `dict` isn't a sparse vector to Qdrant: pydantic refuses it.
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
3. **Statements, the engine and prepared statements** with retrieve, merge, rerank and take,
   the filter each run requires and the validity check. **(done)**
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
waiting time and throughput. For the measurement, each run will record each stage's duration
in its trace (planned, section 5).

## 11. Open questions

- **Chunk length against several models** (section 6, "Chunk length and the models"):
  whether any case should fail rather than be reported, such as a model whose window is
  shorter than the configured chunk size. Embedders, the reranker and later the chat LLM
  all read chunks, in different roles; settle it when the preparation is reworked.
- **The agent step API** (later, with the controller):
  - How a statement starts from earlier candidates. A likely shape: the LLM sees short handles
    (`p2`), and the controller maps them back; an invented handle doesn't resolve. Earlier
    candidates pass the filter and the validity check again, since documents change between
    rounds.
  - What the controller, and so the LLM, sees after a tool call: passage text and handles, how
    much of the trail, how many passages per round (the context window limits this).
  - Everything the LLM writes is untrusted input. What keeps that safe: every run is given
    the user's filter, and a statement can name only what the app registered.
- **Keeping agents away from indexes (mostly settled).** A source used to hold its
  `Index`, so code holding a statement could call `source.index.find(where=Everything())`
  with no access filter. Statements now name their sources, and the engine resolves the
  names from its private, read-only registry (section 5), so a statement or a prepared
  statement leads to no index. What remains is a convention: agents get prepared
  statements (or tools over them), never the `Engine` or an `Index`, whose `find` reads with
  only the filter its caller gives. Python can't hide the engine's `_sources`; ruff's
  `SLF001` (private member access) would flag reaching for it outside the engine.
- **Still to design:** score scales for thresholds (a prune needs to know whose score it
  cuts on), multi-vector bindings for bge-m3's dense, sparse and multi-vector outputs from
  one model call (for now an adapter's concern: "Embedders"), adapter error types, and deadlines between stages.

## 12. Techniques to come, and where they belong

**The rule: the salemdex packages hold only techniques that work for any knowledge base.
Anything about regulations, Vietnamese or another domain belongs to the app.** Why: the
packages are meant to be reused, and a domain rule built into one would be wrong for the
next corpus and couldn't be taken out without breaking the first.

**The framework gives the mechanism, the app the knowledge**, as with the guarantee slots
(section 6). The framework fixes the stage, its contract and where it runs; the app supplies
what it means: a parser, a glossary, which field holds the parent section. When a technique
can't be stated without the domain, its general shape may still be a framework piece.

A test for each new piece: could its docstring and tests use a corpus of recipes or product
manuals instead? If not, it's the app's.

### Retrieval (`salemdex_retrieval`)

| Technique | Framework piece | The app supplies |
|---|---|---|
| Hybrid dense + sparse | a `Sparse` vector on the schema, `SparseEmbedder` and `SparseBinding`, Qdrant sparse vectors, RRF merge (all built); BM25 and `SparseEncoder` adapters (next) | the model, e.g. a SPLADE model or bge-m3's lexical weights; BM25's tokenizer |
| Lookup by field | a source that turns the question into a filter on declared fields and fetches with `find` | the parser, e.g. "Điều 12 Nghị định 145" → headings |
| Normalizing and expanding the question | a query transform taking a function or a glossary | diacritics restoration, abbreviations (NLĐ → người lao động), the domain's terms |
| Rewrite, multi-query, HyDE, decomposition | query transforms over an LLM protocol | prompts in the domain's language |
| Generated entries | the resolving source (section 6), an entries schema, a generator protocol | what to generate, the prompts |
| Small-to-big | expand to the chunks sharing a parent field, or to neighbours by position | which field is the parent (Điều, a recipe's step list) |
| Following links | expand over a field of ids, with the kinds of link as data | extracting references from text; the kinds (amends, replaces) |
| Coarse to fine | a step on `Question` producing a scope from a coarse index (section 6, "Several indexes") | what the coarse entries are |
| Prune | dedupe near-identical text, MMR diversity, cut at a score gap or threshold | the thresholds, set from evaluation |
| Time scope | nothing new: a filter on date fields, and the validity slot | effective dates, what "in force" means |
| LLM rerank | a reranker adapter over the LLM protocol | the model |
| Evaluation | question types, runs, judges (quote, LLM), scorecards and comparisons (section 13) | the question sets, under `data/`; the corpus; the LLM judge's model and prompts |

### Agents and workflows (the later agents package, section 10)

| Technique | Framework piece | The app supplies |
|---|---|---|
| LLM providers | a protocol with local adapters (Ollama), like the embedders | the model |
| The controller | loops over acyclic statements, budgets (rounds, time), re-checking the final output (section 8) | the budgets |
| Corrective retrieval (CRAG) | grade → rewrite → retry, or abstain | the grading prompt, thresholds |
| Checking claims | each claim tied to a passage handle; unsupported claims revised or abstained from | — |
| Routing | classify the question, pick a statement | the categories and their statements |
| Conversation | condensing a follow-up question with the history | how many turns |
| Tools for an agent | tools over prepared statements, run with the user's filter: search, expand, get by field, a document's outline; handles (`p2`) instead of ids (section 11) | which tools, the field names |
| Browsing structure | an outline as a tree over a path field on chunks; opening a node | the levels (Chương, Điều) |
| Extraction at ingestion | an LLM step filling a declared schema from a document, checked by the schema | the schema (issuer, number, effective date) |
| Workflow patterns | prompt chaining, routing, parallel calls, orchestrator-workers, evaluator-optimizer, as small composable pieces | the workflows |
| Evaluating agents | supported claims, correct citations, abstention, rounds, latency | the labelled cases |

### The app only

- What a domain's documents look like: profiles such as `vn_regulation` (moving, section
  10), reference syntax, effective dates, what amending means.
- Language: diacritics restoration, glossaries, abbreviations, prompts.
- Domain workflows: compliance checks against regulations, the impact of an amendment,
  comparing regulations.
- Every threshold and budget, and every question set.

## 13. Evaluation

Being built (build step 5 in "Pick up here"); each part says what's built. Salemdex holds
the evaluation; the app supplies the questions and the corpus (`salemdex-product.md`,
sections 4 and 5).

**The problem.** The old evaluator (`document_processor.retrieval.evaluation`) grades a hit
at three levels: by document, by reference key (`"dieu:9"`), by quote. The new index has
neither of the first two as the old one knew them: `document_id` is a UUID assigned at
upload, and salemdex chunks carry no reference keys. A port would grade nothing. It would
also measure only the final list, while improving one lever at a time needs to know which
stage lost the answer. And a grade fused into the run leaves nothing for a judge added
later (an LLM's) to work on.

What it's for, in order: show that the new engine scores at least as well as the old one,
on the same documents, model and questions (the gate for deleting the old package); then
compare statements on the developer's own questions (depths, rerank or not, later hybrid).

### Kinds of evaluation

| Kind | Question it answers | Input | Measures | When |
|---|---|---|---|---|
| quote occurrences (facts) | is each gold quote whole inside some chunk, in its place? | questions, the corpus; no model | per quote: chunks in place, chunks elsewhere | now; first, since a quote no chunk holds makes that question's score meaningless |
| retrieval | did the key points reach the results, at what rank, at each stage? | questions, a prepared statement, a limit | hit@k, MRR, all-key-points rate@k, found by each source, rerank moves; per tag; misses | now |
| comparison | is statement B better than A, on which questions? | two runs scored on one assessment, a cut-off k | questions improved, worsened, unchanged at k, each with both ranks | built |
| abstention | can a score tell answerable from unanswerable? | a run | top scores of each | now; a threshold holds for one reranker only |
| cost | what does each stage cost? | the trace (planned, section 5) | duration and candidates read, per stage | with the trace |
| reasoning, agents | is the answer right, with the real context and with the gold passages? supported claims, citations? | answers, agent runs | a judge or labelled answers | with the answers and agents packages |

### The question set: types, not a file format

```text
Question   id, text, tags (sorted, no two alike), key_points: tuple[KeyPoint, ...]   all needed
           is_answerable: key_points isn't ()
KeyPoint   one thing a full answer says: quotes: tuple[Quote, ...]   any one will do; at least one
Quote      text, where: Filter                                       the chunks that may hold it
```

Built in `evaluation/questions.py`, tested in `tests/evaluation/test_questions.py`.

- **A question needs all its key points; a key point needs any one of its quotes.** A key
  point is one thing a full answer says (a rule, its exception); its quotes are the places
  stating it, alternatives to one another. TREC calls key points nuggets. The old flat list
  ("any answer counts") can't say "several places": 7 of the current set's 53 questions
  need two places each, and were graded found when only one was.
- **Grade by quote** (`salemdex-product.md`, section 5): a quote survives re-chunking and a
  new outline; an article label doesn't. So evaluation needs no reference keys.
- **`where` instead of a document id:** for this server, `KbChunks.document_id.eq(…)`.
  Salemdex then knows nothing of documents, and any schema works.
- **Types only.** A file format would need filters written as JSON, a design of its own;
  the app's loader builds the objects (mapping its file names to document ids). A format
  comes when a second app needs one.
- **No location.** A label such as "Article 1, clause 2: …" is cheap to write but goes
  stale, can't be checked, and often paraphrases the document, so it's restricted text. A
  miss is explained instead by the ids of the chunks its quotes occur in;
  the app can show those chunks' headings locally.
- **Not limited to one document:** each quote has its own `where`, so key points, and the
  quotes within one, may sit in different documents. Every document a question
  cites must be in the corpus.

### Runs and judgments, kept apart

The pattern from TREC: a **run** records what a system returned, **judgments** (TREC's
"qrels") record what is relevant, and a **metric** is a function of both.

```text
Run            statement, chunks_by_question: question id → tuple[ReturnedChunk, ...]
               (best first), evaluator_fingerprint                no grading, no text
ReturnedChunk  chunk_id, trail      a chunk the statement returned, by its id
Judgment   question id, chunk id (a UUID), grade, key_points: tuple[int, ...], judge (a name)
           grade: "unrelated" | "related" | "answered"; only "answered" may name key points
Assessment one judge's judgments, by (question id, chunk id): judge, judged_depth,
           judgment_of(question id, chunk id) → Judgment | None (never judged),
           agreement_with(other) (later)
Judge      protocol: name, e.g. "quote"; judge(question, chunks) → one question's judgments
```

- **The grade is about one chunk:** "answered" when it supplies any key point. Whether
  every key point was found is the question's, across the top k (the all-key-points rate):
  a chunk holding only the rule still answered, and gets the credit.
- **`key_points` are positions in the question's `key_points`,** a tuple since one chunk
  may supply several (a rule and its exception in one passage). "Answered" with none named
  is allowed, for a judge that can't link (`LlmRelevanceJudge`).
- **No version and no text hash, for now:** judgments live in memory for one evaluation,
  where neither the judge nor the corpus changes. They come back with reusing judgments
  across evaluations ("Later").
- **Why apart:** a judge that is slow, costs something or isn't deterministic (an LLM)
  judges once, and its judgments serve every metric and comparison. An old run can be
  scored again by a new judge.
- **Judgments are keyed by question and chunk, not by run:** comparing two statements
  judges the union of their top k once (pooling). Pooling only saves work: it never
  changes a run's score ("Rules").
- **The run keeps every chunk it returned,** not only the one that matched: a judge added
  later has them all.
- **The matching passage's trail gives per-stage ranks:** which source found it at what
  rank (`Found`), then after fusion and after the rerank.
- **A run holds no text:** chunk ids and trails (ranks and scores) only, so it can be
  logged and diffed. Judges read chunk text by id from the corpus. Hence `ReturnedChunk`
  rather than the engine's `Passage`, which holds its chunk.
- **No questions in the run:** their tags and answerability come from the evaluator, which
  scores the run. A saved run will need them ("Later"); copied in now, they'd be a second
  place to disagree.
- **The fingerprint is a `uuid4` made once per evaluator,** enough to refuse another
  evaluator's run while everything is in memory. A hash of the questions and the corpus
  comes with saving runs.

`Run`, `ReturnedChunk` and `run_statement(prepared, *, questions, fingerprint, where)` are
built in `evaluation/runs.py`, tested in `tests/evaluation/test_runs.py`.

### Judges

| Judge | Asks | Grade | Names the key points | Unlabelled questions |
|---|---|---|---|---|
| `QuoteJudge` (built) | does the chunk match a quote's `where` and contain its text? | answered or unrelated | yes | no |
| `LlmQuoteJudge` (later) | does the passage state what the quote states? | answered or unrelated | yes | no |
| `LlmRelevanceJudge` (later) | does the passage help answer the question? | unrelated, related, answered | no | yes |

`QuoteJudge` is built in `evaluation/judges/quote.py`, `Judgment`, `Judge` and
`Assessment` in `evaluation/judges/protocol.py`, `QuoteMatcher` and `normalize_text` in
`evaluation/matching.py`; tested in `tests/evaluation/judges/` and
`tests/evaluation/test_matching.py`. A `QuoteMatcher` is a quote prepared once (its text
normalised, its `where` built into a matcher); a chunk's text is normalised once. The
quote occurrences use the same matchers, so they and the judge can't disagree about a
match. An unanswerable question has no key points, so every chunk is "unrelated" to it.

### Quote occurrences

```text
QuoteOccurrences     where one gold quote occurs among the corpus's chunks
  question_id, key_point, quote      the quote, by its question and positions (no text)
  chunk_ids_in_place                 chunks matching its `where` whose text holds it
  chunk_ids_elsewhere                chunks outside its `where` whose text holds it
  is_found                           chunk_ids_in_place isn't ()
```

`find_quote_occurrences(questions, *, chunks)` gives one per quote, in question order.
Built in `evaluation/occurrences.py`, tested in `tests/evaluation/test_occurrences.py`.

- **Per quote, not per chunk:** the judge's "this chunk supplies key point 0" can't say
  that the second alternative of key point 0 is a broken label; this can.
- **Two id tuples tell a wrong place from wrong text:** found only elsewhere, the `where`
  is wrong (likely the loader's document-id mapping); found nowhere, the text is wrong,
  split across chunks, or its document isn't in the corpus.
- **A question stays gradable while every key point has a quote found;** a quote not
  found is still listed, for fixing, even when an alternative is.
- **It trusts the evaluator:** question ids unique, each `where` checked against the
  index's schema. The text is checked before the place, so a wrong field raises only on a
  chunk holding the text.

- **The quote judge is cheap and exact,** and judges everything. `LlmQuoteJudge` finds the
  same fact worded differently or stated in another chunk; `LlmRelevanceJudge` needs no
  labels, which lets real users' questions in.
- **An LLM judge is checked against the quotes:** `by_llm.agreement_with(by_quote)` gives the
  agreement and the disagreements where both judged, saying whether to trust it where only
  it judges.
- **One `Assessment` holds one judge's grades:** grades from two judges aren't comparable;
  combining judges is a new judge with its own name.
- **An LLM judge's output that doesn't parse raises;** the pair stays unjudged, never a
  made-up 0.
- **The LLM protocol is the agents package's** (planned); prompts are the app's, in the
  domain's language.

### The toolkit: three classes

```python
evaluator = Evaluator.from_index(questions, index=index)  # checks, reads, finds once

evaluator.quotes_not_found  # QuoteOccurrences: labels or chunking to fix
evaluator.gradable_questions  # the ones every metric counts

dense = evaluator.run_statement(engine.prepare(dense_only))
reranked = evaluator.run_statement(engine.prepare(dense_then_rerank))

assessment = evaluator.judge_top_chunks(
    [dense, reranked], depth=10, judge=QuoteJudge()
)

scorecard = evaluator.score(reranked, assessment=assessment)
scorecard.hit_rate_at(5)  # any cut-off up to the judged depth, 10
scorecard.mean_reciprocal_rank_at(10)
scorecard.all_key_points_rate_at(10)  # every key point in the top 10
scorecard.found_by_source_rate("dense")  # retrieval reached it at all
scorecard.count_rerank_moves()  # up / down / unchanged
scorecard.top_scores_by_answerability()  # for abstention
scorecard.questions_missed_at(10)  # the rows to look into
scorecard.questions_left_out  # a quote not found: not counted by any metric
scorecard.for_tag("paraphrase").hit_rate_at(5)  # any metric, one kind

comparison = evaluator.compare(reranked, baseline=dense, assessment=assessment)
comparison.questions_improved_at(5)  # RankChanges: the question and both ranks
comparison.questions_worsened_at(5)
comparison.questions_unchanged_at(5)  # missed by both included
comparison.for_tag("paraphrase").questions_worsened_at(5)
```

| Class | Holds | Offers |
|---|---|---|
| `Evaluator` | questions, the corpus (every chunk, by id, read once from the index), where each quote occurs, the fingerprint | `quotes_not_found`, `gradable_questions`, `run_statement`, `judge_top_chunks`, `score`, `compare` |
| `Scorecard` | one run's top chunks joined with their judgments, question by question; the questions left out; the judged depth | `hit_rate_at`, `mean_reciprocal_rank_at`, `all_key_points_rate_at`, `found_by_source_rate`, `count_rerank_moves`, `top_scores_by_answerability`, `questions_missed_at`, `questions_left_out`, `tags`, `for_tag` |
| `Comparison` | two runs' scorecards, from one assessment: the candidate and its baseline | `questions_improved_at`, `questions_worsened_at`, `questions_unchanged_at`, `for_tag` |
| `Run`, `Assessment` | plain data | saved, loaded, passed around |

- **Three classes, not one:** the evaluator holds what every step shares; a scorecard holds
  one run and its judgments, so its metrics take only their cut-off. One class holding every
  metric would take a run and judgments on each call.
- **`for_tag` instead of a `tag=` on every metric:** it returns a scorecard over that subset,
  so every metric, those added later too, works per tag.
- **Quotes are found once, when the evaluator is built,** and the result stays visible:
  `quotes_not_found` lists the quotes no chunk in their place holds, and a scorecard reports
  the questions it left out. A wrong label never counts as a retrieval miss.
- **Built by `Evaluator.from_index`, frozen:** it checks the questions (not empty, ids
  unique, each `where` against the schema) before reading the index, then keeps its own
  copy of the questions and the corpus: every chunk, by id, read once. The corpus is a
  snapshot. The index itself isn't kept, and nothing checks that a statement searches the
  same one: the caller prepares both from one index.
- **Gradable questions** have a quote found for every key point; an unanswerable question
  has none to find, so it's gradable, for the abstention scores. Every question is run,
  gradable or not: what's left out is decided when scoring.
- **Each run records the evaluator's fingerprint**, a `uuid4` for now; `score` and
  `compare` refuse a run from another evaluator, so a comparison is always on the same
  corpus and questions.
- **The classes are thin:** the logic lives in the modules below and is tested there.

### The scorecard

```text
Scorecard        statement, questions: tuple[ScoredQuestion, ...],
                 questions_left_out: tuple[Question, ...], judged_depth
ScoredQuestion   question, chunks: tuple[RankedChunk, ...]
                   (best first, at most judged_depth)
                 first_answered_rank, all_key_points_rank: from 1, or None
                 first_answered_rank_within(k): the same, None below k
                 is_answered_within(k), has_all_key_points_within(k)
RankedChunk      returned: ReturnedChunk, judgment: Judgment; is_answered
```

`evaluator.score(run, *, assessment)` joins the run, the assessment and the questions once,
so every metric reads one row per question. It refuses a run from another evaluator, and a
run whose top `judged_depth` chunks aren't all judged: one that wasn't passed to
`judge_top_chunks`. Built in `evaluation/scorecard.py` (`hit_rate_at`,
`mean_reciprocal_rank_at`, `all_key_points_rate_at`, `questions_missed_at`,
`first_answered_ranks_within`, `answerable_questions`, `for_tag`, `tags`) and
`evaluation/evaluator.py` (`score`), tested in
`tests/evaluation/test_scorecard.py` and `tests/evaluation/test_evaluator.py`.

- **A join, but each unit kept whole:** a row holds its `Question`, and each chunk its
  `ReturnedChunk` and `Judgment`, never copies of their fields. Copies would be a second
  place to disagree and would drop the checks those types make.
- **Rows only for gradable questions,** so no metric can forget the filter. The others are
  kept as `questions_left_out`, `Question`s rather than a count, so `for_tag` can filter
  them too. Unanswerable questions are rows: abstention reads them.
- **Rows hold only the top `judged_depth` chunks, every one judged,** so a judgment is
  never `None` on the card and no metric reads below the depth.
- **A metric checks its own subset:** a card of any questions, even none, can be built, and
  a rank metric with no answerable question to rank raises rather than returning 0. A card
  of unanswerable questions still serves abstention.
- **Each metric with a cut-off reads one predicate of the row** (`is_answered_within`,
  `has_all_key_points_within`), so the hit rate and the questions missed can't drift
  apart: the missed are exactly the questions the hit rate doesn't count.
- **The all-key-points rank compares sets, not counts:** the key points named so far must
  include every position, so a judgment naming a position the question lacks can't
  complete it. `judge_top_chunks` refuses such a judgment anyway; an assessment built
  some other way doesn't pass that check.
- **`questions_missed_at` returns the rows,** not ids: a miss is looked into through its
  chunks, judgments and trails.
- **`for_tag` returns a card over the tag's rows and left-out questions,** with the same
  statement and depth, so every metric works per tag. A tag whose questions were all left
  out gives a card with no rows, which says its labels need fixing; a tag no question
  has is refused, to catch a typo. `tags` is worked out from the questions, never stored.
- **Planned for the stage metrics** (`found_by_source_rate`, `count_rerank_moves`): they need
  chunks the rerank left below the depth. Each run then pools the top `depth` of each of
  its own stages, so those chunks are judged because of the run itself, never because of
  the runs pooled with it.

### The comparison

```text
Comparison   candidate: Scorecard, baseline: Scorecard
             questions_improved_at(k), questions_worsened_at(k), questions_unchanged_at(k)
               → tuple[RankChange, ...], answerable questions only, in question order
             for_tag(tag) → Comparison
RankChange   question, baseline_rank, candidate_rank: from 1, or None (missed within k)
             is_improvement, is_regression
```

`evaluator.compare(candidate, *, baseline, assessment)` checks both runs are its own and
scores both on the one assessment. Built in `evaluation/comparison.py` and
`evaluation/evaluator.py` (`compare`), tested in `tests/evaluation/test_comparison.py` and
`tests/evaluation/test_evaluator.py`.

- **A question's outcome is its first answered rank within k,** the rank the scorecard's
  metrics read; past k it counts as missed, the worst rank. Any rank move and crossing a
  cut-off are then one rule: at the judged depth every move counts, at a small k (the
  passages an answer reads) only the moves that matter there. 3 → 2 is unchanged at
  `k=1`, improved at `k=3`.
- **The changes account for the scorecards' difference:** the questions improved and
  worsened at k are exactly those whose 1/rank changed, so they make up the difference in
  `mean_reciprocal_rank_at(k)`, and those that crossed k make up the difference in
  `hit_rate_at(k)`. A test checks the first at every k.
- **Lists of `RankChange`, not counts:** the comparison is for finding which questions to
  look into, and "3 → 1" says more than "improved". A count is `len(…)`.
- **Unchanged includes missed by both,** so the three lists hold every answerable question
  once.
- **Built on two scorecards, checked by `compare`:** the comparison reads ranks through
  `Scorecard.first_answered_ranks_within`, so the cut-off is checked as every metric checks
  it, and `for_tag` is each card's. `compare` scores both runs on one assessment, which
  gives them the same judge, judged depth and questions; built directly, nothing checks
  that, as with `Scorecard`. Ranks are paired by question id, never by position.
- **Direction at the call site:** `compare(reranked, baseline=dense, …)`: the run tried is
  positional, the one it's measured against is named.
- **Small sets are noisy:** with about 50 questions, one 2 → 1 flip is weak evidence. The
  lists say what to look into; paired tests come later ("Later").

### Rules

- **A chunk holds a quote** when it matches the quote's `where` and its `text`, never its
  `embedded_text` (which adds the headings path), contains the quote after NFC,
  case-folding and collapsing whitespace. Diacritics are kept.
- **Matching a filter in Python** is `filters.build_chunk_matcher` (built), used by the
  in-memory index and the quote judge, so a filter means the same in search and grading.
  Each quote's `where` is checked against the index's schema once, when the evaluator is
  built: a chunk doesn't know its schema, so the matcher can't check it.
- **A run asks for every candidate:** the limit is the sum of the sources' depths, worked
  out from the statement rather than given (after a merge, the union can exceed any one
  depth but never their sum). A metric takes its own cut-off, so no caller needs fewer, and
  a smaller limit would hide a chunk a source found but the rerank left low. The engine runs with `AlwaysValid` on a
  fixed corpus, so no gate hides a chunk; `where` is `Everything()` unless the caller
  evaluates under one asker's access.
- **A question's rank** is the best rank of any chunk supplying a key point (hit@k, MRR);
  the all-key-points rate needs every key point in the top k. Hit rate counts "answered"
  as relevant; nDCG, once grades are graded, maps the three grades to gains.
- **Unanswerable questions** are left out of the rank metrics and appear only in the
  abstention scores.
- **A metric's cut-off is at most the assessment's `judged_depth`,** and pooling never
  changes a run's score. Within the depth, every chunk of a judged run is judged whatever
  else was pooled; below it, a chunk may be judged only because another run ranked it
  higher, so a score read there would depend on which runs were judged together. To score
  deeper, judge deeper: the quote judge is cheap.
- **A metric that needs key point links** (the all-key-points rate) refuses a card with an
  answered chunk that names none (`LlmRelevanceJudge`'s), rather than counting it as
  supplying nothing. It checks the whole card, not the top k: naming key points is the
  judge's way, so the rate is readable at every k or at none. Unanswerable questions are
  skipped, since they have no key points to name.
- **The old/new gate** compares the new numbers with the old evaluator's passage level,
  which also required the reference key: quote-only grading is slightly more lenient.

### Layout

```text
evaluation/
  __init__.py       the public names
  questions.py      Question, KeyPoint, Quote, and their checks (built)
  evaluator.py      Evaluator: from_index, quotes_not_found, gradable_questions,
                    run_statement, judge_top_chunks, score, compare (built)
  runs.py           Run, ReturnedChunk, run_statement: running a prepared statement
                    (no grading) (built)
  judges/
    protocol.py     Judge, Judgment, Assessment (built); agreement
    quote.py        QuoteJudge (built)
    llm.py          the LLM judges (later, over the agents package's LLM protocol)
  matching.py       QuoteMatcher, normalize_text (built): what "a chunk holds a quote" means,
                    shared by the judge and the occurrences
  occurrences.py    QuoteOccurrences, find_quote_occurrences (built)
  scorecard.py      Scorecard, ScoredQuestion, RankedChunk: the metrics over one run and
                    its judgments (built: hit_rate_at, mean_reciprocal_rank_at,
                    all_key_points_rate_at, questions_missed_at, for_tag, tags)
  comparison.py     Comparison, RankChange: two runs, question by question (built)
```

Dependencies point down: `runs` alone uses the engine (`PreparedStatement`, `Passage`);
`scorecard` and `comparison` read runs and judgments, so they're tested without an engine.
Nothing in `engine/` imports `evaluation/`. Tests mirror the layout under
`tests/evaluation/`, with invented questions and text only (`FakeDenseEmbedder`,
`FakeReranker`, `InMemoryIndex`): the real question set never appears in a test. The
scorecard and comparison tests build their rows by hand with `tests/evaluation/support.py`.

### Security

- Grading reads document text, locally only. Runs and judgments hold ids, ranks, scores
  and grades; a miss is shown by question id, key point position and chunk ids, never
  passage text.
- An LLM judge sends question and document text to a model: for the regulation documents
  it must be a local model (Ollama) unless approved otherwise. The framework takes any
  model; the app's wiring keeps it local.
- A judge's reasoning quotes text, so it never goes in a `Judgment`; if kept to debug
  disagreements, it goes to its own file under `data/`.

### Later

- What a run ran: each role's model identity (statements name roles, not models), a label,
  and the corpus and questions fingerprints in saved runs.
- Saving and loading runs and judgments (a format for `Statement` and trails, versioned).
- Reusing LLM judgments across evaluations (a cache, or known judgments passed back in), so
  each round judges only new pairs and the judge's noise doesn't enter comparisons; with
  it, how a reused judgment is known to be current: the judge's version (its prompt and
  model), and a hash of the chunk's text, since a re-chunked chunk can keep its id.
- Wording variants (`variant_of` on `Question`), per-question scope (`scope: Filter`),
  timing per question.
- Confidence: paired tests in `Comparison`, bootstrap intervals in `Scorecard`.
- People's judgments as a judge (`HumanJudge`, from a file); misses counted by stage
  (facts, retrieval, rerank).
- Text that must not appear (superseded versions, false premises), follow-up pairs,
  context-level and answer evaluation, a saved baseline to compare new runs with.
