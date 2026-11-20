# Knowledge base backend: design guide

The shape of the backend, the CLI and how they sit on `document_processor`. Each decision
comes with the reason for it; section 15 lists the ones that are easy to undo by accident.

Status markers: **(built)** exists on the `apps` branch; **(next)** is being worked on;
everything else is planned.

---

## 1. Repository

```text
pyproject.toml              uv workspace root; document_processor (the framework) stays here
src/document_processor/     the framework: adapters, cleaning, profiles, chunking, preparation, retrieval
tests/                      the framework's tests
apps/
  server/                   package kb-server → document_processor (workspace), FastAPI, SQLAlchemy,
                            Alembic, psycopg, pydantic-settings, argon2-cffi, boto3;
                            extra [models] → torch, sentence-transformers, qdrant-client (search);
                            extra [worker] → [models] + the framework's preparation
  cli/                      package kb-cli → httpx, Typer only (no server or framework imports)
docker/docker-compose.yml   postgres, qdrant (built); minio later
data/                       git-ignored: restricted documents and question sets, never committed
```

## 2. Server package

```text
apps/server/
  pyproject.toml            scripts: kb-server = "kb_server.manage:main"
  alembic.ini
  migrations/               env.py (URL from Settings, target_metadata = Base.metadata), versions/
  src/kb_server/
    settings.py             Settings (pydantic-settings, env prefix KB_, .env next to where it runs)
    wiring.py               Services: builds what every process needs; open_job_tools, the worker's
                            tools; open_search_tools and build_search_service, the API's (built)
    app.py                  create_app(services, search_service) → FastAPI; the lifespan loads the
                            search models once (built)
    manage.py               kb-server create-admin | serve | worker (built); migrate | reprepare | gc
    api/                    routers + request/response schemas; HTTP only
    services/               use cases and permission checks; no HTTP, no SQL
    search/                 the search statement (DenseThenRerank) and the validity adapter
                            (ServedVersionsOnly), plugged into salemdex_retrieval (built)
    domain/                 frozen dataclasses and errors: User, Principal, Document, StoredFile, …;
                            chunks.py: the index schema (Guarded, KbChunks), read by the worker and search
    storage/                SQL and files only
      database.py  models.py  repositories/  files.py
    indexing/               the worker's side (models load only here)
      prepared.py           PreparationSettings, PreparedParts, save/load/delete prepared output (built)
      handlers.py           ProfileChooser, ClaimedJob, JobTools, run_prepare_job, run_index_job,
                            run_delete_job (built)
      worker.py             Worker: run_once, run_forever, recover_stuck_jobs (built)
  tests/                    api/ services/ storage/ indexing/ domain/ test_wiring.py
```

### Layering

- `api` → `services` → `domain` and `storage`; `services` → `search` → `storage` and
  salemdex_retrieval; `indexing` → `storage` and the framework. Nothing imports upward. Why: each layer can be tested without the ones above it.
- Routers convert `XIn` → arguments, call one service method, convert the result → `XOut`.
- Services check permissions and raise domain errors. Repositories only report (`None`, `bool`,
  counts); they never raise for "not found". Why: one place decides what's an error and which one.
- Repositories take a `Session` and return domain dataclasses, never rows.
- One transaction per request: commit on return, roll back on raise. Two exceptions, so that no
  connection is held while a model runs: authentication (`get_principal`) commits in its own
  short transaction before the route runs, and `SearchService` lives for the whole app and
  opens a short transaction for each read (the asker's groups, validity at each gate, the
  titles). Why: a request holding one connection while waiting for another can exhaust the
  pool (hold-and-wait), and a search spends seconds in the models.
- Job handlers never touch the database: they return a result, and the worker records it. Why:
  handlers can be tested with only a file store.

### Kinds of objects

| Kind | Use for | Naming |
|---|---|---|
| Pydantic `BaseModel` | HTTP bodies | `DocumentIn`, `DocumentOut` |
| Pydantic `BaseSettings` | configuration from the environment | `Settings` |
| frozen `@dataclass` | domain values, results | `Document`, `Principal`, `PrepareResult` |
| SQLAlchemy model | tables, inside `storage/` only | `DocumentRow` |
| class | services, repositories, stores | `DocumentService`, `JobRepository` |
| `Protocol` | swappable parts | `FileStore`, the framework's `Chunker`, `Embedder` |
| function | pure helpers | `original_key`, `prepared_version_folder` |

### Parameters by role

A function's parameters usually play three roles: the thing being worked on (first,
positional), the settings and the tools (keyword-only). `run_prepare_job(job, *, chooser,
store)`: this job, prepared this way, with this store.

---

## 3. Settings (environment)

```text
built      KB_DATABASE_URL  KB_LOG_LEVEL  KB_SESSION_TTL_HOURS  KB_ACT_FOR_HEADERS
           KB_MAX_UPLOAD_MB  KB_STORAGE_KIND (local | s3)  KB_STORAGE_PATH
built,     KB_PROFILES (candidates, most specific first)  KB_FALLBACK_PROFILE (plain)
worker     KB_CHUNK_WORDS  KB_MODELS_FOLDER  KB_EMBEDDER_SPEC  KB_EMBEDDER_DEVICE
           KB_EMBEDDER_BATCH_SIZE  KB_QDRANT_URL  KB_QDRANT_COLLECTION  KB_QDRANT_API_KEY
           KB_WORKER_POLL_SECONDS  KB_JOB_MAX_ATTEMPTS  KB_JOB_TIMEOUT_MINUTES
built,     KB_MODELS_FOLDER  KB_EMBEDDER_*  KB_QDRANT_* (as above)  KB_RERANKER_SPEC
search     KB_RERANKER_DEVICE  KB_RERANKER_BATCH_SIZE  KB_RERANK_DEPTH  KB_SEARCH_MAX_LIMIT
           (≤ KB_RERANK_DEPTH, checked when Settings load)
later      KB_S3_ENDPOINT  KB_S3_BUCKET  KB_S3_ACCESS_KEY  KB_S3_SECRET_KEY
           KB_OLLAMA_URL  KB_CHAT_MODEL  KB_CHAT_CONTEXT_SIZE  KB_CHAT_TIMEOUT
           KB_ANSWER_PASSAGES  KB_MIN_SCORE  KB_MIN_PASSAGE_SCORE  KB_SECTION_CHARS  KB_HISTORY_TURNS
```

`.env` holds the database password: it is git-ignored and never committed.

- Model and Qdrant settings have defaults or are unset, so `create-admin` and the other
  commands start without them; `serve` and the worker check the ones they need when they
  build their tools.
- Tuple settings are JSON in the environment: `KB_PROFILES=["vn_regulation"]`.
- `KB_MODELS_FOLDER` holds every model, each at `<hub id>/<commit hash>` (e.g.
  `BAAI/bge-m3/5617a9f…`), found from its spec by the framework's `model_folder`. Why: one
  setting per model would have to be changed with its spec and could point at the wrong
  commit; a folder named by its reference can't, and two commits sit side by side while
  re-embedding. Use an absolute path: the server runs from `apps/server`.
- `Settings` never imports the framework's model code: every process reads it.

---

## 4. Database

- SQLAlchemy 2.0 ORM, synchronous, psycopg 3. Code stays database-agnostic: no driver-specific
  internals.
- Alembic for every schema change; constraint naming convention on `MetaData`.
- Tests migrate a real Postgres once with Alembic, then run each test in a transaction that is
  rolled back.
- Ids: UUIDs. Timestamps: `timestamptz`.
- Columns are added by migration when a feature first uses them, not ahead of time.
- Enum columns are `VARCHAR` + `CHECK` (`native_enum=False`). Autogenerate doesn't see a new
  value, so its migration is written by hand, and a test writes every value of each `Literal`.
  Adding a value is one-way: the downgrade keeps the wider constraint. Why: narrowing it would
  fail or lose the rows that use the value. Rolling back the code still means finishing or
  clearing those rows first, because the ORM refuses a value it doesn't know.

### Tables

```text
built
users                 id, email (unique, lower-case), display_name, password_hash, role (reader|manager|admin),
                      is_active, created_at
groups                id, name (unique, lower-case), created_at      (`everyone` is seeded; new users join it)
user_groups           user_id, group_id                              pk (user_id, group_id)
credentials           id, user_id, kind (session|api_key), name, prefix, token_hash (unique),
                      act_for_users, created_at, last_used_at, expires_at, revoked_at
collections           id, slug (unique), title, description, created_at
collection_groups     collection_id, group_id (restrict)             default read groups for new documents
collection_managers   collection_id, user_id
stored_files          id, sha256 (unique), size, media_type, storage_key, created_at
documents             id, collection_id (restrict), title, status (draft|active|deleting), current_version,
                      indexed_version (nullable), created_by, created_at, updated_at
document_versions     document_id, version, file_id (restrict), original_filename, provenance (json, nullable),
                      uploaded_by, created_at                        pk (document_id, version)
document_groups       document_id, group_id (restrict)               who may read it
indexing_jobs         id, kind (prepare|index|sync_attributes|delete), document_id (cascade), version (nullable),
                      status (queued|running|done|failed), attempts, error, created_by,
                      created_at, started_at, finished_at            index (status, created_at)

later
document_preparation  document_id (pk), profile, set_by, set_at      a manager's profile override
tags, document_tags, document_attributes
document_relations    source_id, target_id, kind (replaces|amends|guides|appendix_of|based_on)
retirements, retirement_documents
upload_batches, batch_groups
answers, feedback
```

There is no `profile` column on `documents`: the profile is chosen automatically when
preparing (section 7), and an override lives in its own table. Why: most documents never need
one, and the override is a decision with its own author and time.

### Rules

- Read access: a reader sees a document when they share a group with it. Managers of its
  collection and admins see all of the collection's documents. A document with no groups is
  visible only to them; public means the `everyone` group.
- A document's own groups are what's checked; a collection's groups are only copied to new
  documents. New document's groups: `group_ids` given → those; left out → the collection's
  defaults; `private=true` → none.
- Readers see a collection when they share one of its default groups. A hidden collection or
  document is a 404, never a 403. Why: a 403 would confirm that it exists.
- Managing a collection needs the manager role and a `collection_managers` row (admins manage
  all; a front end acting for someone manages nothing). Uploading needs `can_manage`.
- A collection with documents can't be deleted; a group in use can't be deleted (`restrict`).
- The same file (sha256) as a document's current version in the same collection → 409
  `duplicate_file` with that document's id.
- A single upload is `active` at once; `draft` is for batches.
- Changing groups, tags or attributes → a `sync_attributes` job, no re-embedding. Changing the
  title changes only the database: the title isn't on the chunks (section 5).
- A new file → a new version → a `prepare` job. A profile override → a `prepare` job for the
  current version.
- Deleting a document removes it everywhere (section 8, "Deleting").

---

## 5. Index (Qdrant): the chunk schema

```text
built, on salemdex_retrieval   the schema below (domain/chunks.py)
later                          tags   retired_on   part
```

```python
class Guarded(Schema):  # every searchable schema inherits it
    access_group_ids = Many(UUID, indexed=True)
    collection_id = One(UUID, indexed=True)
    document_id = One(UUID, indexed=True)
    document_version = One(int)


class KbChunks(Guarded):
    position = One(int)
    headings = Many(str)  # where the passage sits, for citing it; empty for `plain`
    dense = Dense(size=1024, distance="cosine")
```

- **A field goes in `Guarded` when a guarantee slot reads it from every source:** the access
  filter (`access_group_ids`, `collection_id`) and validity (`document_id`,
  `document_version`). Why: a session holds one filter and the engine one validity check, and
  both apply to every index a statement searches. A field on one schema only would be refused
  on the others (retrieval design, section 6, "One filter for every schema"). Generated
  entries will inherit it too.
- **The access filter, by asker:**

  ```python
  reader = Guarded.access_group_ids.any_of(groups)
  manager = or_(reader, Guarded.collection_id.in_(managed_collections))
  admin = Everything()
  ```

  Why `collection_id`: managers see every document of the collections they manage, private
  ones included (section 4, "Rules"), and a private document has no groups. An empty `in_`
  matches nothing, so a manager of no collection sees what their groups allow. A front end
  acting for someone gets that user's reader or manager filter, never `Everything()`
  (`is_admin` is false while acting for someone).
- **Search filters inside the index.** Why: filtering only after the search loses results and
  reveals that hidden hits exist.
- **Indexed fields are the ones filtered by:** groups and collection (every search),
  `document_id` (replacing and deleting a document's chunks). `document_version` isn't:
  validity reads it from the hit, and deleting a document's other versions is narrowed by
  `document_id` first.
- **Chunk ids:** `uuid5(document_id, f"{version}:{position}")`. The version is in the id, so a
  new version's chunks never overwrite the old one's while both are in the index.
- **Kept off the chunks:** `status` and the indexed version (their truth is in the database,
  so validity checks them: a hit is kept when its document is `active` and its version is
  `indexed_version`), and `title` (read from the database when passages are shaped, so a
  rename doesn't touch the index). Why: a condition whose truth lives in the database would go
  stale on the chunks.
- **`indexed_version` is the version search shows** (built). While a new version is being
  indexed, the index holds two versions of a document; validity shows only `indexed_version`,
  and the worker deletes the old chunks after the commit that moves it (section 8, "Index
  job"). A document never indexed has `NULL`, and search shows none of it. Why not
  `current_version`: it changes at upload, so a re-index would hide the document for minutes,
  and a version that fails to index would hide it for good.

---

## 6. File storage

```python
class FileStore(Protocol):
    def put(self, key: str, data: BinaryIO, *, size: int, media_type: str) -> None:
        ...
        # keeps whatever is already at the key

    def open(self, key: str) -> BinaryIO: ...  # missing → FileNotFoundError
    def exists(self, key: str) -> bool: ...  # files only
    def delete(
        self, key: str
    ) -> None: ...  # one file; missing → nothing; a folder → ValueError
    def delete_folder(self, folder: str) -> int: ...  # every key under folder + "/"
```

- A folder is a key without its last `/name` part. On S3 a folder exists only as the start
  of its files' keys.
- `LocalFileStore` (built) writes atomically: a temporary file in the same folder, `fsync`,
  `os.replace`. It refuses keys outside its root. `S3FileStore` later, chosen by
  `KB_STORAGE_KIND`.
- **Checking the store can be written (built):** `check_store_writable` puts a small file at
  `write_checks/<uuid>/probe` and deletes that folder; any `OSError` becomes
  `StoreNotWritable`, with the cause chained. `kb-server serve` and `kb-server worker` run it
  at startup and exit with one line; the worker runs it before loading any model. `GET /ready`
  runs it on every call and answers `503 {"status": "storage not writable"}`, logging the
  reason rather than returning it. Why both: the startup check catches a wrong folder when a
  person starts the server; `/ready` catches one that breaks later (a full disk, a volume
  remounted read-only) and covers deployments that start uvicorn directly. Why a new key each
  time: `put` keeps whatever is already at a key, so a fixed key would pass without writing.

### Layout

```text
originals/sha256/<aa>/<bb>/<sha256>          the uploaded file, addressed by its content
prepared/<document_id>/v<version>/           one version's prepared output
    logical.json  chunks.jsonl  review.html
```

- Originals are content-addressed: the same bytes are stored once, and a key can't point at
  the wrong content. Several versions may share one original, so it's deleted only when no
  `stored_files` row needs it.
- Prepared output belongs to its version, not to the file's content. Why: it retires with the
  document (no cache that grows forever), and nothing is silently reused after settings or
  code change. Re-preparing replaces the folder: `save_prepared` deletes it first, so a save
  that stops halfway loads as nothing, never as old and new files mixed.
- Only `indexing/prepared.py` knows this layout: callers pass a document id and version.

### Upload order (built)

```text
1. check the caller may manage the collection (before reading the file)
2. stream to a temporary file: sha256 + size, enforce KB_MAX_UPLOAD_MB → 413
3. check the type by content: .docx (a zip with word/document.xml) or .txt (UTF-8, no NUL) → else 415
4. duplicate check → 409
5. FileStore.put
6. one transaction: stored_files (if new) → documents → document_versions → document_groups
   → indexing_jobs (prepare)
→ 201 DocumentOut with the queued job
```

The file is written before the transaction: a failed transaction leaves an unused file, never a
row without its file. Writing the document and its job in one transaction means there is never
a document without its job (the outbox idea).

---

## 7. Preparation and profiles

- Profiles describe how a kind of document is structured. Built in: `vn_regulation`
  (Chương, Điều, khoản…) and `plain` (no levels: chunked by size only).
- Chunks are sized in words (`KB_CHUNK_WORDS`, 450 by default: about 512 bge-m3 tokens for
  Vietnamese), so preparation needs no model's tokenizer. Nothing yet checks that a chunk
  fits a model: sentence-transformers truncates a longer text without warning. Checking fit
  against each model by its own tokenization comes later (retrieval design, section 6,
  "Chunk length and the models").
- **Automatic choice:** `KB_PROFILES` lists candidates; each is tried in order, and the first
  whose outline has a section wins. If none does, `KB_FALLBACK_PROFILE` (`plain`) is used.
  List candidates from most specific to most general.
- First-match is right while there is one structured profile. When a second arrives, switch to
  scoring every candidate (coverage, numbering in sequence, sensible depth, warnings), checked
  against a small labelled set of documents. List order then only breaks ties.
- `ProfileChooser.from_profiles(...)` is the only way to build the chooser, so its preparers and
  its `PreparationSettings` always agree. Unknown profile names fail at worker startup.
- `PreparationSettings`: candidate profiles and fallback (`name@version`), and the chunker: its
  type, options and tokenizer (`"tokenizer": "words"`). The tokenizer is recorded by a fixed
  name per class, so versions sized in tokens and in words differ for `reprepare --stale`,
  and renaming a class doesn't make every version look stale. No embedder: words need no
  model's tokenizer.
- Each preparation saves a review page (`review.html`) next to its parts, for a person to check
  the outline and chunks.

---

## 8. The worker and its jobs

### Rules that keep it safe

The worker writes to three places that share no transaction: Postgres, the file store and the
index. These rules make a crash at any point harmless:

1. Every step is safe to repeat (idempotent consumer): delete-then-write, put-if-absent,
   replace-all-chunks. A job may run more than once (at-least-once delivery).
2. The database is the truth; prepared output and the index can be rebuilt from it.
3. Do the work first, then record it in one transaction.
4. Order writes so a crash leaves something harmless: an orphaned file, never a row pointing
   at nothing.
5. One worker: jobs never race. A change during a job queues a follow-up job, which runs after.

### Decisions

- W1: separate `prepare` (no models) and `index` (embeds) jobs; prepared output is saved
  between them (a claim check: the index job loads it by reference).
- W2: index attributes are `access_group_ids`, `collection_id`, `document_version` (section 5).
- W3: retries up to `KB_JOB_MAX_ATTEMPTS`; `PermanentJobError` fails a job at once.
- W4: one worker; a job `running` longer than `KB_JOB_TIMEOUT_MINUTES` is requeued at startup
  (a lease that expired).
- W5: models and the Qdrant client load only in the processes that use them: the worker
  (`open_job_tools`) and `serve` (`open_search_tools`, in the app's lifespan, for search).
  They're installed by the `kb-server[models]` extra (`[worker]` includes it) and imported
  inside those functions, so importing the app and the other commands never load them; a test
  checks it in a fresh interpreter.
- W6: a superseded job (its version isn't current, or the document is `deleting`) is marked done
  without doing anything. A job with no version (`sync_attributes`) applies to the whole
  document and is never superseded by a new version.

### Running one job (built)

```text
transaction 1   claim the oldest queued job (running, attempts + 1, started_at)
                load its document, current version, stored file and groups
                  a row missing → fail the job, claim the next one
                superseded → mark it done, claim the next one
                → commit
no transaction  run the handler for its kind
transaction 2   success → provenance, follow-up job (prepare of an active document → index), done
                  an index job also sets indexed_version, and after the commit its document's
                  other versions are removed from the index ("Index job" below)
                  a delete → its own steps instead ("Deleting" below)
                PermanentJobError, or attempts ≥ KB_JOB_MAX_ATTEMPTS → failed
                any other error → queued again, the error kept
```

- The claim commits before the handler runs. Why: a crash or a failing handler must not roll
  the claim back. If it did, `attempts` would never grow and the same job would be claimed
  forever. It also keeps a transaction from staying open while a model embeds.
- Loading failures and superseded jobs are handled inside transaction 1, so `run_once` returns
  false only when no job is waiting, and the loop sleeps only then.
- The worker takes a `sessionmaker`, its `JobTools` (chooser, index tools, file store), the
  maximum attempts and a clock, not `Settings`. It opens one session per transaction.
- Any `Exception` from a handler is recorded on the job (`Type: message`). Logs carry the type
  only, because a message could quote document text. `KeyboardInterrupt` and `SystemExit` pass
  through: the job stays `running` and the stuck-job sweep requeues it.
- If recording a success fails, the error propagates and the job stays `running`. The sweep
  requeues it, and repeating the job is harmless (rule 1).

### Running the worker (built)

```text
kb-server worker   logging (KB_LOG_LEVEL) → open_job_tools → recover_stuck_jobs
                   → run_forever until SIGTERM or SIGINT → close the Qdrant client and the engine
```

- `open_job_tools` builds the worker's tools in a fixed order: the embedder and its identity
  (`huggingface:<hub id>@<commit>`, built from the spec, since the embedder doesn't name
  itself), then the profile chooser (chunks in words), then the
  index of `KbChunks` (`QdrantIndex.open`). It's a context manager, so
  the Qdrant client is closed however the worker stops, and callers never see Qdrant.
- `run_forever` runs jobs back to back, waits `KB_WORKER_POLL_SECONDS` only when none is
  waiting, and checks the stop event between jobs; an idle worker stops at once. A signal
  during a job lets it finish. A process killed mid-job (SIGKILL after Docker's grace period)
  leaves it `running`, and the sweep recovers it.
- The stuck-job sweep requeues jobs `running` for longer than `KB_JOB_TIMEOUT_MINUTES`,
  counted from `started_at`, and fails those on their last attempt: a job that kills the
  worker would otherwise be retried forever (a poison message).
- A database error ends `run_forever`, and the process exits. Restarting it (Compose or
  systemd) runs the sweep, which recovers the job it was running.
- Startup errors exit with one line, not a traceback (`ToolSetupError`): the step that
  failed, then the underlying reason. A message names a `KB_*` variable only where the wiring
  checks that variable itself, so it can't go stale when a library fails for another reason.
- The Qdrant payload indexes come from the schema: `QdrantIndex.open` creates one for each
  `indexed=True` field of `KbChunks`. A test pins which fields those are.
- Known gap: nothing checks that the collection's vectors came from the configured model.
  `Dense(size=1024)` catches a model of another size, not another model of the same size or
  a new revision.

### Prepare job

```text
1. media type → extension (.docx / .txt); anything else → PermanentJobError
2. copy the original into a temporary folder as document.<ext>; missing → PermanentJobError
3. ProfileChooser.prepare: OSError → retry; any other error → PermanentJobError
   (preparation gives the same result for the same bytes and code)
4. save_prepared, then review.html
5. log warning counts (ids and codes only); return PrepareResult
```

`PrepareResult`: profile, profiles without structure, settings, and whether to index next (the
document is `active`). The worker then, in one transaction: sets the version's provenance to
`{"preparation": …}` (dropping any old `index` part), marks the job done, and queues `index`.

### Index job

```text
handler  1. load_prepared; missing → PermanentJobError ("prepare it again")
         2. to KbChunks chunks: id uuid5(document_id, "<version>:<position>"), the Guarded
            fields (access_group_ids sorted), position, headings; embedded text = headings
            path + text
         3. embed_chunks, then index.replace(where=document_id AND version N): this version's
            chunks only; other versions are left for search to keep showing
         4. log the chunk count; return IndexResult
worker   5. one transaction: provenance["index"], indexed_version = N, job done   (the flip)
         6. after the commit: index.delete(where=document_id AND version ≠ N)
```

This is the pointer-flip pattern: write the new copy, switch one pointer, remove the old copy.

- **Each step's filter says which copy it owns.** The handler touches only version N, the
  worker's cleanup only the other versions, the delete job every version. Nothing filters by
  `indexed_version` before the flip, because that's the copy readers are looking at.
- **Readers see all of the old version, then all of the new one:** never a mix of versions, never
  nothing. Re-indexing the same version is the exception: its chunks are overwritten by id, so
  for a moment old and new chunks of that version mix. Accepted: it's rare and brief.
- **`replace`, not `put`, within the version,** so a version prepared again with fewer chunks
  leaves none of its old ones behind; validity would show them, since their version matches.
- **Step 6 never fails the job,** which is already recorded. Any `Exception` is logged (the
  document, the version and the error type) and the worker moves on: the `Index` protocol names
  no errors, and every failure has the same answer. The chunks left behind are hidden by
  validity, and removed by the document's next index job or by `kb-server gc`.
- Groups are read when the job is claimed, so the newest groups are indexed. Errors from the
  index or the model pass through and are retried.

### Provenance

Where a version's prepared output and vectors came from: the inputs and the decisions, not the
results (counts and warnings are rebuilt from the files when shown).

```json
{
  "preparation": {"profile": "vn_regulation@4", "profiles_without_structure": [], "settings": {"...": "..."}},
  "index": {"embedder": "huggingface:BAAI/bge-m3@<revision>", "index": "qdrant:kb_chunks"}
}
```

Uses: explaining a version to a reviewer; `kb-server reprepare --stale` (recorded settings
differ from the current ones); finding versions to re-embed after an embedder change. A
version with `preparation` but no `index` isn't searchable with its current preparation.
Who ran what and when is in `indexing_jobs`, not here.

### Deleting (built)

```text
request  DELETE /documents/{id} → 202, no body. One transaction: status = deleting (only if it
         isn't already) and, if it changed, a delete job with version = None.
         Hidden → 404; readable but not manageable → 403; a second DELETE queues nothing.
handler  run_delete_job: 1. remove its chunks from the index
                         2. delete prepared/<document_id>/
worker   3. one transaction: read every version's file id, delete the document row (versions,
            groups and jobs cascade, this job's row included), then delete the stored_files
            rows among those files that no version uses
         4. after the commit: delete those originals from the store
```

- Reads treat a `deleting` document as gone (a tombstone): `GET` is 404, and the duplicate-file
  check skips it, so the same file can be uploaded again at once. Search drops its hits too:
  validity shows only `active` documents (section 5).
- Every other job of a `deleting` document is skipped (W6); the delete job isn't.
- The delete job has no version. Why: it applies to the whole document, so a new version can't
  supersede it and leave the tombstone behind.
- Steps 1–2 are safe to repeat. The database goes before the originals (rule 4): a crash
  between 3 and 4 leaves an orphaned original, never a row pointing at a missing file.
- Nothing is recorded: the job's row goes with the document. So step 4 can't fail the job: an
  `OSError` is logged with the stored file's id, and `kb-server gc` removes the orphan later.
- No job id in the response, unlike other long work. Why: the job's row disappears when it
  finishes, so `GET /jobs/{id}` couldn't tell "finished" from "never existed".
- `DocumentRepository.delete` removes only a document already marked `deleting`. Why: a bug
  can't hard-delete an active document.
- Known window, accepted: if the same file is uploaded again between step 3's commit and
  step 4, the upload finds the original still there and doesn't write it, then step 4 deletes
  it; the new document's prepare job fails with "stored file missing". The window is
  milliseconds. While the delete job waits in the queue, a re-upload is safe: step 3 deletes
  only rows no version uses.
- Open: a delete job that fails for good (the index down for every attempt) leaves the
  tombstone in place, and a second DELETE queues nothing. A retry path (DELETE queues a new
  job when the last one failed, or an operator command) comes with operations (step 8).

### Logging

Python `logging`, to stderr; Docker Compose or systemd collect it. Configured once at startup
(`KB_LOG_LEVEL`). Log lines carry ids, versions, profile names and codes only: never document
text, tokens or emails.

---

## 9. Authentication and authorization (built)

- Passwords: argon2id. Tokens: opaque (`kb_s_` session, `kb_k_` key), SHA-256 stored only,
  shown once, never logged.
- `Authorization: Bearer <token>` → `Principal(user, credential_id, credential_kind,
  acting_for_others)`.
- A front-end key acts for the user named in `KB_ACT_FOR_HEADERS`; the user must exist and be
  active. `Principal.is_admin` is false while acting for someone.

| Role | May |
|---|---|
| reader | search, answer, download — within their groups |
| manager | + documents, batches, relations, retirements — in collections they manage |
| admin | everything; users, groups, keys, collections, reindex |

---

## 10. REST API: `/api/v1`

```text
errors       {"error": {"code": "document_not_found", "message": "…", "details": …}} + HTTP status
lists        ?limit=50&cursor=…  →  {"items": [...], "next_cursor": "…" | null}
long work    202 Accepted + job body + Location: /api/v1/jobs/{id}
```

| Area | Endpoints |
|---|---|
| system (built) | `GET /health`  `GET /ready` |
| auth (built) | `POST /auth/login`  `POST /auth/logout`  `GET /me`  `POST /me/password` |
| keys (built) | `POST /keys`  `GET /keys`  `DELETE /keys/{id}`  admin variants |
| users (built) | `POST /users`  `GET /users`  `GET /users/{id}`  `PATCH /users/{id}`  … |
| groups (built) | `POST /groups`  `GET /groups`  `DELETE /groups/{id}`  members |
| collections (built) | CRUD, `PUT /collections/{id}/groups`, managers |
| documents | `POST /documents` (built)  `GET /documents/{id}` (built)  `GET /documents`  `PATCH /documents/{id}`  `PUT /documents/{id}/file`  `DELETE /documents/{id}` (built)  `GET /documents/{id}/versions`  `GET /documents/{id}/file`  `GET /documents/{id}/review` |
| jobs | `GET /jobs`  `GET /jobs/{id}` |
| search | `POST /search`  `POST /answer` |
| later | batches, relations, retirements, feedback, `/admin/*`, `/openai/v1` |

Error codes so far: 401 `login_failed`, `not_authenticated`; 403 `not_allowed`; 404
`user_not_found`, `group_not_found`, `collection_not_found`, `document_not_found`,
`credential_not_found`; 409 `email_taken`, `group_name_taken`, `group_in_use`,
`collection_slug_taken`, `duplicate_file`; 413 `file_too_large`; 415 `unsupported_file_type`;
422 `invalid_request`; 500 `unexpected_error`.

An error no handler knows becomes `500 unexpected_error` with a fixed message: its own message
could reveal paths or SQL. Starlette re-raises it after answering, so uvicorn logs the
traceback; the handler doesn't log it again.

## 11. OpenAI-compatible: `/openai/v1` (later)

```text
GET  /openai/v1/models              one model per collection + "knowledge-base" (all)
POST /openai/v1/chat/completions    conversation in, answer + sources out; stream supported
```

## 12. CLI (`kb-cli`, later)

HTTP only, config in `~/.config/kb/config.toml`. `kb login | docs | jobs | search | ask | keys
| users | groups | collections`.

## 13. Server commands (`kb-server`)

```text
kb-server create-admin              first admin user (built)
kb-server serve [--reload]          uvicorn (built)
kb-server worker                    run jobs until SIGTERM or SIGINT (built)
kb-server migrate                   alembic upgrade head
kb-server reprepare --stale|--all   queue prepare jobs for out-of-date versions
kb-server gc                        remove originals no row needs, and chunks of versions
                                    other than indexed_version
```

---

## 14. Build order

1. Skeleton: settings, database, wiring, app factory, health, Alembic, tests. **(built)**
2. Users, groups, credentials, login, API keys, `Principal`, `create-admin`. **(built)**
3. Collections. **(built)**
4. Documents and the worker:
   - a. FileStore, document tables, upload, `GET /documents/{id}`. **(built)**
   - b. The worker:
     1. `prepared.py` and `PreparationSettings` **(built)**
     2. the `plain` profile **(built)**
     3. `ProfileChooser`, `ClaimedJob`, `PermanentJobError` **(built)**
     4. the prepare job **(built)**
     5. the index job **(built)**
     6. `Worker`:
        - `run_once`: claim → load → superseded check → handler → record; retries **(built)**
        - `run_forever`: polling, graceful shutdown; the stuck-job sweep at startup **(built)**
     7. settings, `IndexTools`, the `kb-server[worker]` extra, `kb-server worker`, Qdrant in
        Docker Compose, logging setup **(built)**
     8. a real end-to-end run with a real regulation and bge-m3 **(built)**
     9. deleting documents **(built)**
     10. `indexed_version` and the switch to a new version **(built)**
5. Search: `POST /search` with the access filter. Moved ahead of the read side because it is
   the reason the system exists, and it tests the index decisions early.
6. Read side: list with paging, versions, download, review page, `GET /jobs/{id}`.
7. Editing: title (database only) and groups (`sync_attributes`), new versions, outline
   preview, profile override.
8. Operations: `reprepare`, re-index after an embedder change, `gc`, backups.
9. Retrieval evaluation: a labelled question set under `data/`, hit rate and MRR.
10. Answers, the OpenAI-compatible surface, batches, relations and retirements, feedback, CLI.

Project hygiene alongside: CI (ruff, mypy, pytest with Postgres), a pre-commit guard refusing
`data/`, `.docx`, `.pdf`, `.env`, and short decision records.

## 15. Decisions that are easy to undo by accident

| Decision | Why |
|---|---|
| Hidden things are 404, not 403 | A 403 confirms that something exists |
| Repositories report, services raise | One place decides what's an error |
| No `profile` column on `documents` | Chosen automatically; overrides have their own table |
| Prepared output per version, no cache | It retires with its document; nothing stale is reused |
| `save_prepared` deletes before writing | A half-save loads as nothing, never as a mix |
| Handlers never touch the database | They can be tested with a file store only |
| Search filters inside the index | Post-filtering loses results and leaks hidden hits |
| Provenance holds inputs and decisions, not results | Results can be rebuilt; settings can't |
| One worker | No races between jobs; follow-up jobs run in order |
| Model dependencies imported inside `open_job_tools` and `open_search_tools` | A top-level import loads torch in every command |
| `SearchService` lives for the app and reads in short transactions | A request holding a connection through the models can exhaust the pool |
| Models found by reference under one `KB_MODELS_FOLDER` | A folder setting per model drifts from its spec |
| The claim commits before the handler runs | A rolled-back claim is retried forever and never counts its attempts |
| A job with no version is never superseded by a new version | It applies to the whole document, not to one version |
| Every job but the delete job of a `deleting` document is skipped | Its work would be removed anyway; an index job would re-embed it |
| Originals are deleted after the commit | A crash leaves an orphaned file, never a row pointing at nothing |
| Validity shows `indexed_version`, not `current_version` | `current_version` changes at upload: a re-index would hide the document, a failed one for good |
| The index handler touches only its own version; old versions go after the commit | Readers see all of the old version until the flip, then all of the new one |
| The cleanup after an index job never fails the job | The job is recorded; validity hides leftovers, and the next index job removes them |
| A delete job records nothing | Its row is deleted with the document; `mark_done` would fail |
| `DocumentRepository.delete` needs status `deleting` | A bug can't hard-delete an active document |
| Adding an enum value has a no-op downgrade | Narrowing it would fail or lose the rows that use the value |
| `vn_regulation` isn't the default | Most documents aren't regulations; `plain` is the safe fallback |
| Fields the access filter and validity read live in one preset (`Guarded`) | One filter and one validity check apply to every index a statement searches |

## 16. Later

- Framework: outline levels on physical blocks; generic profiles (`headings`, `numbered`);
  profile scoring; extract the physical document once for several profiles.
- Answer tone per user; routing of non-questions; section scoping by heading relevance.
- Unresolved references to documents not yet uploaded.
- Conversations stored server-side; nested collections; automatic related documents.
- An audit trail; deployment; index rebuild tools.
