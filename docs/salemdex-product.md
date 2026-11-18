# Salemdex: the product

Status: direction, from discussions on 2026-11-18 and 19. It holds what concerns the whole
product; each package's design goes in its own document:

- `salemdex-retrieval`: `salemdex-retrieval-plan.md`
- `salemdex-documents`: `salemdex-documents-plan.md`
- the server (`kb_server`): `backend-design.md`

The repository is public: examples here are generic, never taken from the restricted
documents or the question sets.

---

## 1. The goal

**A person brings a question, or a problem to solve, and gets an answer they can act on.**

That is the only end. Fidelity, structure, citations, retrieval techniques and models are
means to it. A feature earns its place by the questions it helps answer, and is judged by
whether answers get better.

From the end down, each layer serves the one above it:

```text
the person   gets their question answered, or their problem solved
the answer   is correct, complete, current, and can be checked before acting on it
search       finds the passages that answer it and ranks them first, showing only
             what the person may see and what is still current
documents    give passages that each hold a whole idea, say where they sit, and
             carry the source's own words, so they can be shown and checked
```

Why each means matters, traced to the end:

- **Fidelity:** missing text is a missing answer; changed text is a wrong one.
- **Structure:** a passage cut from its Điều and khoản is ambiguous.
- **Citations and source locations:** a person acting on a regulation must be able to check
  it first. Addresses ("Điều 5, khoản 2, điểm a") matter most for citing an answer; people
  rarely ask by address, they describe their situation.
- **Document facts** (issuer, dates, what replaces what): whether a regulation is in force
  decides whether an answer is right.
- **Tables, lists, notes kept as they are:** an answer inside a flattened table can't be
  found or read.
- **Diagnostics and review:** a person sees where the document side failed before a wrong
  answer does.

## 2. The questions people ask

A question isn't one type: it has a value on each of six axes, and each axis tests a
different layer.

| Axis | What it asks | Layer it tests |
|---|---|---|
| Purpose | why they ask | answer |
| Answer shape | what form the answer takes | answer, documents |
| Evidence shape | where the evidence sits, and how | documents, search |
| Wording gap | how the question's words meet the text | search |
| Time | which moment the answer is for | data model |
| Reasoning | what must happen after the evidence is found | answer, agents |

- **Purpose:** know a fact; may I / must I (must, may, must not, exempt, with conditions);
  does this apply to me; what are the steps and whom do I submit to; who is responsible;
  what does a term mean, or why a rule exists (often not in the text); find a document,
  article or form; compare or choose; check my own content for compliance; what changed;
  produce a checklist or a summary; verify a claim (the premise may be false).
- **Answer shape:** yes/no with conditions; a value with its unit; an entity; a closed list
  (a partial list looks complete); a definition; an ordered procedure; conditions; a
  reference; a comparison; a count; "the text doesn't say"; "which one do you mean?".
- **Evidence shape:**

  | Shape | The risk |
  |---|---|
  | one passage | low |
  | an item under its lead-in ("Các trường hợp sau **không**…") | the item without its lead-in reverses the meaning |
  | a rule here, its exception elsewhere ("trừ trường hợp quy định tại khoản 5") | the rule is found without the exception: a wrong "yes" |
  | a term defined in another article | read with the wrong meaning |
  | a cross-reference ("theo quy định tại Điều 5") | the cited text isn't fetched |
  | a table cell | meaningless without its row and column headers |
  | several articles, or several documents (an amendment, a guidance) | incomplete or contradictory |
  | front matter, signature block, appendix, form | often not chunked, or a separate file |
  | an image or diagram | the text is lost silently |
  | outside the text (versions, owner, upload date) | in the database, not the index |
  | absent: truly, a near miss, or only in a document the person can't see | all three must look the same |
  | silent: nothing forbids it | "not addressed" isn't "allowed" |

  Real regulations break the assumed hierarchy: decimal numbering inside a khoản, a khoản
  number used twice, numbered items inside a điểm. Addresses aren't always unique.
- **Wording gap:** the same words as the text; a paraphrase; a situation described in the
  person's own terms; no diacritics; abbreviations; English mixed with Vietnamese; numbers
  written another way ("07", "7", "bảy", "một tuần"); by number or address; old names;
  very short or very long; several questions in one; a false premise; ambiguous; a
  follow-up that depends on earlier turns.
- **Time:** now; at a past date (audits, incidents); issued but not yet in force; what
  changed or what replaced what; transitional cases; deadlines counted from an event
  ("05 ngày làm việc kể từ…").
- **Reasoning:** read off; match a situation against conditions; assemble steps from
  several articles; enumerate exhaustively ("every obligation of unit X"); count; date
  arithmetic in working days; resolve conflicts (higher rank, newer, specific over
  general); show absence; compare.

## 3. The three levers

Answers improve through three levers, each improved locally:

1. **Facts:** what is extracted from documents, and how it's represented.
2. **Retrieval:** the workflow that brings evidence into the context, in one pass or several.
3. **Reasoning:** the model that reads the evidence and answers.

**They form a chain, and each caps the next.** A model can't reason over evidence search
didn't return; search can't return a fact the documents never represented. So for each
wrong answer, ask in order:

1. Is the needed information in some passage, whole and readable on its own? If not: facts.
2. Did that passage reach the model's context? If not: retrieval.
3. Was the answer wrong with the evidence in front of it? Then: reasoning.

**When several levers could fix a problem, prefer the earliest that fixes it
deterministically.** Why: preparation runs once per document, can be reviewed, and helps
every question and every model; retrieval runs per question but can be measured; reasoning
runs per question, costs most, is hardest to audit and can invent. And models improve with
each release without our work: reasoning is the lever bought, facts and retrieval the ones
built.

| Problem | Primary lever | Why not the others |
|---|---|---|
| item cut from its lead-in | facts: the passage carries its lead-in | the model never sees the lead-in |
| exception in another article | facts: references resolved to nodes; retrieval follows them | the model can't apply an exception it never saw |
| defined terms | facts: uses linked to their definition | same |
| tables | facts: each cell with its headers | same |
| effective dates, issuer, replacements | facts: document facts in the database | not in any passage |
| point in time, not yet in force | facts (dates, versions kept), then retrieval (validity by date) | the model can't know what isn't stored |
| conflicts between documents | facts (relations, rank) and reasoning to apply them | the model alone guesses the hierarchy |
| paraphrase, situational | retrieval: hybrid search, reranker, query rewriting | facts help through generated questions |
| no diacritics, abbreviations | retrieval (query normalisation) using a fact (the document's own abbreviations list) | the model can rewrite, slower and less reliably |
| several places | retrieval: depth, several queries, several passes | the model can't combine what it didn't get |
| exhaustive | facts (structure, who an obligation falls on) and a retrieval path that scans rather than ranks | top-k is incomplete by design |
| multi-hop references | facts (resolved references), else multi-pass retrieval | facts are cheaper per question |
| follow-ups | reasoning inside retrieval: rewrite with the history | nothing else knows the conversation |
| a situation against conditions | reasoning | an inference, not a lookup |
| abstaining, false premises, ambiguity | reasoning, helped by retrieval scores | search always finds something |
| working-day arithmetic | a tool the model calls | model size doesn't fix arithmetic |
| compliance checks | reasoning (an agent) over multi-pass retrieval | several questions in one |

### Two kinds of facts

1. **Structural facts, derived by rules:** outline, lead-ins, references, definitions, table
   headers, front matter. Reliable and reviewable; they may be shown as evidence.
2. **Semantic facts, extracted by a model:** obligations by who they fall on, hypothetical
   questions, summaries, image descriptions. They help finding, but aren't the source's
   words.

**Extracted facts lead to passages; only passages are evidence.** The retrieval design
already has the place for this: generated entries, searched as a resolving source that
returns the real chunks. So an extraction error can cost recall, never a wrong citation.

### Long context blurs retrieval and reasoning

A regulation of a few dozen pages fits a modern model's context. For a question inside one
document, finding the document and letting the model read it whole, or the relevant
chapters, can beat passage retrieval: several places, exceptions and exhaustive questions
mostly stop being problems. Retrieval's job then splits: choosing documents or sections,
where search is essential, and choosing passages within them, which a long-context model
can partly take over.

## 4. Who chooses what

Salemdex is a framework: the budget and the models are the developer's choice. So each lever
is built as well as it can be and made configurable, while the guarantees stay fixed
whatever is chosen. It extends the retrieval design's fixed frame and pluggable stages to
the whole product.

| Lever | Fixed: the guarantees | Configurable: quality and cost | Default |
|---|---|---|---|
| facts | structural facts derived by rules, always; only the source's words shown as evidence | model-extracted facts through a port, stored as generated entries | rules only |
| retrieval | the access filter on every source; the validity gates | the statement: sources, fusion, rerank, one pass or several; what enters the context: passages, sections or whole documents | dense and rerank, passages |
| reasoning | the answer cites only passages it received; abstaining is always possible; claims checked against the evidence | the model, context size, tools, the agent loop | one model, one pass |

What it asks of each package:

- **The context unit is configurable, so documents address sections.** A small local model
  gets short passages; a long-context model gets whole chapters. Both work only if every
  passage knows its outline node, and a section or a document can be fetched from a
  passage: outline nodes are addressable, passages point at them (in retrieval, an
  outline-aware `expand`).
- **Model-extracted facts sit behind a port,** like embedders: the framework defines the
  shape, the developer plugs in any model, and the output becomes generated entries.
- **Evaluation lives in salemdex,** not in an app: a developer choosing a configuration
  needs to compare configurations on their own questions (section 5). The app supplies the
  questions.

**Few settings, each earned.** Every setting multiplies the combinations to test, and an
unmeasured default is a guess. Each lever starts with one good default; a setting is added
when evaluation shows the choice changes results for some kind of question. "The best
option" is measured on the developer's own questions, not asserted by us.

## 5. Evaluation

Improvements are made locally, one lever at a time, so evaluation reports each stage per
question, and the failures counted by lever say where the next improvement pays:

| Stage | Measure | How |
|---|---|---|
| facts | is the gold quote whole inside one passage? | text matching, no model |
| retrieval | did that passage reach the context, at what rank? every one, for several places? | hit rate, MRR, recall of all answers; the candidates' trail |
| reasoning | right with the actual context? right with the gold passages (an oracle)? | a judge, or labelled answers |
| abstention | for unanswerable questions, does it decline? | the best score below a threshold, or the answer says so |

Wrong with the gold passages: reasoning. Right with them, wrong without: retrieval or facts.
Re-check neighbouring levers now and then: they interact (a better reranker makes deeper
retrieval worth it; bigger chunks help the model and hurt precision).

- **Grade by quote, not by article.** A hit is a passage containing the gold quote. An
  article-level label counts any chunk of a long article, and doesn't survive re-chunking or
  a new outline; a quote does. Matching tolerates normalisation (NFC, whitespace), and each
  quote is checked once against the document.
- **Tag each question on two axes at least:** wording gap and evidence shape (section 2), so
  a change shows which kinds it helped.
- **Evaluation exists before the documents rework starts.** Without it there's no telling
  whether the rework helps people or only looks better.
- **The current question set** is AI-generated, over one document, tagged by wording
  (`lookup`, `paraphrase`, `situational`), `several_places` and `unanswerable`, labelled by
  article. It still needs:
  - quote-based grading;
  - wording variants of existing questions (no diacritics, abbreviations, typos), which
    reuse their labels;
  - new kinds: lead-ins, exceptions, defined terms, tables, exhaustive, false premises,
    ambiguous, follow-up pairs;
  - questions across several documents (more documents), and about past dates (two
    versions of one document);
  - over time, real people's questions in place of generated ones.

## 6. Open questions

- **Who asks:** every staff member, specialists, or auditors and compliance too?
  Situational questions dominate for staff; exhaustive and past-date questions for auditors.
- **The corpus:** internal documents only, or the laws and circulars they implement too? If
  the latter, rank and conflicts between documents become central.
- **Past dates:** must an answer say what applied on a given date? That cuts into search's
  "current" guarantee (retrieval design, section 2), so it's decided deliberately, with
  effective dates and retirements.
- **How far "solve" goes:** answers with citations, or also checklists, obligations by unit,
  and checks of a person's own design against the rules.
- **Restricted documents and models:** a deployment that can't send documents to a hosted
  model relies on local models, so facts and retrieval carry more of the weight there. The
  framework stays configurable either way (section 4).
