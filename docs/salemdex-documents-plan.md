# salemdex-documents: plan

Status: intent, first suggestions and a brainstorm of key approaches (2026-11-19), not yet a
design. Nothing here is decided except the goal, the intent and the package name; the
details come once the direction is settled.

## The goal

The product's goal, the questions people ask, the three levers (facts, retrieval,
reasoning) and evaluation are in `salemdex-product.md`. This plan covers the facts lever:
the document side.

## Intent

Rework the framework's document side (adapters, cleaning, profiles, outlines, chunking,
review) into a new package, `salemdex-documents`, next to `salemdex-retrieval`, so that it
serves the goal above better than the current framework does, and better than general
document frameworks do for structured documents:

- **Serves answers:** parsing gives search passages it can rank and people can check, not
  just text.
- **Easy to use:** an API as plain as `salemdex-retrieval`'s, with useful utilities around
  it.
- **Not shaped by `.docx`:** the core models are reconsidered so they suit every format,
  with `.docx` as one adapter among several.

## Why rework

The current framework has strengths worth carrying over:
- models that check themselves when they're built;
- provenance back to the source;
- the profile language (marker templates, `inside`, title rules, numbering-aware choices);
- table handling;
- the review page;
- a large test suite.

A review on 2026-11-18 found problems that a port alone wouldn't fix:

- **Text is lost silently or documents are refused.** The `.docx` reader drops tracked
  insertions, inline content controls, simple fields and smart tags. It refuses a
  document with a content control or bookmark at body level, or a nested table.
- **Passages exceed the size limit.** A paragraph longer than the budget becomes one
  passage; `Chunk.text_span` exists for this but is never set.
- **The models are shaped by `.docx`.** Lists use Word's numbering vocabulary and ids, and
  formatting is a Word style name. The source reference is an untyped XPath, and
  `Region` mixes page furniture with notes. Table cells are strings, with list markers
  pasted into them. A paragraph is split into blocks around images, and the set of
  formats is a closed enum.
- **The models also hold fields nothing fills or reads:** `char_spans`, `page`,
  `SpanStyle`, and `region` other than `BODY`. Formatting is collected but never used by
  the outline.
- **The API and its errors:**
  - the entry point is a file path, and the adapter is chosen by suffix;
  - one `Preparer` runs every stage per profile, so the chooser parses and chunks again
    for each candidate;
  - errors are plain `ValueError`s, and only the review page reports warnings.

## Suggestions

### The package at a glance

**`salemdex-documents` turns a document into passages and facts that search can find and a
person can check.** It is the facts lever (`salemdex-product.md`, section 3).

```python
document = read(data, media_type=DOCX)
outline = choose_outline(document, profiles=[vn_regulation, plain])
passages = split_passages(outline, max_words=250)
facts = find_facts(document, outline)       # references, definitions, front matter…
```

- **Fidelity it can prove:** every character shown is read or reported as unread, images
  included. The source text is never changed; normalisation is a view.
- **Passages hold a whole idea and know where they sit:** sized in words, at sentence
  boundaries, never over the limit; the source's own words, with what's needed to find
  and read them beside it (headings path, lead-in, table headers); each points at its
  outline node, so a section or the whole document can be fetched from it.
- **Facts in two kinds:** structural, derived by rules (references resolved to nodes,
  definitions linked to their uses, the abbreviations list, front matter), which may be
  shown as evidence; semantic, extracted by a model through a port, used for finding only.
- **Diagnostics with codes and locations:** the library reports; the app decides what
  stops the work.
- **No dependency on `salemdex-retrieval`:** the app turns passages into chunks.

### Where to compete

Other frameworks compete mostly on recovering layout (PDF, scans, OCR, ML models), and give
generic labels. For structured documents, the answers depend on what they don't do:
structure from profiles, guaranteed fidelity, addressing precise enough to cite, and
review. Compete there, and use the layout tools as front ends: a PDF can come in through
an adapter over an existing parser.

**How other frameworks use ML.** Mainly because a PDF or a scan doesn't declare its
structure: it's characters at coordinates, or pixels. Three families:

| Family | What it does | Strong at | Weak at |
|---|---|---|---|
| a pipeline of models (Docling, MinerU, Azure Document Intelligence) | OCR, then a layout model that boxes and labels regions (title, section header, text, table, figure…), reading order, a table-structure model, then assembly | PDFs, scans, tables in PDFs | heading levels (a generic "section header"), domain conventions |
| one vision-language model per page (Nougat, olmOCR, SmolDocling, general VLMs) | reads a page image, writes Markdown | messy layouts, formulas, many languages | can skip or rephrase text; loses structure across pages; often no source location; needs a GPU or an API |
| an LLM after parsing (structured extraction, contextual retrieval, generated questions, knowledge graphs) | extracts fields into a schema, adds context or entries for search | facts, recall | fidelity, cost per document |

For `.docx`, the main frameworks use rules, not models. Docling 2.128 (checked in its
source) finds a heading only through a heading style or Word's `outlineLvl`, a list item
only through Word's numbering or a list style; bold is recorded, never used; nothing reads
the text. So typed markers ("Chương I", "1.", "a)", `-`, `+`) stay ordinary paragraphs.
Converting such a `.docx` to PDF gives Docling a better outline, because its layout model
sees what a reader sees.

What that means here: ML as a front end for PDFs and scans; as an optional source of hints
(an outline proposed, a profile drafted); and for semantic facts behind a port. The core
(structure, fidelity, addresses) stays deterministic.

### Directions for the model

- Keep apart **content**, **hints** and **location**. The outline reads content and hints,
  never the format. Hints come in three kinds:

  | Kind | `.docx` | PDF (through a parser) |
  |---|---|---|
  | declared: styles, heading levels, list numbering | read from the file | rarely present |
  | appearance: bold, size, caps, alignment, indentation | resolved from styles and direct formatting, not rendered | from the layout parser |
  | text: markers, keywords, glyphs | the text | the text |

  Appearance is the hint every format can give. Why it matters: authors who type their
  markers put the structure in what the page looks like, not in what the file declares.
  It stays evidence, never proof (bold is also emphasis).
- **A marker is one concept, with an origin:** typed (a range at the start of the block's
  text, marked, never removed) or generated by the format's numbering (rendered, stored
  beside the text). The outline reads it whatever the origin. A generated marker is
  content a person sees, so it counts for fidelity and citations. A declared level that
  shows nothing still gives nesting. In a PDF every marker is typed, so the model covers
  both formats without special cases.
- Content is a tree of blocks (paragraph, list item, table whose cells hold blocks,
  figure, note), in three parts: body, furniture and notes.
- Inline information as ranges beside one plain text per block (emphasis, links, note
  references, cross-references).
- The outline is an interpretation that points at blocks rather than copying their text,
  so several candidate outlines over one document are cheap.
- Locations come in a few kinds (flow, paged, line), not one per format; a format is a
  media type, not an enum.

### Structure without depending on the right profile

**The problem:** in the first version a profile is required. With the right one, typed
markers, inconsistent styles and `-`/`+` nesting come out right; that's why it exists, and
its tests keep those cases. Without it the outline falls off a cliff to `plain`; a profile
that almost fits gives a wrong outline silently; and the choice is first match. Frameworks
without profiles degrade gracefully: they lose detail, never everything. The aim is a
slope: no profile is good, the right profile is better, a wrong one no worse than none.

Profiles stay: when structure exists only as typed text, recovering it takes knowledge of
the kind of document ("1." may be a khoản, a list item or an ordinary paragraph). What
changes is the cost of a missing or wrong one.

- **Two kinds of convention:**
  - *self-describing*, which a rule can apply without a profile: decimal numbering gives
    its own depth; sequences counting up tie items of one level together; consecutive
    lines with the same glyph form one list, and within a run the glyph seen first is the
    outer level (covers `-` then `+` without hard-coding it); keyword markers, given a list
    of keywords;
  - *declared*, which only a profile knows: the keywords and their order, which bare
    family is which level, names and addresses, title rules.

  A profile becomes a composition of shared conventions plus its own knowledge and names.
- **Profiles in layers,** general to specific: a generic layer (styles, keyword headings),
  a Vietnamese legal layer (Phần, Chương, Mục, Điều), then `vn_regulation` (khoản, điểm). A
  document no specific profile fits still gets the general layers.
- **Tolerant profiles:** an anomaly (a repeated number, a level deeper than the profile
  knows, a broken sequence) is kept as an unnamed sub-level and reported, rather than
  distorting the outline.
- **Choose by score, not first match:** coverage, sequences in order, sensible depth, few
  diagnostics. The fallback is always a candidate.
- **A deterministic fallback:** declared hints and Word's numbering (what Docling uses),
  plus appearance, plus the self-describing conventions. A heading style is a deliberate
  act, so the fallback errs by missing structure rather than inventing it. One real error
  remains: partial styling gives wrong levels (a styled Điều above an unstyled Chương). It
  says when it's out of its depth: paragraphs that look like markers but that no rule
  claimed become a diagnostic ("this document has typed structure; a profile is needed").
- **Models, optional and checked:**
  - drafting a profile once per kind of document, reviewed by a person, then run
    deterministically: the model's judgement is spent once and checked once;
  - proposing an outline for one document no profile fits, through a port, at a cost per
    document. Because the outline points at blocks, a model can structure the text
    wrongly but never change it, and rules check it (sequences, coverage).

### Directions for the API

- Each stage is a plain function (read, find the outline, split into passages), taking bytes
  and a media type rather than a path.
- Utilities around them:
  - addressing and citations;
  - export to Markdown or HTML;
  - the review page;
  - coverage and diagnostics;
  - tools for writing profiles;
  - version diffs.
- No bridge to `salemdex-retrieval` inside the package: neither imports the other. The
  app converts passages into its own schema's chunks (the retrieval design puts that
  conversion in app code, since the app declares the schema).

### A possible order

0. Evaluation on the labelled question set, and the questions sorted by kind
   (`salemdex-product.md`, section 5). The kinds people bring most may reorder the steps
   below.
1. The model, with readers for `.docx` (with a fidelity test), `.txt` and Markdown. A
   second structured format checks that the model really is independent of format.
2. The outline: shared conventions, the profiles ported as layers, scoring, and the
   deterministic fallback.
3. Splitting into passages: words, at sentence boundaries, never over the limit.
4. Utilities: addressing, Markdown export, review, diagnostics.
5. Later: PDF through an existing parser, version diffs, cross-references, document facts.

## Open questions

- **Which questions people bring:** the kinds, and how often each comes up. Answered by
  sorting the question set; it settles the order of work.
- **The edge:** structure, fidelity and citations for structured documents, with layout
  recovery left to other tools?
- **Inline information:** ranges beside the text, or a nested inline tree? Leaning to
  ranges: the text stays the one thing checked, cited and highlighted.
- **The text a passage shows:** with generated markers ("a) …")? Leaning yes, since it's
  what the source shows, with generated ranges marked so quote matching knows.
- **Profile layers:** how layers compose, and what the scoring weighs.
- **PDF:** an adapter over an existing parser, or our own reader?
- **Which source parts count:** footnotes, headers and footers, text boxes.
- **Unknown content:** a warning, or a refusal?
- **Moving over:** how the old prepared output retires. Its seam with the server is the
  prepared-chunks-to-salemdex-`Chunk` conversion (retrieval design, "Pick up here",
  step 1).
- **In the meantime:** fix the text-loss bugs in the old `.docx` reader before the real
  end-to-end run (backend build step 4b.8), since the worker uses it today. Among them:
  numbering a paragraph inherits from its style isn't read, so auto-numbered headings
  linked to a multilevel list lose their markers.
