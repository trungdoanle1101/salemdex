# salemdex-documents: plan

Status: intent and first suggestions, not yet a design. Nothing here is decided except the
intent and the package name; the details come once the direction is settled.

## Intent

Rework the framework's document side (adapters, cleaning, profiles, outlines, chunking,
review) into a new package, `salemdex-documents`, next to `salemdex-retrieval`. The goal is
a project that holds its own against modern document frameworks:

- **Powerful:** parsing gives a knowledge base real value, not just text.
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

### What "value" means

Answers that can be cited, checked and trusted. A parser serves that with:

- **fidelity it can prove:** every character is read, or reported as unread;
- **structure a reader names:** "Điều 5, khoản 2, điểm a";
- **provenance down to the source location,** for highlighting the passage in the
  original;
- **meaning kept, not flattened:** tables, lists, footnotes attached to their references,
  links, and cross-references resolved to their nodes;
- **facts about the document:** number, issuer, dates, from the front matter;
- **changes between versions,** node by node;
- **honesty about uncertainty:** diagnostics with codes and locations, and review.

### Where to compete

Other frameworks compete mostly on recovering layout (PDF, scans, OCR, ML models), and give
generic labels. Compete instead on what they don't do: structure from profiles, guaranteed
fidelity, addressing precise enough to cite, and review. Use the layout tools as front ends:
a PDF can come in through an adapter over an existing parser.

### Directions for the model

- Keep apart **content**, **hints** (what the format declared: styles, heading levels,
  numbering, emphasis) and **location**. The outline reads content and hints, never the
  format.
- Content is a tree of blocks (paragraph, list item, table whose cells hold blocks,
  figure, note), in three parts: body, furniture and notes.
- Inline information as ranges beside one plain text per block (emphasis, links, note
  references, cross-references).
- The outline is an interpretation that points at blocks rather than copying their text,
  so several candidate outlines over one document are cheap.
- Locations come in a few kinds (flow, paged, line), not one per format; a format is a
  media type, not an enum.

### Directions for the API

- Each stage is a plain function (read, find the outline, split into passages), taking bytes
  and a media type rather than a path.
- Utilities around them:
  - addressing and citations;
  - export to Markdown or HTML;
  - the review page;
  - coverage and diagnostics;
  - tools for writing profiles;
  - version diffs;
  - a bridge to `salemdex-retrieval` chunks.

### A possible order

1. The model, with readers for `.docx` (with a fidelity test), `.txt` and Markdown. A
   second structured format checks that the model really is independent of format.
2. The outline: the profiles ported, using heading hints.
3. Splitting into passages: words, at sentence boundaries, never over the limit.
4. Utilities: addressing, Markdown export, review, diagnostics.
5. Later: PDF through an existing parser, version diffs, cross-references, document facts.

## Open questions

- **The edge:** structure, fidelity and citations for structured documents, with layout
  recovery left to other tools?
- **Inline information:** ranges beside the text, or a nested inline tree?
- **PDF:** an adapter over an existing parser, or our own reader?
- **Which source parts count:** footnotes, headers and footers, text boxes.
- **Unknown content:** a warning, or a refusal?
- **Moving over:** how the old prepared output retires. Its seam with the server is the
  prepared-chunks-to-salemdex-`Chunk` conversion (retrieval design, "Pick up here",
  step 1).
- **In the meantime:** fix the text-loss bugs in the old `.docx` reader before the real
  end-to-end run (backend build step 4b.8), since the worker uses it today.
