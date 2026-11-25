"""`HuggingFaceReranker`: scoring with a tiny model built here, and a real model when one
is set. Its settings and loading are tested with the embedder's, in
tests/test_huggingface.py.

The tiny models have random weights from a fixed seed, so their scores mean nothing
but are the same on every run; they show the shape of what comes back and the refusal
of a classifier, with no download. The ranking tests
run only when KB_TEST_RERANKER_MODEL names a model, as a hub id or a local folder, e.g.
`BAAI/bge-reranker-v2-m3`. They use English text, so any model can run them.
"""

import os
import re
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    PreTrainedTokenizerFast,
)

from salemdex_retrieval.errors import ModelLoadError
from salemdex_retrieval.rerankers.huggingface import HuggingFaceReranker
from salemdex_retrieval.rerankers.protocol import Reranker, score_texts

MODEL = os.environ.get("KB_TEST_RERANKER_MODEL")

QUESTION = "How many days of annual leave do I get?"
PASSAGES = [
    "Late payment of an invoice is charged interest of two percent per month.",
    "Annual leave is twelve working days for each full year of work.",
    "Scope of application: this regulation applies to every employee of the company.",
]


# A tiny model

# How many tokens the tiny model can read
TINY_MODEL_LIMIT = 64


@pytest.fixture(scope="module")
def tiny_reranker(tmp_path_factory: pytest.TempPathFactory) -> HuggingFaceReranker:
    """Return a reranker of a tiny one-label cross-encoder, built once for the module."""
    folder = tmp_path_factory.mktemp("tiny-reranker")
    _build_cross_encoder(folder, labels=1)
    # A batch smaller than PASSAGES, so a call spans batches.
    return HuggingFaceReranker(folder, device="cpu", batch_size=2)


def test_it_is_a_reranker(tiny_reranker: HuggingFaceReranker) -> None:
    usable: Reranker = tiny_reranker

    assert usable is tiny_reranker


def test_each_text_gets_one_plain_float_between_zero_and_one(
    tiny_reranker: HuggingFaceReranker,
) -> None:
    scores = tiny_reranker.score(QUESTION, PASSAGES)

    assert len(scores) == len(PASSAGES)
    assert all(type(score) is float and 0.0 <= score <= 1.0 for score in scores)


def test_scores_come_back_in_the_order_of_the_texts(
    tiny_reranker: HuggingFaceReranker,
) -> None:
    together = tiny_reranker.score(QUESTION, PASSAGES)
    one_by_one = [tiny_reranker.score(QUESTION, [text])[0] for text in PASSAGES]

    assert together == pytest.approx(one_by_one, rel=1e-4)


def test_no_texts_give_no_scores(tiny_reranker: HuggingFaceReranker) -> None:
    assert tiny_reranker.score(QUESTION, []) == []


def test_its_scores_pass_the_checks_where_they_enter(
    tiny_reranker: HuggingFaceReranker,
) -> None:
    scores = score_texts(QUESTION, PASSAGES, reranker=tiny_reranker)

    assert len(scores) == len(PASSAGES)


def test_words_past_max_length_are_never_read_and_without_it_the_model_reads_them(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "tiny"
    _build_cross_encoder(folder, labels=1)
    short = HuggingFaceReranker(folder, device="cpu", batch_size=2, max_length=8)
    full = HuggingFaceReranker(folder, device="cpu", batch_size=2)
    # With the question, the pair passes 8 tokens before the texts differ
    start = "annual leave days annual leave days"
    texts = [f"{start} annual", f"{start} interest"]

    cut = short.score("leave", texts)
    read = full.score("leave", texts)

    # Exact comparisons: cut to the same tokens, the scores are the same number; the
    # one at the model's own limit tells them apart, so the model sees the difference
    assert cut[0] == cut[1]
    assert read[0] != read[1]


def test_a_max_length_above_the_model_limit_is_refused_when_it_loads(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "tiny"
    _build_cross_encoder(folder, labels=1)

    message = (
        f"max_length {TINY_MODEL_LIMIT + 1} is above the {TINY_MODEL_LIMIT} tokens "
        f"model {str(folder)!r} can read"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        HuggingFaceReranker(
            folder, max_length=TINY_MODEL_LIMIT + 1, device="cpu", batch_size=2
        )


def test_a_model_with_more_than_one_label_is_refused_when_it_loads(
    tmp_path: Path,
) -> None:
    classifier = tmp_path / "classifier"
    _build_cross_encoder(classifier, labels=3)

    message = f"model {str(classifier)!r} has 3 labels; a reranker needs 1"
    with pytest.raises(ModelLoadError, match=re.escape(message)):
        HuggingFaceReranker(classifier, device="cpu", batch_size=2)


# A real model


@pytest.fixture(scope="module")
def reranker() -> HuggingFaceReranker:
    """Return the reranker of KB_TEST_RERANKER_MODEL, loaded once for the module."""
    if MODEL is None:
        pytest.skip("KB_TEST_RERANKER_MODEL is not set")
    return HuggingFaceReranker(MODEL, device="cpu", batch_size=2)


def test_the_passage_that_answers_the_question_scores_highest(
    reranker: HuggingFaceReranker,
) -> None:
    scores = score_texts(QUESTION, PASSAGES, reranker=reranker)

    assert max(range(len(scores)), key=scores.__getitem__) == 1


def test_a_real_model_gives_one_score_per_text(reranker: HuggingFaceReranker) -> None:
    assert len(reranker.score(QUESTION, PASSAGES)) == len(PASSAGES)


def _build_cross_encoder(folder: Path, *, labels: int) -> None:
    """Save a tiny BERT cross-encoder with random weights and `labels` outputs."""
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "annual", "leave", "days", "interest"]
    tokenizer = Tokenizer(
        models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    # transformers leaves these constructors untyped.
    PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_max_length=TINY_MODEL_LIMIT,
    ).save_pretrained(folder)
    config = BertConfig(
        vocab_size=len(words),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=16,
        max_position_embeddings=TINY_MODEL_LIMIT,
        # BERT's default (0.02) leaves one token's effect on the score so small that
        # texts differing by a word tied in about 1 model in 5, and 1.0 ties too (in
        # about 1 in 12); from 0.2 to 0.5, none of 300 seeds tied
        initializer_range=0.3,
    )
    config.num_labels = labels
    # A fixed seed, so every run builds the same model; forked, so the seed doesn't
    # leak into other tests
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = BertForSequenceClassification(config)  # type: ignore[no-untyped-call]
    model.save_pretrained(folder)
