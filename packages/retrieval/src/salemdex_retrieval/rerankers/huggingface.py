"""A reranker running a sentence-transformers cross-encoder, behind the `huggingface` extra.

The model is used as Hugging Face publishes it, its own score function included: for a
cross-encoder with one output, sentence-transformers applies a sigmoid, so scores run
from 0 to 1.
"""

import os
from collections.abc import Sequence
from typing import Final

from sentence_transformers import CrossEncoder

from salemdex_retrieval._huggingface import Device, load_model, resolve_device
from salemdex_retrieval.checks import check_positive_int
from salemdex_retrieval.errors import ModelLoadError


class HuggingFaceReranker:
    """A `Reranker` running a sentence-transformers cross-encoder in this process.

    The model reads the question and each text together, up to `max_length` tokens
    for the pair, or the model's own limit when it isn't given; the rest is cut,
    without an error.

    Attributes:
        device: Where the model runs, with `"auto"` already decided, for logging.

    Example:
        reranker = HuggingFaceReranker(
            "BAAI/bge-reranker-v2-m3", max_length=1024, device="auto", batch_size=16
        )
        statement = search(question).retrieve(dense).rerank(reranker).take(10)
    """

    def __init__(
        self,
        model: str | os.PathLike[str],
        *,
        device: Device,
        batch_size: int,
        max_length: int | None = None,
    ) -> None:
        """Load the cross-encoder, from a local folder or from the hub.

        Args:
            model: A local folder or a hub id, as sentence-transformers takes it. A
                hub model comes from the Hugging Face cache, or is downloaded into it
                unless `HF_HUB_OFFLINE` is set.
            device: Where the model runs. `"auto"` picks CUDA, then MPS, then the CPU.
            batch_size: How many question-text pairs the model scores at once.
            max_length: The most tokens the model reads of each pair, the question and
                the text together; at most the model's own limit. Each pair costs time
                and memory in proportion to its length, so set it when the model's
                limit is far longer than the texts. Left `None`, the model's own
                limit applies.

        Raises:
            ValueError: `batch_size`, or `max_length` when given, isn't a positive
                `int`, or `device` names a GPU PyTorch can't find, all checked before
                the model loads; or `max_length` is above the model's own limit.
            ModelLoadError: No local folder or hub model has that name, what is there
                isn't a model, or the model gives more than one output per pair (a
                classifier, not a reranker).
        """
        if max_length is not None:
            check_positive_int(max_length, name="max_length")
        check_positive_int(batch_size, name="batch_size")
        self.device: Final = resolve_device(device)
        self._model = load_model(model, model_class=CrossEncoder, device=self.device)
        # A classifier (an NLI model, say) gives a row per pair; refuse it at startup
        # rather than at the first search.
        num_labels = self._model.num_labels
        if num_labels != 1:
            raise ModelLoadError(
                f"model {os.fspath(model)!r} has {num_labels} labels; a reranker needs 1"
            )
        if max_length is not None:
            _limit_length(self._model, max_length=max_length, model=model)
        self._batch_size = batch_size

    def score(self, question: str, texts: Sequence[str]) -> list[float]:
        """Return one score per text in `texts`, in the same order.

        Args:
            question: The question, as asked.
            texts: The passages to judge against it.

        Returns:
            One score per text; higher means a better answer.
        """
        if not texts:
            return []

        pairs = [(question, text) for text in texts]
        # mypy sees CrossEncoder as Any, so nothing here is type-checked; the count and
        # each number are checked where the scores enter (score_texts).
        scores: list[float] = self._model.predict(
            pairs, batch_size=self._batch_size, show_progress_bar=False
        ).tolist()
        return scores


def _limit_length(
    cross_encoder: CrossEncoder, *, max_length: int, model: str | os.PathLike[str]
) -> None:
    """Make `cross_encoder` read at most `max_length` tokens of each pair.

    Set after loading, not passed to `load_model`, so it can be checked against the
    model's own limit, which is `None` for a model that declares none; the embedder,
    loaded by the same function, keeps its own length.

    Raises:
        ValueError: `max_length` is above the model's own limit.
    """
    # Above the limit, a long pair would reach positions the model has no embedding
    # for and fail at search time; refused here, at startup, instead
    model_limit = cross_encoder.max_seq_length
    if model_limit is not None and max_length > model_limit:
        raise ValueError(
            f"max_length {max_length} is above the {model_limit} tokens "
            f"model {os.fspath(model)!r} can read"
        )
    cross_encoder.max_seq_length = max_length
