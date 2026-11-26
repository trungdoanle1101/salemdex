"""Embedders running sentence-transformers models, behind the `huggingface` extra.

`HuggingFaceDenseEmbedder` runs a dense model (`SentenceTransformer`) and
`HuggingFaceSparseEmbedder` a learned sparse one (`SparseEncoder`, e.g. SPLADE). Each
model is used as Hugging Face publishes it: nothing about it is kept here. Its query and
document prompts, if it declares any, come from its own configuration.
"""

import json
import os
from collections.abc import Mapping, Sequence
from typing import Final

from sentence_transformers import SentenceTransformer, SparseEncoder
from sentence_transformers.util import load_file_path
from torch import Tensor

from salemdex_retrieval._huggingface import Device, load_model, resolve_device
from salemdex_retrieval.checks import check_positive_int
from salemdex_retrieval.errors import ModelLoadError

_CONFIG_SENTENCE_TRANSFORMERS_JSON = "config_sentence_transformers.json"


class HuggingFaceDenseEmbedder:
    """A `DenseEmbedder` running a sentence-transformers model in this process.

    Queries go through the model's `encode_query` and passages through its
    `encode_document`, so each gets the prompt the model declares for it, or none.

    Attributes:
        device: Where the model runs, with `"auto"` already decided, for logging.

    Example:
        embedder = HuggingFaceDenseEmbedder("BAAI/bge-m3", device="auto", batch_size=16)
        binding = DenseBinding(vector=KbChunks.dense, embedder=embedder)
    """

    def __init__(
        self, model: str | os.PathLike[str], *, device: Device, batch_size: int
    ) -> None:
        """Load the model, from a local folder or from the hub.

        Args:
            model: A local folder or a hub id, as sentence-transformers takes it. A
                hub model comes from the Hugging Face cache, or is downloaded into it
                unless `HF_HUB_OFFLINE` is set.
            device: Where the model runs. `"auto"` picks CUDA, then MPS, then the CPU.
            batch_size: How many texts the model encodes at once.

        Raises:
            ValueError: `batch_size` isn't a positive `int`, or `device` names a GPU
                PyTorch can't find. Both are checked before the model loads.
            ModelLoadError: No local folder or hub model has that name, or what is
                there isn't a model. The error from Hugging Face is chained.
        """
        check_positive_int(batch_size, name="batch_size")
        self.device: Final = resolve_device(device)
        self._model = load_model(
            model, model_class=SentenceTransformer, device=self.device
        )
        self._batch_size = batch_size

    def embed_queries(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one row of numbers per question in `texts`, in the same order.

        Args:
            texts: The questions, as asked.

        Returns:
            One row per text, each as long as the model's output.
        """
        if not texts:
            return []

        embeddings = self._model.encode_query(
            list(texts), batch_size=self._batch_size, show_progress_bar=False
        )
        # The shape and the numbers are checked where the rows enter (Dense.query).
        rows: list[list[float]] = embeddings.tolist()
        return rows

    def embed_passages(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one row of numbers per passage in `texts`, in the same order.

        Args:
            texts: The passages, as they are to be found.

        Returns:
            One row per text, each as long as the model's output.
        """
        if not texts:
            return []

        embeddings = self._model.encode_document(
            list(texts), batch_size=self._batch_size, show_progress_bar=False
        )
        # The shape and the numbers are checked where the rows enter (Dense.value).
        rows: list[list[float]] = embeddings.tolist()
        return rows


class HuggingFaceSparseEmbedder:
    """A `SparseEmbedder` running a sentence-transformers `SparseEncoder` here.

    Ids are the model's vocabulary token ids. Its weights are learned, IDF included, so
    bind it to a `Sparse(modifier="none")` vector: `"idf"` would scale them twice.

    Queries go through the model's `encode_query` and passages through its
    `encode_document`, so each gets the prompt the model declares for it, and an
    inference-free model runs its network on passages only, as it was trained to.

    Attributes:
        device: Where the model runs, with `"auto"` already decided, for logging.

    Example:
        embedder = HuggingFaceSparseEmbedder(
            "naver/splade-v3", device="auto", batch_size=16
        )
        binding = SparseBinding(vector=KbChunks.lexical, embedder=embedder)
    """

    def __init__(
        self, model: str | os.PathLike[str], *, device: Device, batch_size: int
    ) -> None:
        """Load the model, from a local folder or from the hub.

        Args:
            model: A local folder or a hub id, as sentence-transformers takes it. A
                hub model comes from the Hugging Face cache, or is downloaded into it
                unless `HF_HUB_OFFLINE` is set.
            device: Where the model runs. `"auto"` picks CUDA, then MPS, then the CPU.
            batch_size: How many texts the model encodes at once.

        Raises:
            ValueError: `batch_size` isn't a positive `int`, or `device` names a GPU
                PyTorch can't find. Both are checked before the model loads.
            ModelLoadError: No local folder or hub model has that name, or what is
                there isn't a model (the error from Hugging Face is chained), or it
                isn't a `SparseEncoder`: a dense model or a bare transformers
                checkpoint, which sentence-transformers would turn into a sparse
                encoder with weights never trained for it.
        """
        check_positive_int(batch_size, name="batch_size")
        self.device: Final = resolve_device(device)
        self._model = load_model(model, model_class=SparseEncoder, device=self.device)
        # After loading, so a name that isn't a model gets load_model's error
        _check_is_sparse_encoder(model)
        self._batch_size = batch_size

    def embed_queries(self, texts: Sequence[str]) -> Sequence[Mapping[int, float]]:
        """Return one mapping of token id to weight per question in `texts`, in order.

        Args:
            texts: The questions, as asked.

        Returns:
            One mapping per text, holding only the ids the model weighs above 0;
            empty when it weighs none.
        """
        # An empty batch comes back as a tensor of another shape: (0,), not (0, vocab)
        if not texts:
            return []

        embeddings = self._model.encode_query(
            list(texts), batch_size=self._batch_size, show_progress_bar=False
        )
        # The ids and the weights are checked where they enter (Sparse.query).
        return _weights_by_row(embeddings, row_count=len(texts))

    def embed_passages(self, texts: Sequence[str]) -> Sequence[Mapping[int, float]]:
        """Return one mapping of token id to weight per passage in `texts`, in order.

        Args:
            texts: The passages: each chunk's `embedded_text`.

        Returns:
            One mapping per text, holding only the ids the model weighs above 0;
            empty when it weighs none.
        """
        # An empty batch comes back as a tensor of another shape: (0,), not (0, vocab)
        if not texts:
            return []

        embeddings = self._model.encode_document(
            list(texts), batch_size=self._batch_size, show_progress_bar=False
        )
        # The ids and the weights are checked where they enter (Sparse.value).
        return _weights_by_row(embeddings, row_count=len(texts))


def _check_is_sparse_encoder(model: str | os.PathLike[str]) -> None:
    """Check `model` was saved as a `SparseEncoder`, as sentence-transformers reads it.

    Raises:
        ModelLoadError: Its `config_sentence_transformers.json` is missing (a bare
            transformers checkpoint) or names another type, `SentenceTransformer`
            when it names none, as sentence-transformers reads it.
    """
    # The lookup sentence-transformers' own loading uses: a folder, the cache or the hub
    path = load_file_path(os.fspath(model), _CONFIG_SENTENCE_TRANSFORMERS_JSON)
    if path is None:
        raise ModelLoadError(
            f"{os.fspath(model)!r} has no {_CONFIG_SENTENCE_TRANSFORMERS_JSON}: "
            "not a SparseEncoder"
        )
    with open(path, encoding="utf-8") as file:
        model_type = json.load(file).get("model_type", "SentenceTransformer")
    if model_type != "SparseEncoder":
        raise ModelLoadError(
            f"{os.fspath(model)!r} is a {model_type} model, not a SparseEncoder"
        )


def _weights_by_row(embeddings: Tensor, *, row_count: int) -> list[dict[int, float]]:
    """Return each of `row_count` rows of a sparse batch as plain token id to weight."""
    # Coalesced, as indices() needs: duplicate entries summed, entries sorted
    coalesced = embeddings.coalesce()
    # tolist() is typed Any: annotated here, so mypy checks everything below
    indices: list[list[int]] = coalesced.indices().tolist()
    rows, ids = indices
    weights: list[float] = coalesced.values().tolist()
    # One per row up front: a row the model weighs nothing has no entries at all
    mappings: list[dict[int, float]] = [{} for _ in range(row_count)]
    for row, sparse_id, weight in zip(rows, ids, weights, strict=True):
        mappings[row][sparse_id] = weight
    return mappings
