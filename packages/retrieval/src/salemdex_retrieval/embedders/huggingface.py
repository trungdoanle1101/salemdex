"""A dense embedder running a sentence-transformers model, behind the `huggingface` extra.

The model is used as Hugging Face publishes it: nothing about it is kept here. Its query
and document prompts, if it declares any, come from its own configuration.
"""

import os
from collections.abc import Sequence
from typing import Literal

import torch
from sentence_transformers import SentenceTransformer

from salemdex_retrieval.checks import check_positive_int
from salemdex_retrieval.errors import ModelLoadError

type Device = Literal["auto", "cuda", "mps", "cpu"]
type ResolvedDevice = Literal["cuda", "mps", "cpu"]


class HuggingFaceDenseEmbedder:
    """A `DenseEmbedder` running a sentence-transformers model in this process.

    Queries go through the model's `encode_query` and passages through its
    `encode_document`, so each gets the prompt the model declares for it, or none.

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
        self._device = _resolve_device(device)
        try:
            self._model = SentenceTransformer(
                os.fspath(model),
                device=self._device,
                # Full precision on every device: vectors made on a GPU then match
                # those made on the CPU, so moving the worker doesn't change ranking.
                model_kwargs={"dtype": torch.float32},
            )
        except (OSError, ValueError) as error:
            raise ModelLoadError(
                f"can't load model {os.fspath(model)!r} as a local folder or a hub model"
            ) from error

        self._batch_size = batch_size

    def embed_queries(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one row of numbers per question in `texts`, in the same order.

        Args:
            texts: The questions, as asked.

        Returns:
            One row per text, each as long as the model's output.
        """
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
        embeddings = self._model.encode_document(
            list(texts), batch_size=self._batch_size, show_progress_bar=False
        )
        # The shape and the numbers are checked where the rows enter (Dense.value).
        rows: list[list[float]] = embeddings.tolist()
        return rows


def _resolve_device(device: Device) -> ResolvedDevice:
    """Return the device to run on: `device` itself, or the best one for `"auto"`.

    Raises:
        ValueError: `device` names a GPU PyTorch can't find.
    """
    cuda_available = torch.cuda.is_available()
    if device == "cuda":
        if not cuda_available:
            raise ValueError(
                "device 'cuda' was asked for, but PyTorch found no CUDA-compatible GPU; use 'auto' or 'cpu'"
            )
        return "cuda"

    mps_available = torch.backends.mps.is_available()
    if device == "mps":
        if not mps_available:
            raise ValueError(
                "device 'mps' was asked for, but PyTorch found no MPS-compatible GPU; use 'auto' or 'cpu'"
            )
        return "mps"

    if device == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
    return "cpu"
