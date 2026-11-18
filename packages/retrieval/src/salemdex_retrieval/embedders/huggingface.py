"""A dense embedder running a sentence-transformers model, behind the `huggingface` extra.

The model is used as Hugging Face publishes it: nothing about it is kept here. Its query
and document prompts, if it declares any, come from its own configuration.
"""

import os
from collections.abc import Sequence
from typing import Final

from sentence_transformers import SentenceTransformer

from salemdex_retrieval._huggingface import Device, load_model, resolve_device
from salemdex_retrieval.checks import check_positive_int


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
