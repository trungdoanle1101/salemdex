"""What the Hugging Face adapters share: choosing a device and loading a model.

Behind the `huggingface` extra, like the adapters: importing it loads torch.
"""

import os
from collections.abc import Callable
from typing import Literal

import torch

from salemdex_retrieval.errors import ModelLoadError

# Where a model may be asked to run; "auto" picks the best device PyTorch finds.
type Device = Literal["auto", "cuda", "mps", "cpu"]
# Where a model runs, once "auto" is decided.
type ResolvedDevice = Literal["cuda", "mps", "cpu"]


def resolve_device(device: Device) -> ResolvedDevice:
    """Return the device to run on: `device` itself, or the best one for `"auto"`.

    Args:
        device: The device asked for. `"auto"` picks CUDA, then MPS, then the CPU.

    Returns:
        The device the model will run on.

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


def load_model[T](
    model: str | os.PathLike[str],
    *,
    model_class: Callable[..., T],
    device: ResolvedDevice,
) -> T:
    """Return `model` loaded by `model_class`, in full precision on `device`.

    Args:
        model: A local folder or a hub id, as sentence-transformers takes it.
        model_class: The sentence-transformers class that loads it, such as
            `SentenceTransformer` or `CrossEncoder`.
        device: Where the model runs.

    Returns:
        The loaded model.

    Raises:
        ModelLoadError: No local folder or hub model has that name, or what is there
            isn't a model. The error from Hugging Face is chained.
    """
    try:
        return model_class(
            os.fspath(model),
            device=device,
            # Full precision on every device: results made on a GPU then match those
            # made on the CPU, so moving the worker doesn't change ranking.
            model_kwargs={"dtype": torch.float32},
        )
    except (OSError, ValueError) as error:
        raise ModelLoadError(
            f"can't load model {os.fspath(model)!r} as a local folder or a hub model"
        ) from error
