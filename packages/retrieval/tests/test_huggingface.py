"""What both Hugging Face adapters share: settings checked before loading, the device,
and the error for a model that can't be loaded. None of these tests needs a model or the
network.
"""

import re
from pathlib import Path
from typing import Any

import pytest
import torch

from salemdex_retrieval._huggingface import Device, ResolvedDevice, resolve_device
from salemdex_retrieval.embedders.huggingface import HuggingFaceDenseEmbedder
from salemdex_retrieval.errors import ModelLoadError
from salemdex_retrieval.rerankers.huggingface import HuggingFaceReranker

ADAPTERS = [
    pytest.param(HuggingFaceDenseEmbedder, id="embedder"),
    pytest.param(HuggingFaceReranker, id="reranker"),
]


# The device


@pytest.mark.parametrize(
    ("device", "cuda", "mps", "expected"),
    [
        ("auto", True, True, "cuda"),
        ("auto", False, True, "mps"),
        ("auto", False, False, "cpu"),
        ("cpu", True, True, "cpu"),
        ("cuda", True, False, "cuda"),
        ("mps", False, True, "mps"),
    ],
)
def test_the_device_is_the_one_asked_for_or_the_best_one_for_auto(
    device: Device,
    cuda: bool,
    mps: bool,
    expected: ResolvedDevice,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_gpus(monkeypatch, cuda=cuda, mps=mps)

    assert resolve_device(device) == expected


# Settings, checked before the model loads


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize("batch_size", [0, -1, True, 2.0])
def test_a_batch_size_that_is_not_a_positive_int_is_refused(
    adapter: Any, batch_size: object, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        adapter(tmp_path / "missing", device="cpu", batch_size=batch_size)


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_a_gpu_pytorch_cannot_find_is_refused_naming_it(
    adapter: Any, device: Device, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_gpus(monkeypatch, cuda=False, mps=False)

    message = f"device {device!r} was asked for, but PyTorch found no"
    with pytest.raises(ValueError, match=re.escape(message)):
        adapter(tmp_path / "missing", device=device, batch_size=8)


# Loading


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_a_model_that_cannot_be_found_raises_naming_it(
    adapter: Any, tmp_path: Path
) -> None:
    missing = tmp_path / "missing"

    message = f"can't load model {str(missing)!r} as a local folder or a hub model"
    with pytest.raises(ModelLoadError, match=re.escape(message)) as raised:
        adapter(missing, device="cpu", batch_size=8)

    assert isinstance(raised.value.__cause__, OSError)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_a_folder_that_is_not_a_model_raises_the_same_error(
    adapter: Any, tmp_path: Path
) -> None:
    not_a_model = tmp_path / "notes"
    not_a_model.mkdir()

    message = f"can't load model {str(not_a_model)!r} as a local folder or a hub model"
    with pytest.raises(ModelLoadError, match=re.escape(message)) as raised:
        adapter(not_a_model, device="cpu", batch_size=8)

    assert isinstance(raised.value.__cause__, ValueError)


def test_a_load_error_is_neither_an_os_error_nor_a_value_error() -> None:
    # Code catching a bad batch size (ValueError) or a file problem (OSError) mustn't
    # swallow a model that can't be loaded by accident.
    assert not issubclass(ModelLoadError, (OSError, ValueError))


def _set_gpus(monkeypatch: pytest.MonkeyPatch, *, cuda: bool, mps: bool) -> None:
    """Make PyTorch report the given GPUs as available or not."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
