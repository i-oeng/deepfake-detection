import importlib

import pytest

TORCH_MODULES = ("benchmark", "fusion", "loop", "rgb", "summarize")


@pytest.mark.parametrize("name", TORCH_MODULES)
def test_training_module_imports(name: str) -> None:
    """Catch cross-module import breaks that unit tests of single modules miss."""
    pytest.importorskip("torch")
    importlib.import_module(f"deepfake_detection.train.{name}")
