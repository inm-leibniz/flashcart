import math

import pytest

from helpers import tiny_model_cfg, write_tiny_extxyz
from flashcart.model.atomistic import save_inference_metadata
from flashcart.model.flashcart import FlashCartPotential

pytestmark = pytest.mark.slow


def test_eval_from_checkpoint_without_train_path(tmp_path):
    cfg = tiny_model_cfg(avg_neighbors=1.0, fit_atomic_shifts=False)
    model = FlashCartPotential.from_config(cfg)
    ckpt = tmp_path / "best"
    ckpt.mkdir()
    model.save_weights(ckpt)
    save_inference_metadata(ckpt, model.to_model_config())

    valid_path = tmp_path / "valid.extxyz"
    write_tiny_extxyz(valid_path, n_structures=2)

    from flashcart.cli.test import test as run_cli_test

    metrics = run_cli_test(
        checkpoint=str(ckpt),
        train_path=None,
        valid_path=str(valid_path),
        device="cpu",
        devices=1,
        valid_losses=[{"property": "energy", "reduce": "mae", "normalization": "none"}],
        dynamic_batching=False,
        eval_batch_size=1,
    )

    assert set(metrics) == {"valid"}
    assert metrics["valid"], "evaluation returned no metrics"
    for name, value in metrics["valid"].items():
        assert math.isfinite(float(value)), f"{name} is not finite: {value}"
    assert (tmp_path / "test.log").exists()
