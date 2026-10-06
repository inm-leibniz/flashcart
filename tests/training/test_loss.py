from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from flashcart.training.loss import LossAccumulator, PropertyLossFunction, WeightedSumLoss, loss_from_config
from flashcart.utils.torch_geometric.data import Data


def _huber(res: torch.Tensor, delta: float) -> torch.Tensor:
    return F.huber_loss(res, torch.zeros_like(res), reduction="none", delta=delta)


@pytest.mark.parametrize(
    ("reduce", "extensive", "denominator"),
    [
        pytest.param("huber_sum", True, 1, id="sum"),
        pytest.param("huber_mean", False, 4 * 3, id="mean"),
    ],
)
def test_huber_reduction_matches_functional_reference(reduce, extensive, denominator):
    batch = Data(forces=torch.ones(4, 3), n_atoms=torch.tensor([2, 2]))
    results = {"forces": torch.ones(4, 3) + 0.1}
    loss = PropertyLossFunction("forces", reduce=reduce, normalization="none", delta=0.1)

    assert loss.is_extensive is extensive
    expected = _huber(results["forces"] - batch.forces, 0.1).sum() / denominator
    assert torch.allclose(loss(results, batch), expected)


def test_loss_from_config_with_delta():
    loss = loss_from_config(
        {
            "weighted_sum": [
                {
                    "property": "energy",
                    "reduce": "huber_sum",
                    "normalization": "per_sqrt_atom",
                    "delta": 0.01,
                    "weight": 1.0,
                },
                {"property": "forces", "reduce": "huber_sum", "normalization": "none", "delta": 0.05, "weight": 0.05},
                {"property": "stress", "reduce": "huber_sum", "normalization": "none", "delta": 0.1, "weight": 0.01},
            ]
        }
    )
    assert isinstance(loss, WeightedSumLoss)
    assert loss.losses[0].delta == 0.01
    assert loss.losses[1].delta == 0.05
    assert loss.losses[2].key == "stress"


def test_training_loss_scales_only_extensive_terms():
    batch = SimpleNamespace(
        energy=torch.tensor([1.0, 2.0]),
        forces=torch.zeros(2, 3),
        n_atoms=torch.tensor([1, 1]),
    )
    results = {
        "energy": torch.tensor([2.0, 4.0]),
        "forces": torch.ones(2, 3),
    }
    loss = WeightedSumLoss(
        [
            PropertyLossFunction("energy", reduce="sse", normalization="none"),
            PropertyLossFunction("forces", reduce="mae", normalization="none"),
        ]
    )

    assert loss.is_extensive is False
    torch.testing.assert_close(loss(results, batch), torch.tensor(6.0))
    torch.testing.assert_close(loss.training_loss(results, batch, world_size=4), torch.tensor(21.0))


@pytest.mark.parametrize(
    ("local_errors", "peer_errors"),
    [
        ([2.0, -5.0], [-3.0, 4.0]),
        ([-3.0, 4.0], [2.0, -5.0]),
        ([], [2.0, -5.0, -3.0, 4.0]),
    ],
)
@pytest.mark.parametrize("weighted", [False, True])
def test_distributed_maxe(local_errors, peer_errors, weighted):
    max_loss = PropertyLossFunction("energy", reduce="maxe", normalization="none")
    loss = (
        WeightedSumLoss(
            [
                max_loss,
                PropertyLossFunction("energy", reduce="mae", normalization="none"),
            ],
            weights=[2.0, 3.0],
        )
        if weighted
        else max_loss
    )
    accumulator = LossAccumulator(loss)

    for error in local_errors:
        batch = SimpleNamespace(
            energy=torch.zeros(1, dtype=torch.float64),
            n_atoms=torch.ones(1, dtype=torch.long),
        )
        accumulator.update(
            {"energy": torch.tensor([error], dtype=torch.float64)},
            batch,
        )

    peer = torch.tensor(peer_errors, dtype=torch.float64)

    def all_reduce(value, reduce_op):
        if value.ndim == 1:
            assert reduce_op == "sum"
            return value + peer.numel()
        if reduce_op == "max":
            return torch.maximum(value, peer.abs().max())
        assert reduce_op == "sum"
        return value + peer.abs().sum()

    fabric = SimpleNamespace(
        world_size=2,
        device=torch.device("cpu"),
        all_reduce=all_reduce,
    )
    errors = torch.tensor(local_errors + peer_errors, dtype=torch.float64)
    expected = errors.abs().max()
    if weighted:
        expected = 2.0 * expected + 3.0 * errors.abs().mean()

    torch.testing.assert_close(accumulator.compute(fabric), expected)
