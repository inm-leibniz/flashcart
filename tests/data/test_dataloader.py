import pytest
import torch

from flashcart.data import dataset as dataset_module
from flashcart.data.samplers import DistributedStridedSampler, DynamicBatchSampler
from flashcart.utils.torch_geometric.dataloader import DataLoader


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> int:
        return idx


def _flatten(loader: DataLoader) -> list[int]:
    return torch.cat([batch for batch in loader]).tolist()


def test_make_loader_uses_data_seed_for_dynamic_batching(monkeypatch):
    monkeypatch.setattr(
        dataset_module.AtomicDataset,
        "from_extxyz",
        staticmethod(lambda *args, **kwargs: TinyDataset(10)),
    )
    monkeypatch.setattr(TinyDataset, "get_sizes", lambda self: [(1, 1)] * len(self), raising=False)

    loader = dataset_module.make_loader(
        "unused.xyz",
        ["H"],
        r_max=1.0,
        batch_size=3,
        shuffle=True,
        dynamic_batching=True,
        data_seed=17,
    )

    expected = torch.randperm(10, generator=torch.Generator().manual_seed(17)).tolist()
    assert _flatten(loader) == expected


def test_make_loader_uses_data_seed_for_fixed_batching(monkeypatch):
    monkeypatch.setattr(
        dataset_module.AtomicDataset,
        "from_extxyz",
        staticmethod(lambda *args, **kwargs: TinyDataset(10)),
    )

    def order(seed: int) -> list[int]:
        loader = dataset_module.make_loader(
            "unused.xyz",
            ["H"],
            r_max=1.0,
            batch_size=3,
            shuffle=True,
            dynamic_batching=False,
            data_seed=seed,
        )
        return _flatten(loader)

    assert order(17) == order(17)
    assert order(17) != order(18)


@pytest.mark.parametrize(
    ("loader_kwargs", "expected"),
    [
        pytest.param({"dynamic_batching": True}, DynamicBatchSampler, id="dynamic"),
        pytest.param(
            {"dynamic_batching": False, "n_replicas": 3, "rank": 0, "balance_batches": True},
            torch.utils.data.DistributedSampler,
            id="balanced",
        ),
        pytest.param(
            {"dynamic_batching": False, "n_replicas": 3, "rank": 0},
            DistributedStridedSampler,
            id="strided",
        ),
    ],
)
def test_make_loader_sampler_matches_batching_mode(monkeypatch, loader_kwargs, expected):
    monkeypatch.setattr(
        dataset_module.AtomicDataset,
        "from_extxyz",
        staticmethod(lambda *args, **kwargs: TinyDataset(20)),
    )
    monkeypatch.setattr(TinyDataset, "get_sizes", lambda self: [(1, 1)] * len(self), raising=False)

    loader = dataset_module.make_loader(
        "unused.xyz",
        ["H"],
        r_max=1.0,
        batch_size=3,
        shuffle=True,
        data_seed=17,
        **loader_kwargs,
    )

    sampler = loader.batch_sampler if expected is DynamicBatchSampler else loader.sampler
    assert isinstance(sampler, expected)
    assert sampler.seed == 17
    if expected is torch.utils.data.DistributedSampler:
        assert len(sampler) == 7
    if expected is DistributedStridedSampler:
        assert len(sampler) == 7

