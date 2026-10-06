import torch

from flashcart.data.samplers import DynamicBatchSampler
from flashcart.utils.torch_geometric.dataloader import DataLoader


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> int:
        return idx


def test_epoch_advances_only_after_full_pass():
    sampler = DynamicBatchSampler(
        sizes=[(1, 1)] * 10,
        max_nodes=3,
        max_edges=3,
        shuffle=True,
        seed=5,
    )
    loader = DataLoader(TinyDataset(10), batch_sampler=sampler)

    partial = iter(loader)
    next(partial)
    del partial

    expected_epoch0 = torch.randperm(10, generator=torch.Generator().manual_seed(5)).tolist()
    expected_epoch1 = torch.randperm(10, generator=torch.Generator().manual_seed(6)).tolist()

    assert torch.cat([batch for batch in loader]).tolist() == expected_epoch0
    assert torch.cat([batch for batch in loader]).tolist() == expected_epoch1


def test_balanced_batches_never_empty_across_ranks():
    sizes = [(1, 1), (1, 1)]
    rank_batches = [
        list(
            DynamicBatchSampler.from_target_batch_size(
                sizes,
                target_batch_size=1,
                n_replicas=4,
                rank=rank,
                shuffle=False,
                drop_last=False,
                balance_batches=True,
            )
        )
        for rank in range(4)
    ]

    assert [len(batches) for batches in rank_batches] == [1, 1, 1, 1]
