import pytest
import torch

from flashcart.training.loss import PropertyLossFunction
from flashcart.training.schedulers import make_scheduler, scheduler_from_config
from flashcart.training.tasks import TrainingTask


def _lr_trajectory(*, base_lr=1.0e-2, max_epochs, **kwargs):
    optimizer = torch.optim.AdamW(torch.nn.Linear(1, 1).parameters(), lr=base_lr)
    scheduler = make_scheduler(optimizer, max_epochs=max_epochs, **kwargs)
    optimizer.step()
    lrs = []
    for _ in range(max_epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    return lrs


def test_scheduler_from_config_returns_none_for_constant_lr():
    optimizer = torch.optim.AdamW(torch.nn.Linear(1, 1).parameters(), lr=1.0e-2)
    cfg = {"lr": 1.0e-2, "warmup_epochs": 0, "decay_epochs": 0, "max_epochs": 50}
    assert scheduler_from_config(optimizer, cfg) is None


def test_task_steps_epoch_scheduler_without_validation_metric():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 / (epoch + 1))
    loss = PropertyLossFunction("energy", reduce="mae", normalization="none")
    task = TrainingTask(model, loss, loss, [loss], optimizer, scheduler=scheduler)

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(1, 1)).sum().backward()
        optimizer.step()
        task._step_scheduler()

    assert scheduler.last_epoch == 2
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize("decay_epochs", [0, 20])
def test_lr_follows_warmup_stable_decay_phases(decay_epochs):
    lrs = _lr_trajectory(
        max_epochs=100,
        warmup_epochs=10,
        decay_epochs=decay_epochs,
        start_factor=0.1,
        end_factor=1.0e-3,
        decay_shape="linear",
    )
    assert lrs[0] == pytest.approx(1.0e-3)
    assert max(lrs) == pytest.approx(1.0e-2)
    assert lrs[50] == pytest.approx(1.0e-2)
    if decay_epochs == 0:
        assert all(lr == pytest.approx(1.0e-2) for lr in lrs[10:])
    else:
        decay = lrs[100 - decay_epochs :]
        assert all(later <= earlier + 1e-12 for earlier, later in zip(decay, decay[1:]))
        assert lrs[-1] < 1.0e-3
