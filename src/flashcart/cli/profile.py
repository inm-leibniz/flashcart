"""Profile FlashCart inference and training paths (the ``flashcart-profile`` script)."""

import argparse
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from flashcart.data.dataset import make_loader
from flashcart.data.utils import get_elements
from flashcart.model.flashcart import FlashCartPotential
from flashcart.o3._tensor_product import KERNEL_L_MAX
from flashcart.training.loss import loss_from_config
from flashcart.training.tasks import predict_kwargs
from flashcart.utils.config import load_config, parse_config_args, torch_device_from_config

PROFILE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "profile.yaml"

# The [timer] and [phase-timer] lines report timings for these phase labels.
# The operation tables omit them, while traces retain them.
_PHASE_PREFIX = "phase:"


def _sync(device: torch.device) -> None:
    """Wait for pending CUDA operations. Other device types require no action."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _set_requires_grad(model: torch.nn.Module, requires_grad: bool) -> None:
    """Set the gradient requirement of every model parameter."""
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def _load_profile_config(overrides: dict[str, Any]) -> dict[str, Any]:
    """Combine training defaults, profiling defaults, and caller overrides.

    Args:
        overrides (dict[str, Any]): Settings that replace defaults for matching keys.

    Returns:
        dict[str, Any]: Merged configuration. Caller settings replace profiling
            defaults, which replace training defaults.
    """
    cfg = load_config(PROFILE_CONFIG, required=[])
    cfg.update(overrides)
    return cfg


def _make_profile_loader(
    path: Path,
    cfg: dict[str, Any],
    model: FlashCartPotential,
    batch_size: int,
    shuffle: bool = False,
) -> Any:
    """Build a loader using the model cutoff and profiling batch settings.

    Args:
        path (Path): Structure dataset to load.
        cfg (dict[str, Any]): Reference-property keys, seed, and profiling settings.
        model (FlashCartPotential): Model supplying the element order and cutoff.
        batch_size (int): Structures per batch, or the target used for dynamic budgets.
        shuffle (bool, optional): Whether to shuffle structures. Default: False.

    Returns:
        DataLoader: Loader constructing graph batches for the requested dataset.
    """
    return make_loader(
        path,
        model.elements,
        model.r_max,
        batch_size=batch_size,
        energy_key=cfg.get("energy_key", "REF_energy"),
        forces_key=cfg.get("forces_key", "REF_forces"),
        shuffle=shuffle,
        dynamic_batching=cfg.get("profile_dynamic_batching", False),
        data_seed=int(cfg.get("data_seed", 0)),
    )


def _materialize_profile_batches(loader: Iterable[Any], device: torch.device, n_batches: int) -> list[Any]:
    """Move up to the requested number of batches to the profiling device.

    CUDA transfers are synchronized before returning, outside the timed computation.

    Args:
        loader (Iterable): Source of graph batches.
        device (torch.device): Device used for profiling.
        n_batches (int): Maximum number of batches to keep.

    Returns:
        list: Graph batches on the requested device.

    Raises:
        RuntimeError: No batches were collected.
    """
    batches = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= n_batches:
            break
        batches.append(batch.to(device))
    if not batches:
        raise RuntimeError("No batches available for profiling.")
    _sync(device)
    return batches


def _batch_summary(batches: Sequence[Any]) -> str:
    """Describe the batch, structure, atom, and edge counts.

    Args:
        batches (Sequence): Graph batches to summarize.

    Returns:
        str: Counts formatted for the profiling report.
    """
    atoms = sum(int(b.n_atoms.sum().item()) for b in batches if hasattr(b, "n_atoms"))
    structures = sum(int(b.n_atoms.shape[0]) for b in batches if hasattr(b, "n_atoms"))
    edges = sum(int(b.edge_index.shape[1]) for b in batches if hasattr(b, "edge_index"))
    return f"batches={len(batches)}, structures={structures}, atoms={atoms}, edges={edges}"


def _time_run(label: str, run: Callable[[], Any], device: torch.device, warmup: int, repeat: int) -> None:
    """Print elapsed wall time for repeated calls after untimed warmup.

    CUDA execution is synchronized before and after the measured repetitions.
    The reported seconds per run are the total elapsed time divided by ``repeat``.
    One run performs all work in the supplied callable.

    Args:
        label (str): Name printed with the timing result.
        run (Callable): No-argument function performing one complete run.
        device (torch.device): Device synchronized when using CUDA.
        warmup (int): Number of untimed calls.
        repeat (int): Number of measured calls. Must be positive.
    """
    for _ in range(warmup):
        run()
        _sync(device)
    _sync(device)
    start = time.perf_counter()
    for _ in range(repeat):
        run()
    _sync(device)
    elapsed = time.perf_counter() - start
    print(f"[timer] {label}: {elapsed:.4f} s total, {elapsed / repeat:.4f} s/run")


def _event_cuda_time_us(evt: Any, device: torch.device) -> float:
    """Read an event's accumulated CUDA time using the available profiler attribute.

    Args:
        evt: Averaged profiler event.
        device (torch.device): Device used for profiling.

    Returns:
        float: Device time in microseconds, or zero for CPU profiling or an event
            without a CUDA or device-time attribute.
    """
    if device.type != "cuda":
        return 0.0
    for attr in ("cuda_time_total", "device_time_total"):
        if hasattr(evt, attr):
            return float(getattr(evt, attr))
    return 0.0


def _operation_averages(prof: torch.profiler.profile) -> Any:
    """Return operation averages with the enclosing phase labels removed.

    Args:
        prof (torch.profiler.profile): Completed profiler run.

    Returns:
        EventList: Averaged operation events, excluding names beginning with ``phase:``.
    """
    averages = prof.key_averages()
    averages[:] = [evt for evt in averages if not evt.key.startswith(_PHASE_PREFIX)]
    return averages


def _print_interesting_ops(prof: torch.profiler.profile, device: torch.device, row_limit: int) -> None:
    """Print selected model, tensor, and autograd operations ordered by elapsed time.

    Args:
        prof (torch.profiler.profile): Completed profiler run.
        device (torch.device): Device whose elapsed times determine the ordering.
        row_limit (int): Maximum number of operations to print.
    """
    keywords = (
        "flashcart::",
        "tp_",
        "irreps",
        "triton",
        "aten::mm",
        "aten::bmm",
        "aten::addmm",
        "aten::scatter",
        "aten::index",
        "aten::copy",
        "autograd",
    )
    rows = [evt for evt in _operation_averages(prof) if any(k in evt.key for k in keywords)]
    if device.type == "cuda":
        rows.sort(key=lambda evt: _event_cuda_time_us(evt, device), reverse=True)
        sort_name = "cuda/device_time_total"
    else:
        rows.sort(key=lambda evt: evt.cpu_time_total, reverse=True)
        sort_name = "cpu_time_total"
    if not rows:
        return

    print(f"\n=== filtered interesting operations by {sort_name} ===")
    print(f"{'op':70s} {'calls':>7s} {'cpu ms':>12s} {'cuda ms':>12s}")
    print("-" * 107)
    for evt in rows[:row_limit]:
        cpu_ms = evt.cpu_time_total / 1000.0
        cuda_ms = _event_cuda_time_us(evt, device) / 1000.0
        print(f"{evt.key[:70]:70s} {evt.count:7d} {cpu_ms:12.3f} {cuda_ms:12.3f}")


def _profile_run(
    label: str,
    run: Callable[[], Any],
    device: torch.device,
    warmup: int,
    repeat: int,
    row_limit: int,
    profile_memory: bool,
    record_shapes: bool,
    with_stack: bool,
    trace_dir: Optional[Path],
) -> None:
    """Collect operation timings, optional traces, and CUDA memory statistics.

    Warmup calls precede profiling. Operation times include profiler overhead and
    should be used to locate expensive operations rather than compare total runtimes.

    Args:
        label (str): Name used in printed reports and trace filenames.
        run (Callable): No-argument function performing one complete run.
        device (torch.device): Device used for profiling and synchronization.
        warmup (int): Number of untimed warmup calls.
        repeat (int): Number of calls recorded by the profiler.
        row_limit (int): Maximum number of operations in each printed table.
        profile_memory (bool): Record memory events in the profiler.
        record_shapes (bool): Record tensor shapes for profiled operations.
        with_stack (bool): Record Python stack traces for profiled operations.
        trace_dir (Path or None): Directory for Chrome traces. None disables export.
    """
    for _ in range(warmup):
        run()
        _sync(device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.profiler.profile(
        activities=activities,
        profile_memory=profile_memory,
        record_shapes=record_shapes,
        with_stack=with_stack,
        acc_events=True,  # A single cycle only needs this to silence the event-reset warning.
    ) as prof:
        for _ in range(repeat):
            run()
            _sync(device)

    sort_by = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print(f"\n=== {label}: top operations by {sort_by} ({repeat} profiled runs) ===")
    print(_operation_averages(prof).table(sort_by=sort_by, row_limit=row_limit))
    _print_interesting_ops(prof, device=device, row_limit=row_limit)
    print(
        "[note] times in these tables include profiler overhead (large for many small kernels); "
        "use the [timer] lines for absolute timings"
    )

    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{label.replace(' ', '_').replace('+', 'p')}.json"
        prof.export_chrome_trace(str(trace_path))
        print(f"[trace] wrote {trace_path}")
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"[cuda] peak allocated during {label}: {peak_mb:.1f} MiB")


def _make_inference_run(
    model: FlashCartPotential,
    batches: Sequence[Any],
    predict_compile_kwargs: dict[str, Any],
) -> Callable[[], None]:
    """Build a callable that evaluates energies and forces for all supplied batches.

    Autograd remains enabled for force evaluation. Parameter gradient requirements
    are not changed by this helper.

    Args:
        model (FlashCartPotential): Model used for prediction.
        batches (Sequence): Graph batches already on the model's device.
        predict_compile_kwargs (dict[str, Any]): Prediction compilation options.

    Returns:
        Callable[[], None]: Function performing one complete inference run.
    """
    def run() -> None:
        with torch.enable_grad(), torch.profiler.record_function("phase:inference_forward_plus_force_bwd"):
            for batch in batches:
                out = model.predict(batch, compute_forces=True, **predict_compile_kwargs)
                _ = out["energy"].sum() + out["forces"].sum() * 0.0
                del out

    return run


def _make_training_run(
    model: FlashCartPotential,
    batches: Sequence[Any],
    train_loss: Any,
    predict_compile_kwargs: dict[str, Any],
) -> Callable[[], None]:
    """Build a callable for prediction, loss evaluation, and backward passes.

    Each call processes all supplied batches without optimizer updates. Gradients
    accumulate across batches and are cleared before and after the call. Required
    energy derivatives are selected from the loss function.

    Args:
        model (FlashCartPotential): Model with parameter gradients enabled.
        batches (Sequence): Graph batches already on the model's device.
        train_loss (LossFunction): Loss function evaluated for each batch.
        predict_compile_kwargs (dict[str, Any]): Prediction compilation options.

    Returns:
        Callable[[], None]: Function performing one complete training run.
    """
    kwargs = predict_kwargs([train_loss])

    def run() -> None:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            for batch in batches:
                with torch.profiler.record_function("phase:training_predict_create_graph"):
                    out = model.predict(batch, create_graph=True, **kwargs, **predict_compile_kwargs)
                with torch.profiler.record_function("phase:training_loss"):
                    loss = train_loss(out, batch)
                with torch.profiler.record_function("phase:training_backward_dbwd"):
                    loss.backward()
                del out, loss
        model.zero_grad(set_to_none=True)

    return run


def _time_training_phases(
    model: FlashCartPotential,
    batches: Sequence[Any],
    train_loss: Any,
    device: torch.device,
    warmup: int,
    repeat: int,
    predict_compile_kwargs: dict[str, Any],
) -> None:
    """Report separate times for prediction, loss evaluation, and differentiation.

    Each phase is measured with CUDA synchronization at its boundaries. The resulting
    sum can differ from timing the complete training computation without intermediate
    synchronization. Gradients are cleared before and after processing all batches.

    Args:
        model (FlashCartPotential): Model with parameter gradients enabled.
        batches (Sequence): Graph batches already on the model's device.
        train_loss (LossFunction): Loss function evaluated for each batch.
        device (torch.device): Device used for computation and synchronization.
        warmup (int): Number of untimed repetitions over all batches.
        repeat (int): Number of measured repetitions. Must be positive.
        predict_compile_kwargs (dict[str, Any]): Prediction compilation options.
    """
    kwargs = predict_kwargs([train_loss])

    def measure_once() -> dict[str, float]:
        totals = {"predict_create_graph": 0.0, "loss": 0.0, "backward_dbwd": 0.0}
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            for batch in batches:
                _sync(device)
                start = time.perf_counter()
                out = model.predict(batch, create_graph=True, **kwargs, **predict_compile_kwargs)
                _sync(device)
                totals["predict_create_graph"] += time.perf_counter() - start

                start = time.perf_counter()
                loss = train_loss(out, batch)
                _sync(device)
                totals["loss"] += time.perf_counter() - start

                start = time.perf_counter()
                loss.backward()
                _sync(device)
                totals["backward_dbwd"] += time.perf_counter() - start
                del out, loss
        model.zero_grad(set_to_none=True)
        return totals

    for _ in range(warmup):
        measure_once()

    accum = {"predict_create_graph": 0.0, "loss": 0.0, "backward_dbwd": 0.0}
    for _ in range(repeat):
        one = measure_once()
        for key, value in one.items():
            accum[key] += value

    total = sum(accum.values())
    print("[phase-timer] training:")
    for key, value in accum.items():
        avg = value / repeat
        share = 100.0 * value / total if total > 0 else 0.0
        print(f"  {key:20s} {avg:.4f} s/run  {share:5.1f}%")


def profile(**overrides: Any) -> None:
    """Measure prediction and differentiation costs for a newly initialized model.

    Inference includes energy and force evaluation. The training phase includes
    prediction, loss evaluation, and differentiation with respect to model parameters.
    It does not perform optimizer steps. Data loading, neighbor-list construction,
    and device transfers take place before timing. Each timed run processes all
    selected batches. Reports include elapsed times and profiler operation tables,
    with optional trace files.

    The model uses newly initialized weights and optional fitted data statistics rather
    than loading a checkpoint.

    Args:
        **overrides: Config keys merged over ``flashcart/configs/profile.yaml``. Model
            and data keys are the training config keys. Profiling behavior is controlled
            by the ``profile_*`` keys listed there.
    """
    cfg = _load_profile_config(overrides)

    phase = cfg["profile_phase"]
    if phase not in ("inference", "training", "both"):
        raise ValueError("profile_phase must be one of: inference, training, both.")
    compile_mode = str(cfg.get("profile_compile_mode", "reduce-overhead"))
    if compile_mode not in ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"):
        raise ValueError(f"Unsupported profile_compile_mode: {compile_mode!r}.")

    # One data source per phase: the explicit profile_* path, else the training set.
    inference_path = cfg.get("profile_data_path") or cfg.get("train_path")
    training_path = cfg.get("profile_train_path") or cfg.get("train_path") or cfg.get("profile_data_path")
    if phase in ("inference", "both") and inference_path is None:
        raise ValueError("Provide profile_data_path or train_path.")
    if phase in ("training", "both") and training_path is None:
        raise ValueError("Provide profile_train_path, train_path, or profile_data_path.")
    inference_path = Path(inference_path) if inference_path is not None else None
    training_path = Path(training_path) if training_path is not None else None

    if cfg.get("elements") is None:
        cfg["elements"] = get_elements(
            training_path or inference_path,
            cfg.get("energy_key", "REF_energy"),
            cfg.get("forces_key", "REF_forces"),
        )

    device = torch.device(torch_device_from_config(cfg.get("device", "gpu")))
    torch.manual_seed(int(cfg.get("model_seed", 0)))
    model = FlashCartPotential.from_config(cfg)

    prefit_path = training_path or inference_path
    if cfg.get("profile_pre_fit", True) and prefit_path is not None:
        prefit_loader = _make_profile_loader(
            prefit_path,
            cfg,
            model,
            int(cfg.get("profile_batch_size", 64)),
        )
        if hasattr(prefit_loader, "dataset") and hasattr(prefit_loader.dataset, "get_sizes"):
            prefit_loader.dataset.get_sizes()
        model.pre_fit(prefit_loader)

    model.to(device)
    print(model)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"device: {device}")
    print("weights: fresh random initialization")
    print(f"use_triton: {model.use_triton}")
    print(f"kernel_l_max: {KERNEL_L_MAX}")

    warmup = int(cfg["profile_warmup"])
    repeat = int(cfg["profile_repeat"])
    profile_run_kwargs = {
        "row_limit": int(cfg["profile_row_limit"]),
        "profile_memory": bool(cfg["profile_memory"]),
        "record_shapes": bool(cfg["profile_record_shapes"]),
        "with_stack": bool(cfg["profile_with_stack"]),
        "trace_dir": Path(cfg["profile_trace_dir"]) if cfg.get("profile_trace_dir") else None,
    }
    predict_compile_kwargs = {}
    if bool(cfg.get("profile_compile_predict", False)):
        predict_compile_kwargs = {
            "use_compile": True,
            "compile_mode": compile_mode,
            "fullgraph": bool(cfg.get("profile_compile_fullgraph", True)),
        }
        print(f"predict compile: {compile_mode}, fullgraph={predict_compile_kwargs['fullgraph']}")

    if phase in ("inference", "both"):
        model.eval()
        _set_requires_grad(model, False)
        loader = _make_profile_loader(inference_path, cfg, model, int(cfg["profile_batch_size"]))
        if hasattr(loader, "dataset") and hasattr(loader.dataset, "get_sizes"):
            loader.dataset.get_sizes()
        batches = _materialize_profile_batches(loader, device, int(cfg["profile_n_batches"]))
        print(f"\n[inference] data={inference_path} {_batch_summary(batches)}")
        run = _make_inference_run(model, batches, predict_compile_kwargs)
        _time_run("inference fwd+bwd", run, device, warmup, repeat)
        _profile_run("inference fwd+bwd", run, device, warmup, repeat, **profile_run_kwargs)

    if phase in ("training", "both"):
        model.train()
        _set_requires_grad(model, True)
        train_batch_size = int(cfg.get("profile_train_batch_size") or cfg["profile_batch_size"])
        loader = _make_profile_loader(training_path, cfg, model, train_batch_size)
        if hasattr(loader, "dataset") and hasattr(loader.dataset, "get_sizes"):
            loader.dataset.get_sizes()
        batches = _materialize_profile_batches(loader, device, int(cfg["profile_n_batches"]))
        train_loss = loss_from_config(cfg["train_loss"])
        print(f"\n[training] data={training_path} {_batch_summary(batches)}")
        run = _make_training_run(model, batches, train_loss, predict_compile_kwargs)
        _time_training_phases(model, batches, train_loss, device, warmup, repeat, predict_compile_kwargs)
        _time_run("training fwd+bwd+dbwd", run, device, warmup, repeat)
        _profile_run("training fwd+bwd+dbwd", run, device, warmup, repeat, **profile_run_kwargs)


def main() -> None:
    """CLI entry point: ``flashcart-profile [CONFIG.yaml] [KEY=VALUE ...]``."""
    parser = argparse.ArgumentParser(
        description="Profile FlashCart inference and force-training code paths.",
        epilog=f"Profiling knobs and their defaults: {PROFILE_CONFIG}",
    )
    parser.add_argument(
        "config_and_overrides",
        nargs="*",
        metavar="[CONFIG.yaml] [KEY=VALUE ...]",
    )
    args = parser.parse_args()
    profile(**parse_config_args(args.config_and_overrides))


if __name__ == "__main__":
    main()
