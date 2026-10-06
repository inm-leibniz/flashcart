"""Measure CUDA inference and training computations on one atomic structure.

    python examples/timing/measure.py models/my_run test.extxyz

The example uses the best checkpoint and synthetic training targets. Timings exclude
structure reading, neighbor-list construction, device transfers, and optimizer updates.
"""

import argparse
import os
from importlib.metadata import version
from pathlib import Path
from statistics import mean, stdev

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

import torch
import yaml
from ase.io import read

from flashcart.data.graph import graph_from_ase
from flashcart.model.flashcart import FlashCartPotential
from flashcart.training.loss import loss_from_config


def prepare_step(model, graph, phase, train_loss=None, compile_mode=None):
    """Prepare a repeated prediction or training computation.

    Training targets are detached predictions plus seeded Gaussian noise, with
    standard deviations of 0.001 eV per atom for energy and 0.01 eV/Å for forces.
    The supplied model and graph are updated in place.

    Args:
        model (FlashCartPotential): Model already on the selected device.
        graph (AtomicData): Graph already on the same device as the model.
        phase (str): ``"inference"`` or ``"training"``.
        train_loss (LossFunction, optional): Required for training. Only energy and
            force losses are supported. Default: None.
        compile_mode (str, optional): Prediction compilation mode. Default: None,
            which leaves prediction uncompiled.

    Returns:
        Callable: No-argument function returning predictions or a scalar training
            loss. Training clears parameter gradients before each computation.
    """
    if phase not in ("inference", "training"):
        raise ValueError("phase must be 'inference' or 'training'.")
    training = phase == "training"
    if training:
        if train_loss is None:
            raise ValueError("Training requires a loss function.")
        keys = train_loss.compute_keys
        if keys["compute_stress"] or keys["compute_virials"]:
            raise ValueError("This example supports energy and force losses, not stress or virial losses.")
    model.train(training)
    model.requires_grad_(training)
    model.zero_grad(set_to_none=True)
    options = {
        "compute_forces": train_loss.compute_keys["compute_forces"] if training else True,
        "use_compile": compile_mode is not None,
        "compile_mode": compile_mode or "reduce-overhead",
    }

    if training:
        with torch.enable_grad():
            reference = model.predict(graph, compute_forces=options["compute_forces"])
        generator = torch.Generator(device=graph.positions.device).manual_seed(0)
        energy = reference["energy"].detach()
        graph.energy = energy + 0.001 * graph.n_atoms.to(energy) * torch.randn(
            energy.shape, generator=generator, device=energy.device, dtype=energy.dtype
        )
        if options["compute_forces"]:
            forces = reference["forces"].detach()
            graph.forces = forces + 0.01 * torch.randn(
                forces.shape, generator=generator, device=forces.device, dtype=forces.dtype
            )
        del reference

        def step():
            model.zero_grad(set_to_none=True)
            prediction = model.predict(graph, create_graph=True, **options)
            loss = train_loss.training_loss(prediction, graph, world_size=1)
            loss.backward()
            return loss

    else:

        def step():
            return model.predict(graph, **options)

    return step


def measure(step, device, warmup=10, repeat=50):
    """Measure individual calls with CUDA events after untimed warmup.

    Args:
        step (Callable): No-argument function performing the measured computation.
        device (torch.device): CUDA device used by the computation.
        warmup (int, optional): Number of untimed calls. Default: 10.
        repeat (int, optional): Number of measured calls. Default: 50.

    Returns:
        list[float]: Elapsed CUDA-event time in milliseconds for each measured call.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    with torch.cuda.device(device), torch.enable_grad():
        for _ in range(warmup):
            output = step()
            del output
        torch.cuda.synchronize(device)
        for _ in range(repeat):
            torch.cuda.synchronize(device)
            start.record()
            output = step()
            end.record()
            torch.cuda.synchronize(device)
            times.append(start.elapsed_time(end))
            del output
    return times


def main():
    """Load a checkpoint and report inference or training timings on CUDA."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="training output directory containing best/ and config.yaml")
    parser.add_argument("structure", type=Path, help="structure file readable by ASE (only the first frame is used)")
    parser.add_argument("--phase", choices=("inference", "training", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
        default=None,
        help="enable torch.compile for prediction using this mode",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 2:
        parser.error("--warmup must be nonnegative and --repeat must be at least two.")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for this example. Select a GPU with CUDA_VISIBLE_DEVICES.")

    torch.manual_seed(0)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    checkpoint = args.run_dir / "best"
    model = FlashCartPotential.from_checkpoint(checkpoint, device=device).float()
    atoms = read(args.structure, index=0)
    graph = graph_from_ase(atoms, model.elements, model.r_max, device=device)
    if len(atoms) == 0 or graph.edge_index.shape[1] == 0:
        parser.error("The structure must contain atoms and neighbors within the model cutoff.")
    train_loss = None
    if args.phase != "inference":
        config = yaml.safe_load((args.run_dir / "config.yaml").read_text())
        train_loss = loss_from_config(config["train_loss"])
        if train_loss.compute_keys["compute_stress"] or train_loss.compute_keys["compute_virials"]:
            parser.error("This example supports energy and force losses, not stress or virial losses.")

    print(f"checkpoint: {checkpoint.resolve()}")
    print(f"structure: {args.structure.resolve()} (first frame)")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"software: FlashCart {version('flashcart')}, PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"dtype: {next(model.parameters()).dtype}, TF32: disabled")
    print(f"backend: {'Triton' if model.use_triton else 'PyTorch'}, compile: {args.compile_mode or 'disabled'}")
    print(f"system: {len(atoms)} atoms, {graph.edge_index.shape[1]} edges, cutoff {model.r_max:g} Å")
    print(f"calls: {args.warmup} warmup, {args.repeat} measured per phase")
    phases = ("inference", "training") if args.phase == "both" else (args.phase,)
    for phase in phases:
        step = prepare_step(model, graph, phase, train_loss, args.compile_mode)
        samples = measure(step, device, args.warmup, args.repeat)
        average = mean(samples)
        deviation = stdev(samples)
        print(
            f"{phase}: {average:.6f} ± {deviation:.6f} ms/call, "
            f"{average * 1000 / len(atoms):.6f} ± {deviation * 1000 / len(atoms):.6f} µs/atom "
            "(mean ± sample standard deviation)"
        )


if __name__ == "__main__":
    main()
