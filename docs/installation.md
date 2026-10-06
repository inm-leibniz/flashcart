# Installation

FlashCart requires Python 3.10 or newer and PyTorch 2.9 or newer. Install it in a
virtual environment to keep its dependencies separate from other projects.

## Prepare the environment

For example, on Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Choose a PyTorch build for your hardware using the
[PyTorch installation instructions](https://pytorch.org/get-started/locally/)
before installing FlashCart. For example, the following command installs a
PyTorch build for CUDA 12.8:

```bash
python -m pip install "torch>=2.9" --index-url https://download.pytorch.org/whl/cu128
```

For a CPU or Apple Silicon installation, use the command appropriate to your
platform from the PyTorch instructions.

## Install FlashCart

### From PyPI

```bash
python -m pip install flashcart
```

### From source

Clone the repository and install FlashCart from the local source directory.
The `-e` option makes Python use this directory directly, so edits to the
Python source take effect without reinstalling:

```bash
git clone https://github.com/inm-leibniz/flashcart.git
cd flashcart
python -m pip install -e .
```

## Check the installation

Run the following Python code to check that FlashCart imports and display
its version, the PyTorch version, and the available hardware backends:

```python
import torch
import flashcart
from flashcart.calculators import FlashCartCalculator

print("FlashCart:", flashcart.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())
```

A value of `False` for CUDA or MPS means that backend is unavailable in the
current environment. You can still run FlashCart on the CPU. See
{doc}`quickstart` to train and use a model.

## Hardware support

On NVIDIA GPUs, FlashCart can evaluate tensor products, Cartesian tensor
expansions, and equivariant linear maps with fused Triton kernels. CPUs and
Apple Silicon (`device=mps`) use PyTorch implementations of these operations.
Setting `use_triton=False` also selects the PyTorch implementations on CUDA
devices.

Both implementations evaluate the same mathematical expressions. Their
results can differ because of floating-point rounding.

The first import can take longer while FlashCart generates and caches kernel
source files. GPU compilation and autotuning can add further overhead on the
first evaluation. See {ref}`kernel-generation-compilation` for details.

## Optional components

Running FlashCart in LAMMPS requires a LAMMPS build with Python-enabled ML-IAP.
GPU simulations also require Kokkos and CuPy, as described in {ref}`lammps-build`.

For container-based use, the repository provides a `Dockerfile` that builds an
image containing CUDA 12.8, PyTorch, and FlashCart.

## Development and tests

From the repository root, install the development dependencies:

```bash
python -m pip install -e ".[dev]"
```

These include the testing and packaging tools. To also install the tools needed
to build the documentation, use `python -m pip install -e ".[dev,docs]"`.

Run the tests with:

```bash
python -m pytest
python -m pytest -m slow
```

The first command excludes tests marked `slow`. The second runs only those tests,
which cover the command-line tools, multi-process training, and compilation.
CUDA-dependent tests are skipped when CUDA is unavailable. Use
`python -m pytest -m ""` to run both groups together.
