# Environment variables

Environment variables control the kernel cache, generated tensor ranks,
and some runtime settings. Set them before starting FlashCart. For
example, to store generated files in a local cache directory:

```bash
export FLASHCART_CACHE_DIR="$PWD/.flashcart-cache"
flashcart-train my_run.yaml
```

Boolean `FLASHCART_*` variables accept `1`, `true`, `yes`, or `t` to enable
an option, ignoring case. Other values disable it. Leaving a variable unset
uses its documented default.

## Kernel cache and runtime settings

| Variable | Default | Effect |
|---|---|---|
| `FLASHCART_KERNEL_L_MAX` | `3` | Maximum tensor rank included in generated kernels. Each backend reads it when imported. |
| `FLASHCART_CACHE_DIR` | `$XDG_CACHE_HOME/flashcart` or `~/.cache/flashcart` | Directory for generated kernel modules. |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | PyTorch CUDA memory-allocation settings. The command-line tools and LAMMPS interface supply this default unless the variable is already defined. |

`FLASHCART_CACHE_DIR` overrides the default cache location. Otherwise,
FlashCart uses `$XDG_CACHE_HOME/flashcart` when `XDG_CACHE_HOME` is set,
and `~/.cache/flashcart` when it is not.

`FLASHCART_KERNEL_L_MAX` sets the highest tensor rank included in the
generated modules. The model's `l_max_*` settings select the ranks used
during evaluation. Set the environment variable before importing FlashCart.
If FlashCart has already been imported, restart the Python session or notebook
kernel for the new value to take effect.

FlashCart reuses cached source files when the kernel-generation code and
maximum tensor rank are unchanged. Different versions can share a cache
directory. Missing files are generated automatically, so FlashCart needs write access
to the cache directory.

The cache contains generated source files. GPU compilation and autotuning
can still take place when the kernels are first evaluated.

## Kernel tuning

These options control how GPU kernels divide tensor-product calculations
into separate launches. They affect the Triton implementation. Set these variables
before importing FlashCart. If FlashCart has already been imported, restart
the Python session or notebook kernel for the new values to take effect.
See {ref}`profiling-command` for instructions on comparing settings on the
model and structures you intend to run.

A receiver is the atom that collects contributions from neighboring atoms.
The `csr` kernel groups incoming edges by receiver. The `edge` kernel
processes edges separately.

Group budgets limit the number of output components assigned to one launch
per feature channel. They are component counts, not memory sizes. All
components at a given tensor rank stay in the same group, so that group
can exceed the requested budget.

- `FLASHCART_TP_SCATTER_KERNEL` selects `csr` (default), which groups edges by
  receiver, or `edge`, which evaluates edges separately and accumulates their
  contributions with atomic additions.
- `FLASHCART_TP_CSR_GROUP_BUDGET` sets the output-component budget for a forward
  receiver-grouped launch. Default: `100`.
- `FLASHCART_TP_BWD_GROUP_BUDGET` sets the corresponding budget for the first
  derivative. Default: `100`.
- `FLASHCART_TP_DBWD_GROUP_BUDGET` sets the component budget for fused
  second-derivative launches in the `csr` kernel. When unset, the budget
  is half the smaller of `FLASHCART_TP_CSR_GROUP_BUDGET` and
  `FLASHCART_TP_BWD_GROUP_BUDGET`, rounded down. Separate derivative
  launches use the corresponding forward or backward budget.
- `FLASHCART_TP_BWD_CSR_MIN_DEG` sets the average number of incoming edges
  per receiver at which the `csr` backward calculation is split into
  several groups of edges. Default: `64`.
- `FLASHCART_TP_BWD_CSR_SLOTS` sets the number of these groups directly.
  When unset, the backend chooses the number from the average incoming
  edge count. Set it to `1` to use one group per receiver.
- `FLASHCART_TP_BWD_SPLIT` and `FLASHCART_TP_DBWD_SPLIT` select `fused` or `full`
  launch plans for the first and second derivatives. When unset, the backend
  chooses a plan from the tensor rank and kernel variant. `full` separates groups
  of derivative outputs into distinct launches.
- `FLASHCART_TP_PEAK_AWARE_GROUPS` allows eligible group budgets to increase to
  accommodate the largest active tensor rank. Default: `true`.

## LAMMPS

See {ref}`lammps-troubleshooting` in {doc}`lammps` for the environment variables
used by the interface.
