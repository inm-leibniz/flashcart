# Multi-GPU training

FlashCart distributes training batches across GPUs using
[Lightning Fabric](https://lightning.ai/docs/fabric/stable/). Each GPU runs a
separate process with a copy of the model. The processes combine their gradients
before updating the parameters.

To train on four GPUs on one machine, select the visible devices and set
`devices=4`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 flashcart-train my_run.yaml devices=4
```

Replace the device indices with those available for your run. The `devices`
setting gives the number of GPUs to use on each node. Each process has a unique
index, called its rank.

(distributed-batching)=
## Batch sizes and data distribution

`train_batch_size` and `eval_batch_size` specify targets across all GPUs.
FlashCart divides each target by the number of processes and rounds up.
With fixed-size batching, a target of 192 structures on four GPUs gives
48 structures per process. Rounding can make the combined batch larger
than the requested target.

Dynamic batching is enabled by default. It uses the target for each process
to calculate limits on the number of atoms and neighbor pairs in a batch.
The number of structures can therefore vary between batches. Changing the
number of GPUs can change the batches and training results, even when the
global target is unchanged.

A structure that exceeds either limit is processed alone. These limits do not
split structures or guarantee a maximum memory use.

Distributed training balances the number of batches across ranks so that they
perform the same number of optimization steps. With dynamic batching and
`drop_last: false`, ranks with fewer batches repeat their final batch. With
`drop_last: true`, ranks with more batches discard the extra batches. Fixed-size
training uses PyTorch's distributed sampler, which can also repeat or discard
samples to balance ranks.

Validation and test loaders assign each structure to one rank without this
balancing. Rank batch counts may therefore differ. Metric statistics are combined
after iteration.

## Shared state and output

- Dataset statistics, including energy shifts, force scales, and the average neighbor
  count, are computed on rank 0 and broadcast to the other ranks.
- Rank 0 writes checkpoints, metrics, and logs. Use a shared filesystem for
  `output_path` when multiple machines need access to the run.
- `flashcart-test` also accepts `devices=N` for distributed evaluation.

## Multiple nodes

Every process must be able to read the datasets and any checkpoint being loaded.
Use paths that refer to the same files on every node. Make the run directory
available on all nodes so each process can load checkpoints and resume training.

For training across several nodes, launch the processes using Fabric or a supported
cluster launcher. Set `n_nodes` to the number of nodes assigned to your job and
`devices` to the number of devices to use on each node.
For a Slurm job, see Fabric's
[multi-node launch instructions](https://lightning.ai/docs/fabric/stable/guide/multi_node/slurm).
