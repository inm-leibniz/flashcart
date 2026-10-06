"""Generate configurations for the five SPICE models reported in the FlashCart paper.

    python examples/configs/make_configs.py --out configs

Generate the five SPICE models reported in the FlashCart paper, each with seeds
0, 1, and 2. This writes 15 files named <arch>_s<seed>.yaml. Train each with
``flashcart-train configs/<name>.yaml`` (or train_and_test.sh). Architecture
names read i<interactions>c<correlation>f<features>l<l_max_hidden>, with a
trailing r for the path-reduced product.
"""

import argparse
from pathlib import Path

from flashcart.utils.config import load_yaml, save_yaml

# (n_interactions, correlation, n_hidden_feats, l_max_hidden_feats, path_reduced_product, devices)
MODELS = [
    (2, 3, 20, 2, True, 1),  # FlashCart-69k
    (3, 5, 32, 2, False, 1),  # FlashCart-0.5M
    (3, 3, 64, 2, False, 1),  # FlashCart-1.1M
    (3, 5, 88, 2, False, 1),  # FlashCart-3.0M
    (4, 5, 104, 2, False, 2),  # FlashCart-5.6M
]
SEEDS = [0, 1, 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, default=Path(__file__).with_name("spice.yaml"))
    parser.add_argument("--out", type=Path, default=Path("configs"))
    parser.add_argument("--data", default="data/spice", help="directory written by prepare_spice.py")
    args = parser.parse_args()

    base = load_yaml(args.base)
    for i, c, f, l, reduced, devices in MODELS:
        arch = f"i{i}c{c}f{f}l{l}" + ("r" if reduced else "")
        for seed in SEEDS:
            name = f"{arch}_s{seed}"
            cfg = dict(base)
            cfg.update(
                train_path=f"{args.data}/train_{seed}.extxyz",
                valid_path=f"{args.data}/val_{seed}.extxyz",
                test_path=f"{args.data}/test.extxyz",
                output_path=f"models/{name}",
                data_seed=seed,
                model_seed=seed,
                n_interactions=i,
                correlation=c,
                n_hidden_feats=f,
                l_max_hidden_feats=l,
                path_reduced_product=reduced,
                devices=devices,
            )
            save_yaml(args.out / f"{name}.yaml", cfg)
    print(f"wrote {len(MODELS) * len(SEEDS)} configs to {args.out}")


if __name__ == "__main__":
    main()
