#!/usr/bin/env bash
# Train one model, then evaluate it on its test set and on every test subset.
#
#   bash train_and_test.sh configs/i3c3f64l2_s0.yaml data/spice/test_subsets
#
# Writes <output_path>/test.log and <output_path>/test_subsets/<subset>.log.
set -euo pipefail

CONFIG=$1
SUBSET_DIR=${2:-}

# Disable TF32 for the float32 training protocol.
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

OUT=$(python -c "import sys; from flashcart.utils.config import load_config; print(load_config(sys.argv[1])['output_path'])" "$CONFIG")

flashcart-train "$CONFIG"
flashcart-test "$CONFIG"

if [ -n "$SUBSET_DIR" ]; then
    mkdir -p "$OUT/test_subsets"
    for subset in "$SUBSET_DIR"/*.extxyz; do
        name=$(basename "$subset" .extxyz)
        flashcart-test "$CONFIG" valid_path=null test_path="$subset" 2>&1 | tee "$OUT/test_subsets/$name.log"
    done
fi
