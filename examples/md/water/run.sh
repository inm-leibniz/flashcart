#!/usr/bin/env bash
# Water NVE + NPT campaign for one trained model.
#
#   CUDA_VISIBLE_DEVICES=0 bash run.sh ../../../models/my_run
#
# Needs `lmp` built with ML-IAP (Python) and Kokkos on PATH. See the LAMMPS
# page of the documentation. Each run takes one GPU.
set -euo pipefail

RUN_DIR=$(realpath "$1")               # training output_path
cd "$(dirname "$0")"
OUT=results/$(basename "$RUN_DIR")
LMP="lmp -k on g 1 -sf kk -pk kokkos newton on neigh half"

[ -f water_256.data ] || python build_box.py
[ -f "$RUN_DIR/flashcart-mliap.pt" ] || flashcart-lammps "$RUN_DIR" --compile-mode default
mkdir -p "$OUT"

$LMP -in in.nve -var model "$RUN_DIR/flashcart-mliap.pt" -var out "$OUT" -log "$OUT/log.nve"
for T in 273.15 293.15 313.15 333.15 353.15 373.15; do
    $LMP -in in.npt -var model "$RUN_DIR/flashcart-mliap.pt" -var out "$OUT" -var temp "$T" -log "$OUT/log.npt_${T}K"
done

python analyze.py "$OUT"
