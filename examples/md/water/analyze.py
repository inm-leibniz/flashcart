"""Analyze energy conservation, density, and speed from the water simulations.

    python analyze.py results/<name>

Read log.nve and log.npt_<T>K from the run directory. Over the final
--analysis-ps of each trajectory, report energy drift in meV/atom/ns,
energy fluctuations in meV/atom, and density with a block-averaged standard
error. Simulation speed comes from the LAMMPS production-run summary.
"""

import argparse
import re
from pathlib import Path

import numpy as np

PERF_RE = re.compile(r"Performance: ([\d.eE+-]+) ns/day")
NVE_BLOCK, NPT_BLOCK = 2, 1


def read_log(path, block):
    """Thermo columns of one run block of a LAMMPS log and its ns/day (None if not reached)."""
    blocks = []  # [columns, rows, ns_per_day]
    for line in path.read_text().splitlines():
        parts = line.split()
        perf = PERF_RE.search(line)
        if parts and parts[0] == "Step":
            blocks.append([parts, [], None])
        elif blocks and perf:
            blocks[-1][2] = float(perf[1])
        elif blocks and len(parts) == len(blocks[-1][0]):
            try:
                blocks[-1][1].append([float(p) for p in parts])
            except ValueError:
                pass
    if len(blocks) <= block or len(blocks[block][1]) < 2:
        return None
    cols, rows, ns_per_day = blocks[block]
    table = np.array(rows)
    return {c: table[:, i] for i, c in enumerate(cols)}, ns_per_day


def block_sem(x, t, block_ps):
    """Standard error of the mean from block averages of length block_ps."""
    edges = np.arange(t[0], t[-1] + block_ps, block_ps)
    means = [x[(t >= a) & (t < b)].mean() for a, b in zip(edges[:-1], edges[1:]) if ((t >= a) & (t < b)).any()]
    return float(np.std(means, ddof=1) / np.sqrt(len(means))) if len(means) > 1 else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--n-atoms", type=int, default=768)
    parser.add_argument("--analysis-ps", type=float, default=1000.0, help="final window analyzed, ps")
    parser.add_argument("--block-ps", type=float, default=50.0, help="block length for standard errors, ps")
    args = parser.parse_args()

    log = args.run_dir / "log.nve"
    if log.exists() and (parsed := read_log(log, NVE_BLOCK)) is not None:
        thermo, ns_per_day = parsed
        t = thermo["Time"]
        w = t >= t[-1] - args.analysis_ps
        e = 1e3 * thermo["TotEng"][w] / args.n_atoms  # meV/atom
        slope = np.polyfit(t[w], e, 1)[0]  # meV/atom/ps
        print(f"NVE: drift {1e3 * slope:.4f} meV/atom/ns, std {e.std():.4f} meV/atom, "
              f"T {thermo['Temp'][w].mean():.1f} K, {ns_per_day} ns/day")

    for log in sorted(args.run_dir.glob("log.npt_*K")):
        parsed = read_log(log, NPT_BLOCK)
        if parsed is None:
            continue
        thermo, ns_per_day = parsed
        t = thermo["Time"]
        w = t >= t[-1] - args.analysis_ps
        rho = thermo["Density"][w]
        print(f"NPT {log.name[8:]}: density {rho.mean():.4f} ± {block_sem(rho, t[w], args.block_ps):.4f} g/cm^3, "
              f"T {thermo['Temp'][w].mean():.1f} K, {ns_per_day} ns/day")


if __name__ == "__main__":
    main()
