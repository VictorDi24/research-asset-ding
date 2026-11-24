import argparse
import functools
from posix import mkdir
from typing import override
import h5py
import operator
import random
import time
import logging
import sys

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from bisect import bisect_right
from datetime import datetime
from scipy import stats
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
copied_results_folder = BASE_DIR / "npz_stacked"
results_folder = BASE_DIR / "npz_a"
coarse_prefixes = ['a','b','c','d','e','f','g']
coarse_indices = [6,7,8]
# fine_prefixes = ['a', 'b', 'c', 'd', 'e']
# fine_indices = [8,9,10]

arrays = []

port_type = "berhu"

#---------------
# Additional elastic runs
# ---------------


# ---------------
# ALPHA TURNOVER
# --------------
def main():
    # paths = []
    # for prefix in coarse_prefixes:
    #     paths.append(results_folder / f"results_{port_type}_{prefix}_full.npz")
    # if not paths:
    #     raise RuntimeError("No input files to process")

    # with np.load(paths[0]) as f:
    #     labels = sorted(f.files)
    #     shapes = {k: f[k].shape for k in labels}
    #     dtypes = {k: f[k].dtype for k in labels}

    # for p in paths:
    #     with np.load(p) as f:
    #         # if sorted(f.files) != labels:
    #         #     raise ValueError(f"{p.name}: keys differ from first file")
    #         for k in labels:
    #             if f[k].shape != shapes[k]:
    #                 raise ValueError(
    #                     f"{p.name}: {k} shape {f[k].shape} != {shapes[k]}"
    #                 )

    # out_path = copied_results_folder / f"{port_type}_turnover_stacked_all.npz"
    # stacked = {}
    # n = len(paths)

    # for label in labels:
    #     total_shape = (n, *shapes[label])
    #     buf = np.empty(total_shape, dtype=dtypes[label])
    #     for i, p in enumerate(paths):
    #         with np.load(p) as f:
    #             buf[i] = f[label]
    #     stacked[label] = buf
    #     print(f"stacked {label}: {buf.shape}")

    # np.savez_compressed(out_path, **stacked)
    # print(f"wrote {out_path}")


    # -----------------
    # ELASTIC NET AND BERHU
    # -----------------
    port_type = "berhu"

    indices = [6,7,8]
    prefixes = ['a','b','c','d','e','f','g']

    paths = []
    for prefix in prefixes:
        for i in indices:
            paths.append(results_folder / f"results_{port_type}_{prefix}_full_{i}.npz")

    if not paths:
        raise RuntimeError("No input files to process")

    with np.load(paths[0]) as f:
        labels = sorted(f.files)
        shapes = {k: f[k].shape for k in labels}
        dtypes = {k: f[k].dtype for k in labels}

    for p in paths:
        with np.load(p) as f:
            if sorted(f.files) != labels:
                raise ValueError(f"{p.name}: keys differ from first file")
            for k in labels:
                if f[k].shape != shapes[k]:
                    raise ValueError(
                        f"{p.name}: {k} shape {f[k].shape} != {shapes[k]}"
                    )

    # Stack one label at a time to bound peak memory
    out_path = copied_results_folder / f"{port_type}_stacked_additional.npz"
    stacked = {}
    n = len(paths)

    for label in labels:
        total_shape = (n, *shapes[label])
        buf = np.empty(total_shape, dtype=dtypes[label])
        for i, p in enumerate(paths):
            with np.load(p) as f:
                buf[i] = f[label]
        stacked[label] = buf
        print(f"stacked {label}: {buf.shape}")

    np.savez_compressed(out_path, **stacked)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
