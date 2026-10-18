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

BASE_DIR = Path(__file__).resolve().parent
copied_results_folder = BASE_DIR / "copy_results"
results_folder = BASE_DIR / "results_folder"
coarse_prefixes = [
    'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r','s','t','u','v','w','x', 'y', 'z']
# coarse_indices = [4,5,6]
# fine_prefixes = ['a', 'b', 'c', 'd', 'e']
# fine_indices = [8,9,10]

arrays = []

port_type = "alpha"

# ---------------
# ALPHA TURNOVER
# --------------
def main():
    paths = []
    for prefix in coarse_prefixes:
        paths.append(results_folder / f"results_{port_type}_{prefix}_full.npz")
    if not paths:
        raise RuntimeError("No input files to process")

    with np.load(paths[0]) as f:
        labels = sorted(f.files)
        shapes = {k: f[k].shape for k in labels}
        dtypes = {k: f[k].dtype for k in labels}

    for p in paths:
        with np.load(p) as f:
            # if sorted(f.files) != labels:
            #     raise ValueError(f"{p.name}: keys differ from first file")
            for k in labels:
                if f[k].shape != shapes[k]:
                    raise ValueError(
                        f"{p.name}: {k} shape {f[k].shape} != {shapes[k]}"
                    )

    # Stack one label at a time to bound peak memory
    out_path = copied_results_folder / f"{port_type}_turnover_stacked_all.npz"
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


    # -----------------
    # ELASTIC NET AND BERHU
    # paths = []
    # for prefix in coarse_prefixes:
    #     for i in coarse_indices:
    #         paths.append(copied_results_folder / f"results_{port_type}_{prefix}_full_{i}_copy.npz")
    # for prefix in fine_prefixes:
    #     for i in fine_indices:
    #         paths.append(copied_results_folder / f"results_{port_type}_{prefix}_full_{i}_copy.npz")

    # if not paths:
    #     raise RuntimeError("No input files to process")

    # with np.load(paths[0]) as f:
    #     labels = sorted(f.files)
    #     shapes = {k: f[k].shape for k in labels}
    #     dtypes = {k: f[k].dtype for k in labels}

    # for p in paths:
    #     with np.load(p) as f:
    #         if sorted(f.files) != labels:
    #             raise ValueError(f"{p.name}: keys differ from first file")
    #         for k in labels:
    #             if f[k].shape != shapes[k]:
    #                 raise ValueError(
    #                     f"{p.name}: {k} shape {f[k].shape} != {shapes[k]}"
    #                 )

    # # Stack one label at a time to bound peak memory
    # out_path = copied_results_folder / f"{port_type}_stacked.npz"
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


if __name__ == "__main__":
    main()

# -------------------------
# results in kill ipython
# -------------------------
# for prefix in coarse_prefixes:
#     for i in coarse_indices:
#         arrays.append(np.load(copied_results_folder / f"results_{port_type}_{prefix}_full_{i}_copy.npz"))

# for prefix in fine_prefixes:
#     for i in fine_indices:
#         arrays.append(np.load(copied_results_folder / f"results_{port_type}_{prefix}_full_{i}_copy.npz"))


# labels = arrays[0].files

# stacked = {
#     label: np.stack([f[label] for f in arrays], axis=0)
#     for label in labels
# }

# np.savez_compressed(copied_results_folder / f"{port_type}_stacked.npz", **stacked)

# -------------------------
# legacy
# -------------------------
# for prefix in prefixes:
#     for i in indices:
#         save_filename = f"results_{type}_{prefix}_full_copy"

#         data = np.load(copied_results_folder / f"results_{type}_{prefix}_full_{i}.npz")

#         result = data["result"]
#         turnover = data["turnover"]
#         correlation = data["correlation"]
#         tracking_error = data["tracking_error"]
#         below_eps = data["below_eps"]
#         day_success = data["day_success"]
#         parameters = data["parameters"]
#         num_stocks = data["num_stocks"]
#         penalty_values = data["penalty_values"]
#         np.savez_compressed(copied_results_folder / save_filename, result=result, turnover=turnover, correlation=correlation, tracking_error=tracking_error,
#         day_success=day_success,
#         below_eps = below_eps,
#         parameters=parameters,
#         num_stocks=num_stocks,
#         penalty_values=penalty_values
