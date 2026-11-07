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
results_folder = BASE_DIR / "results_folder"

copied_results_folder = BASE_DIR / "copy_results"

prefixes = [
    'c', 'd'
]
indices = [1]
type = "alpha"

for prefix in prefixes:
    save_filename = f"results_{type}_{prefix}_full_copy"

    data = np.load(results_folder / f"results_{type}_{prefix}_full.npz")

    # CLIPPING
    result = data["result"][128:-124,:,:]
    turnover = data["turnover"][128:-124,:]
    correlation = data["correlation"][128:-124,:]
    tracking_error = data["tracking_error"][128:-124,:]
    # below_eps = data["below_eps"][128:-124,:]
    # day_success = data["day_success"][128:-124,:]
    # parameters = data["parameters"]
    # num_stocks = data["num_stocks"][128:-124,:]
    # penalty_values = data["penalty_values"][128:-124,:]

    # #NO CLIPPING
    # result = data["result"]
    # turnover = data["turnover"]
    # correlation = data["correlation"]
    # tracking_error = data["tracking_error"]
    # below_eps = data["below_eps"]
    # day_success = data["day_success"]
    # parameters = data["parameters"]
    # num_stocks = data["num_stocks"]
    # penalty_values = data["penalty_values"]
    np.savez_compressed(copied_results_folder / save_filename, result=result, turnover=turnover,
    correlation=correlation,
    tracking_error=tracking_error,
    # day_success=day_success,
    # below_eps = below_eps,
    # parameters=parameters,
    # num_stocks=num_stocks,
    # penalty_values=penalty_values
    )



# CLIP FOR BERHU/ELASTIC
# for prefix in prefixes:
#     for i in indices:
#         save_filename = f"results_{type}_{prefix}_full_copy"

#         data = np.load(results_folder / f"results_{type}_{prefix}_full.npz")

#         # CLIPPING
#         result = data["result"][128:-124,:,:]
#         turnover = data["turnover"][128:-124,:]
#         correlation = data["correlation"][128:-124,:]
#         tracking_error = data["tracking_error"][128:-124,:]
#         below_eps = data["below_eps"][128:-124,:]
#         day_success = data["day_success"][128:-124,:]
#         parameters = data["parameters"]
#         num_stocks = data["num_stocks"][128:-124,:]
#         penalty_values = data["penalty_values"][128:-124,:]

#         # #NO CLIPPING
#         # result = data["result"]
#         # turnover = data["turnover"]
#         # correlation = data["correlation"]
#         # tracking_error = data["tracking_error"]
#         # below_eps = data["below_eps"]
#         # day_success = data["day_success"]
#         # parameters = data["parameters"]
#         # num_stocks = data["num_stocks"]
#         # penalty_values = data["penalty_values"]
#         np.savez_compressed(copied_results_folder / save_filename, result=result, turnover=turnover, correlation=correlation, tracking_error=tracking_error,
#         day_success=day_success,
#         below_eps = below_eps,
#         parameters=parameters,
#         num_stocks=num_stocks,
#         penalty_values=penalty_values
#         )
