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
from cases_dict import get_cases


BASE_DIR = Path(__file__).resolve().parent
copied_results_folder = BASE_DIR / "copy_results"

cases = get_cases(copied_results_folder)

prefixes = ['a', 'b', 'c', 'd', 'e']
# prefixes = ['g']
indices = [8]
# indices = [6]

type = "elastic"

a = 0
b = 0

for i in indices:
    if i == 9:
        b = 0.5
    if i == 10:
        b = 1
    a = 0.015
    for prefix in prefixes:

        data = np.load(copied_results_folder / f"results_{type}_{prefix}_full_{i}_copy.npz")

        d = dict(data)

        old_day_success = d["day_success"]
        if old_day_success.shape[0] == 2766:
            new_day_success = np.empty((old_day_success.shape[0] - 252,
            old_day_success.shape[1]))

            new_day_success = old_day_success[128:-124,:]

            d["day_success"] = new_day_success
        # old = d["parameters"]

        # new_dtype = np.dtype([
        #     ('total_turnover', '<f8'),
        #     ('hard_turnover', '<f8'),
        #     ('max_weights', '<f8'),
        #     ('individual_turnover', '<f8'),
        #     ('alpha', '<f8'),
        #     ('lambda_a', '<f8'),
        #     ('lambda_b', '<f8'),
        # ])

        # new_arr = np.empty(old.shape, dtype=new_dtype)

        # new_arr['total_turnover'] = old['total_turnover']
        # new_arr['hard_turnover'] = old['hard_turnover']
        # new_arr['max_weights'] = old['max_weights']
        # new_arr['individual_turnover'] = old['individual_turnover']
        # new_arr['alpha'] = old['alpha']
        # new_arr['lambda_a'] = a
        # new_arr['lambda_b'] = b

        # d["parameters"] = new_arr

        np.savez_compressed(copied_results_folder / f"results_{type}_{prefix}_full_{i}_copy.npz", **d)
        a += 0.01
