import argparse
import functools
from typing import override
import h5py
import operator
import random
import math
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
copy_results = BASE_DIR / "copy_results"
python_graphs = BASE_DIR / "python_graphs"
latex_stock = BASE_DIR / "latex_stock"
stacked_berhu = copy_results / "berhu_stacked.npz"
stacked_elastic = copy_results / "elastic_stacked.npz"
stacked_alpha = copy_results / "alpha_stacked.npz"
no_penalty = copy_results / "results_normal_a_full.npz"
arr = BASE_DIR / "arr.npz"

#
# NTS: Take ratio of tracking error, difference of correlation, difference of turnover
#
def alpha (metric : str = "turnover", speed : int = 0, baseline_eps : float = 1E-8):
    if metric == "turnover":
        type = "alpha"
        data = np.load(stacked_alpha)
        turnover = data["turnover"][:,:,speed].astype(float)

        raw_data = np.load(no_penalty)
        raw_turnover = raw_data["turnover"][:,speed].astype(float)

        turnover_diff = turnover - raw_turnover
        z_values = np.nanmean(turnover_diff, axis = 1)

        rows = []

        for i in range(len(z_values)):
            z_val = z_values[i].item()
            rows.append((z_val))

        # rows.sort(key=lambda r: (r[0], r[1]))

        with open(latex_stock / f"{type}_{speed}_turnover_table.txt", "w") as f:
            f.write("z\n")
            for z_val in rows:
                f.write(f"{z_val:.6f}\n")
        return turnover_diff

    if metric == "correlation":
        data = np.load(stacked_alpha)
        corr = data["correlation"][:,:,speed].astype(float)

        raw_data = np.load(no_penalty)
        raw_corr = raw_data["correlation"][:,speed].astype(float)

        corr_diff = corr - raw_corr
        z_values = np.nanmean(corr_diff, axis = 1)

        rows = []

        for i in range(len(z_values)):
            z_val = z_values[i].item()
            rows.append((z_val))

        with open(latex_stock / f"alpha_{speed}_corr_table.txt", "w") as f:
            f.write("z\n")
            for z_val in rows:
                f.write(f"{z_val:.6f}\n")

        return corr_diff

    if metric == "tracking":
        data = np.load(stacked_alpha)
        tracking = data["tracking_error"][:,:,speed].astype(float)

        raw_data = np.load(no_penalty)
        raw_tracking = raw_data["tracking_error"][:,speed].astype(float)
        raw_num_stocks = raw_data["num_stocks"][:,speed].astype(float)

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = tracking / raw_tracking

        valid_baseline = raw_tracking > baseline_eps

        ratio = np.where(valid_baseline, ratio, np.nan)

        z_values = np.nanmean(ratio, axis = 1)

        rows = []

        for i in range(len(z_values)):
            z_val = z_values[i].item()
            if not np.isnan(z_val):
                rows.append((z_val))

        # rows.sort(key=lambda r: (r[0], r[1]))

        with open(latex_stock / f"alpha_{speed}_tracking_table.txt", "w") as f:
            f.write("z\n")
            for z_val in rows:
                f.write(f"{z_val:.6f}\n")

        return ratio

def speed_turnover():
    data = np.load(arr)
    masks = data['y']
    masks = masks.squeeze()
    speed_days = data['z']
    weights = speed_days[masks]

    slow = weights[:,:,4]
    med = weights[:,:,2]
    fast = weights[:,:,0]

    slow_turnover = np.abs(np.diff(slow, axis=0)).sum(axis=1)
    med_turnover  = np.abs(np.diff(med,  axis=0)).sum(axis=1)
    fast_turnover = np.abs(np.diff(fast, axis=0)).sum(axis=1)

    days = np.arange(1, slow_turnover.shape[0] + 1)

    out = np.column_stack([days, slow_turnover, med_turnover, fast_turnover])
    np.savetxt(
        'speed_turnover.dat',
        out,
        fmt='%d %.10e %.10e %.10e',
        delimiter=' ',
        header='',
        comments=''
    )

    return slow_turnover, med_turnover, fast_turnover

def below_eps(speed: int = 0, type : str = "elastic"):
    if type == "berhu":
        data = np.load(stacked_berhu)
        eps = data["below_eps"][:,:,speed].astype(float)
        num_stocks = data["num_stocks"][:,:,speed].astype(float)
        param = data["parameters"]
    elif type == "elastic":
        data = np.load(stacked_elastic)
        eps = data["below_eps"][:,:,speed].astype(float)
        num_stocks = data["num_stocks"][:,:,speed].astype(float)
        param = data["parameters"]
    else:
        print("input doesn't match a type, using elastic instead")
        data = np.load(stacked_elastic)
        eps = data["below_eps"][:,:,speed].astype(float)
        num_stocks = data["num_stocks"][:,:,speed].astype(float)
        param = data["parameters"]

    raw_data = np.load(no_penalty)
    raw_eps = raw_data["below_eps"][:,speed].astype(float)
    raw_num_stocks = raw_data["num_stocks"][:,speed].astype(float)
    raw_param = raw_data["parameters"]

    if raw_eps.ndim == 3:
        raw_eps = raw_eps[0]
        raw_num_stocks = raw_num_stocks[0]

    frac = eps / np.where(num_stocks > 0, num_stocks, np.nan)
    raw_frac = raw_eps / np.where(raw_num_stocks > 0, raw_num_stocks, np.nan)

    eps_diff = frac - raw_frac
    z_values = np.nanmean(eps_diff, axis=1)

    rows = []

    if type == "berhu":
        for i in range(len(z_values)):
            scale_val = param[i]['scale'].item()
            thresh_val = param[i]['threshold'].item()
            z_val = z_values[i].item()
            rows.append((scale_val, thresh_val, z_val))

        rows.sort(key=lambda r: (r[0], r[1]))

        with open(python_graphs / f"berhu_{speed}_eps_table.txt", "w") as f:
            f.write("scale threshold z\n")
            for scale_val, thresh_val, z_val in rows:
                f.write(f"{scale_val:.6f} {thresh_val:.6f} {z_val:.6f}\n")
    else:
        for i in range(len(z_values)):
            lambda_a_val = param[i]['lambda_a'].item()
            lambda_b_val = param[i]['lambda_b'].item()
            z_val = z_values[i].item()
            rows.append((lambda_a_val, lambda_b_val, z_val))

        rows.sort(key=lambda r: (r[0], r[1]))

        with open(python_graphs / f"elastic_{speed}_eps_table.txt", "w") as f:
            f.write("lambda_a lambda_b z\n")
            for lambda_a_val, lambda_b_val, z_val in rows:
                f.write(f"{lambda_a_val:.6f} {lambda_b_val:.6f} {z_val:.6f}\n")

    return eps_diff


def correlation(speed: int = 0, type : str = "elastic"):
    if type == "berhu":
        data = np.load(stacked_berhu)
        corr = data["correlation"][:,:,speed].astype(float)
        num_stocks = data["num_stocks"][:,:,speed].astype(float)
        param = data["parameters"]
    elif type == "elastic":
        data = np.load(stacked_elastic)
        corr = data["correlation"][:,:,speed].astype(float)
        num_stocks = data["num_stocks"][:,:,speed].astype(float)
        param = data["parameters"]
    else:
        print("input doesn't match a type, using elastic instead")
        data = np.load(stacked_elastic)
        corr = data["correlation"][:,:,speed].astype(float)
        num_stocks = data["num_stocks"][:,:,speed].astype(float)
        param = data["parameters"]

    raw_data = np.load(no_penalty)
    raw_corr = raw_data["correlation"][:,speed].astype(float)
    raw_num_stocks = raw_data["num_stocks"][:,speed].astype(float)
    raw_param = raw_data["parameters"]

    if raw_corr.ndim == 3:
        raw_corr = raw_corr[0]
        raw_num_stocks = raw_num_stocks[0]

    corr_diff = corr - raw_corr
    z_values = np.nanmean(corr_diff, axis=1)

    rows = []

    if type == "berhu":
        for i in range(len(z_values)):
            scale_val = param[i]['scale'].item()
            thresh_val = param[i]['threshold'].item()
            z_val = z_values[i].item()
            rows.append((scale_val, thresh_val, z_val))

        rows.sort(key=lambda r: (r[0], r[1]))

        with open(python_graphs / f"{type}_{speed}_corr_table.txt", "w") as f:
            f.write("scale threshold z\n")
            for scale_val, thresh_val, z_val in rows:
                f.write(f"{scale_val:.6f} {thresh_val:.6f} {z_val:.6f}\n")
    else:
        for i in range(len(z_values)):
            lambda_a_val = param[i]['lambda_a'].item()
            lambda_b_val = param[i]['lambda_b'].item()
            z_val = z_values[i].item()
            rows.append((lambda_a_val, lambda_b_val, z_val))

        rows.sort(key=lambda r: (r[0], r[1]))

        with open(python_graphs / f"{type}_{speed}_corr_table.txt", "w") as f:
            f.write("lambda_a lambda_b z\n")
            for lambda_a_val, lambda_b_val, z_val in rows:
                f.write(f"{lambda_a_val:.6f} {lambda_b_val:.6f} {z_val:.6f}\n")

    return corr_diff

def tracking_error(speed : int = 0, type: str = "elastic", baseline_eps: float = 1e-8):
    if type == "berhu":
        data = np.load(stacked_berhu)
        tracking = data["tracking_error"][:,:,speed].astype(float)
        num_stocks = data["num_stocks"][:,:,speed].astype(float)
        param = data["parameters"]
    else:
        if type != "elastic":
            print("input doesn't match a type, using elastic instead")
        data = np.load(stacked_elastic)
        tracking = data["tracking_error"][:,:,speed].astype(float)
        num_stocks = data["num_stocks"][:,:,speed].astype(float)
        param = data["parameters"]

    raw_data = np.load(no_penalty)
    raw_tracking = raw_data["tracking_error"][:,speed].astype(float)
    raw_num_stocks = raw_data["num_stocks"][:,speed].astype(float)

    if raw_tracking.ndim == 3:
        raw_tracking = raw_tracking[0]
        raw_num_stocks = raw_num_stocks[0]

    frac = tracking / np.where(num_stocks > 0, num_stocks, np.nan)
    raw_frac = raw_tracking / np.where(raw_num_stocks > 0, raw_num_stocks, np.nan)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = frac / raw_frac

    valid_baseline = raw_frac > baseline_eps

    ratio = np.where(valid_baseline, ratio, np.nan)

    z_values = np.nanmean(ratio, axis=1)

    rows = []

    if type == "berhu":
        for i in range(len(z_values)):
            scale_val = param[i]['scale'].item()
            thresh_val = param[i]['threshold'].item()
            z_val = z_values[i].item()
            if not np.isnan(z_val):
                rows.append((scale_val, thresh_val, z_val))

        rows.sort(key=lambda r: (r[0], r[1]))

        with open(python_graphs / f"berhu_{speed}_tracking_table.txt", "w") as f:
            f.write("scale threshold z\n")
            for scale_val, thresh_val, z_val in rows:
                f.write(f"{scale_val:.6f} {thresh_val:.6f} {z_val:.6f}\n")
    else:
        for i in range(len(z_values)):
            lambda_a_val = param[i]['lambda_a'].item()
            lambda_b_val = param[i]['lambda_b'].item()
            z_val = z_values[i].item()
            if not np.isnan(z_val):
                rows.append((lambda_a_val, lambda_b_val, z_val))

        rows.sort(key=lambda r: (r[0], r[1]))

        with open(python_graphs / f"elastic_{speed}_tracking_table.txt", "w") as f:
            f.write("lambda_a lambda_b z\n")
            for lambda_a_val, lambda_b_val, z_val in rows:
                f.write(f"{lambda_a_val:.6f} {lambda_b_val:.6f} {z_val:.6f}\n")

    return ratio
