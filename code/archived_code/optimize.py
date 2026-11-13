import functools
import math
import time
from types import MappingProxyType

import cvxpy as cp
import numpy as np
import scipy.sparse as sp


# define constants
@functools.lru_cache(maxsize=1)
def get_config():
    return {
        "n_stocks": 2500,
        "n_sectors": 15,
        "abs_starting_weights": 0.0006,
        "max_weight": 0.002,
        "total_turnover": 0.5,
        "individual_turnover": 0.003,
        "target_std": 0.0005,
        "sector_abs_weight": 1e-3,
        "eps": 1e-8,
    }


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{end_time - start_time:.4f} seconds")
        return result

    return wrapper


x = cp.Variable(2 * get_config()["n_stocks"], name="x")


def generate_weights():
    config = get_config()

    current_weights = np.random.uniform(
        -config["abs_starting_weights"],
        config["abs_starting_weights"],
        config["n_stocks"],
    )
    current_weights -= current_weights.mean()
    current_weights /= abs(current_weights).sum()
    current_weights *= 2
    current_weights = np.hstack((current_weights, np.zeros(config["n_stocks"])))

    targ_w = current_weights[: config["n_stocks"]] + np.random.normal(
        0, config["target_std"], config["n_stocks"]
    )
    targ_w -= targ_w.mean()
    targ_w /= np.abs(targ_w).sum()
    targ_w *= 2
    target_weights = np.hstack((targ_w, np.zeros(config["n_stocks"])))

    return current_weights, target_weights


def generate_bounds_masks(current_weights, target_weights, config):
    # individual_turnover and abs_weight
    lb_stocks = np.maximum(
        -config["max_weight"],
        current_weights[: config["n_stocks"]] - config["individual_turnover"],
    )
    ub_stocks = np.minimum(
        config["max_weight"],
        current_weights[: config["n_stocks"]] + config["individual_turnover"],
    )

    # Check if allows positive
    pos_mask = target_weights[: config["n_stocks"]] >= 0
    adjust_pos = np.logical_and(ub_stocks >= 0, lb_stocks <= 0)
    force_pos = np.logical_and(pos_mask, adjust_pos)

    # Truncate lower bound
    lb_stocks[force_pos] = 0.0

    # Positive optimized variables
    already_pos = np.logical_and(lb_stocks >= 0, ub_stocks >= 0)
    all_pos = np.logical_or(force_pos, already_pos)

    # Check if allows negative
    neg_mask = target_weights[: config["n_stocks"]] <= 0
    adjust_neg = np.logical_and(ub_stocks >= 0, lb_stocks <= 0)
    force_neg = np.logical_and(neg_mask, adjust_neg)

    # Truncate upper bound
    ub_stocks[force_neg] = 0.0

    # Negative optimized variables
    already_neg = np.logical_and(ub_stocks <= 0, lb_stocks <= 0)
    all_neg = np.logical_or(force_neg, already_neg)

    # Gross exposure x matrices
    all_pos_padded = np.hstack(
        (all_pos, np.zeros(config["n_stocks"], dtype=bool))
    ).astype(float)
    all_neg_padded = np.hstack(
        (all_neg, np.zeros(config["n_stocks"], dtype=bool))
    ).astype(float)
    force_zero = target_weights[: config["n_stocks"]] == 0
    force_zero_padded = np.hstack(
        (force_zero, np.zeros(config["n_stocks"], dtype=bool))
    ).astype(float)

    return lb_stocks, ub_stocks, all_pos_padded, all_neg_padded, force_zero_padded


def create_sectors(config):
    sector_weights = np.random.rand(config["n_sectors"], config["n_stocks"])
    column_sum = sector_weights.sum(axis=0, keepdims=True)
    sector_weights = sector_weights / column_sum
    sector_weights = np.hstack(
        (sector_weights, np.zeros((config["n_sectors"], config["n_stocks"])))
    )
    return sector_weights


def build_constraints(
    current_weights,
    lb_stocks,
    ub_stocks,
    all_pos_padded,
    all_neg_padded,
    force_zero_padded,
    sector_weights,
    config,
):
    constraints = []

    zeros = np.zeros((config["n_stocks"]))
    ones = np.ones((config["n_stocks"]))
    diagonal_a = sp.eye(config["n_stocks"], format="csr")
    double_diagonal = sp.hstack((diagonal_a, diagonal_a), format="csr")
    diagonal_zeros = sp.hstack(
        (diagonal_a, np.zeros((config["n_stocks"], config["n_stocks"]))), format="csr"
    )
    zeros_diagonal = sp.hstack(
        (np.zeros((config["n_stocks"], config["n_stocks"])), diagonal_a), format="csr"
    )
    ones_zeros = sp.hstack((sp.coo_matrix(ones), sp.coo_matrix(zeros)), format="csr")
    zeros_ones = sp.hstack((sp.coo_matrix(zeros), sp.coo_matrix(ones)), format="csr")
    diagonal_negdiagonal = sp.hstack((diagonal_a, -diagonal_a), format="csr")

    # Zero case
    # zeros_case = [force_zero_padded @ x == config["eps"]]

    # Gross exposure
    long_sum = [all_pos_padded @ x == 1.0]

    # Force positives
    # positive_force = [np.diag(all_pos_padded) @ x >= config["eps"]]
    # Force negatives
    # negative_force = [np.diag(all_neg_padded) @ x <= config["eps"]]

    pos_idx = all_pos_padded.astype(bool)
    positive_force = [x[pos_idx] >= config["eps"]]
    neg_idx = all_neg_padded.astype(bool)
    negative_force = [x[neg_idx] <= config["eps"]]

    zero_idx = force_zero_padded.astype(bool)
    zeros_case = [x[zero_idx] == config["eps"]]

    # Dollar neutral
    dollar_neutral = [ones_zeros @ x == config["eps"]]

    # Individual weights
    individual_weight = [
        diagonal_zeros @ x >= lb_stocks,
        diagonal_zeros @ x <= ub_stocks,
    ]

    # Turnover
    total_turnover_geq = [
        double_diagonal @ x >= current_weights[: config["n_stocks"]],
        double_diagonal @ x <= math.inf,
    ]

    total_turnover_leq = [
        diagonal_negdiagonal @ x >= -math.inf,
        diagonal_negdiagonal @ x <= current_weights[: config["n_stocks"]],
    ]

    sum_v = [
        zeros_ones @ x >= 0.0,
        zeros_ones @ x <= config["total_turnover"],
    ]

    positive_v = [
        zeros_diagonal @ x >= 0.0,
        zeros_diagonal @ x <= math.inf,
    ]

    # Sector neutral
    sector_neutrality = [
        sector_weights @ x >= -config["sector_abs_weight"],
        sector_weights @ x <= config["sector_abs_weight"],
    ]
    constraints = sum(
        [
            dollar_neutral,
            long_sum,
            positive_force,
            negative_force,
            sector_neutrality,
            individual_weight,
            positive_v,
            sum_v,
            total_turnover_geq,
            total_turnover_leq,
        ],
        [],
    )

    if force_zero_padded[: config["n_stocks"]].any():
        constraints = sum(
            [
                dollar_neutral,
                long_sum,
                positive_force,
                negative_force,
                sector_neutrality,
                individual_weight,
                positive_v,
                sum_v,
                total_turnover_geq,
                total_turnover_leq,
                zeros_case,
            ],
            [],
        )

    return constraints


@timer
def run_optimize(x, target_weights, constraints, config):

    TOL = 1e-10
    MAX_ITER = 1000000

    # Clarabel Configuration
    clarabel_opts = {
        "solver": cp.CLARABEL,
        "tol_feas": TOL,  # Primal & Dual feasibility threshold
        "tol_gap_abs": TOL,  # Absolute duality gap
        "tol_gap_rel": TOL,  # Relative duality gap
        "max_iter": MAX_ITER,
        "verbose": False,  # Disable stdout logging to ensure raw speed evaluation
    }

    # PIQP Configuration
    piqp_opts = {
        "solver": cp.PIQP,
        "eps_abs": TOL,  # Absolute tolerance (Primal, Dual, and Gap)
        "eps_rel": 0.0,  # Set relative to 0.0 or 1e-9 to enforce absolute alignment
        "max_iter": MAX_ITER,
        "verbose": False,
    }

    def objective():
        diff = x[: config["n_stocks"]] - target_weights[: config["n_stocks"]]
        return cp.sum_squares(diff)

    # # start timer
    # start_time = time.perf_counter()

    problem = cp.Problem(cp.Minimize(objective()), constraints=constraints)

    problem.solve(**clarabel_opts)
    # problem.solve(**piqp_opts)

    if problem.status == "optimal":
        optimize_weights = x.value[: config["n_stocks"]]
    else:
        optimize_weights = None

    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time

    print("Status:", problem.status)
    print("Optimal value:", problem.value)
    # print(f"\nSeconds: {elapsed_time}")

    return optimize_weights, x.value


def verify_optimization(optimize_weights, all_pos, all_neg, result_x, sector_w, config):
    print("---------------- VERIFICATION: --------------")
    if optimize_weights is not None and result_x is not None:
        print(f"dollar neutral: {optimize_weights.sum()}")
        print(f"long sum: {np.minimum(optimize_weights, 0).sum()}")
        print(f"short sum: {np.maximum(optimize_weights, 0).sum()}")

        pos1 = (optimize_weights[: config["n_stocks"]] >= 0).sum()
        print(f"force positive: {pos1 == all_pos.sum()}")
        neg1 = (optimize_weights[: config["n_stocks"]] <= 0).sum()
        print(f"force negative: {neg1 == all_neg.sum()}")
        print(f"sector weights: {sector_w @ result_x <= config['sector_abs_weight']}")
        print(
            f"positive v: {(result_x[config['n_stocks'] :] >= 0).sum() == config['n_stocks']}"
        )
        print(
            f"sum v: {result_x[config['n_stocks'] :].sum() <= config['total_turnover']}"
        )
        print(
            f"turnover: {result_x[: config['n_stocks']].sum() <= config['total_turnover']}"
        )
    else:
        print("optimization failed")
    print("----------------- OPTIMIZATION DETAILS: --------------")
    print(f"n_stocks: {config['n_stocks']}")
    print(f"abs_starting_weights: {config['abs_starting_weights']}")
    print(f"n_sectors: {config['n_sectors']}")
    print(f"max_weight: {config['max_weight']}")
    print(f"total_turnover: {config['total_turnover']}")
    print(f"target_std: {config['target_std']}")
    print(f"sector_abs_weight: {config['sector_abs_weight']}")
    print(f"eps: {config['eps']}")


def run_optimization_pipeline():
    config = get_config()

    current_w, target_w = generate_weights()
    sector_w = create_sectors(config)  # Assume this exists

    lb, ub, all_pos, all_neg, force_zero = generate_bounds_masks(
        current_w, target_w, config
    )

    constraints = build_constraints(
        current_w, lb, ub, all_pos, all_neg, force_zero, sector_w, config
    )

    # breakpoint()

    optimize_weights, result_x = run_optimize(x, target_w, constraints, config)

    verify_optimization(optimize_weights, all_pos, all_neg, result_x, sector_w, config)

    # breakpoint()

    return optimize_weights, result_x


optimize_weights, result_x = run_optimization_pipeline()
