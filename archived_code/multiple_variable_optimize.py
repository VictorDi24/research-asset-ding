import functools
import math
import operator
import time
import typing

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from scipy import stats


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__}: {end_time - start_time:.4f} seconds")
        return result

    return wrapper


# defines constants
@functools.lru_cache(maxsize=1)
def get_config():
    return {
        # "n_stocks": weights_one.shape[0],
        # "n_stocks": 2500,
        # "number_buffer": 2,
        "n_sectors": 15,
        # "abs_starting_weights": 0.012,
        "max_weight": 0.008,
        "total_turnover": 0.8,
        "individual_turnover": 0.005,
        # "target_std": 0.0005,
        "sector_abs_weight": 1e-3,
        "eps": 1e-8,
        "speed": 0,
        "day": 1,
        "alpha": 1,
        # "beta": 0.00,
        "number_gates": 2,
    }


def generate_weights(config):
    start = time.perf_counter()
    temp = np.load("/home/cding/IdeaProjects/cvxpy_optimize/short_arr_a.npz")

    # weights = temp["x"]
    on_off = temp["y"]
    delta = temp["z"]
    weights = delta[
        config["day"] : config["day"] + config["number_gates"], :, config["speed"]
    ]
    day_on_off = on_off[config["day"] + 1]
    day_weights = weights[:, day_on_off]
    day_weights[np.isnan(day_weights)] = 0

    day_weights /= abs(day_weights).sum(axis=1, keepdims=True)
    day_weights -= day_weights.mean(axis=1, keepdims=True)
    day_weights *= 2

    weights_one = day_weights[0, :]
    weights_two = day_weights[1, :]
    end = time.perf_counter()
    print(f"load time: {end - start:.4f} seconds")

    # x = cp.Variable(2 * weights_one.shape[0] + 1, name="x")
    x = cp.Variable(
        weights_one.shape[0],
        name="weights",
    )

    y = cp.Variable(weights_one.shape[0], name="buffers")
    z = cp.Variable(1, name="upper_total_gate")

    return weights_one, weights_two, x, y, z


def generate_bounds_masks(weights_one, weights_two, config):
    # Strict limits from individual_turnover and abs_weight
    lb_stocks = np.maximum(
        -config["max_weight"],
        weights_one - config["individual_turnover"],
    )
    ub_stocks = np.minimum(
        config["max_weight"],
        weights_one + config["individual_turnover"],
    )

    # Check if allows positive
    pos_mask = weights_two > 0
    adjust_pos = np.logical_and(ub_stocks >= 0, lb_stocks <= 0)
    force_pos = np.logical_and(pos_mask, adjust_pos)

    # Truncate lower bound for positive
    lb_stocks[force_pos] = 0.0

    # Positive optimized variables
    already_pos = np.logical_and(lb_stocks >= 0, ub_stocks >= 0)
    all_pos = np.logical_or(force_pos, already_pos)

    # Check if allows negative
    neg_mask = weights_two < 0
    adjust_neg = np.logical_and(ub_stocks >= 0, lb_stocks <= 0)
    force_neg = np.logical_and(neg_mask, adjust_neg)

    # Truncate upper bound for negative
    ub_stocks[force_neg] = 0.0

    # Negative optimized variables
    already_neg = np.logical_and(ub_stocks <= 0, lb_stocks <= 0)
    all_neg = np.logical_or(force_neg, already_neg)

    # Gross exposure x matrices
    force_zero = weights_two == 0
    lb_stocks[force_zero] = np.minimum(lb_stocks[force_zero], 0.0)
    ub_stocks[force_zero] = np.maximum(ub_stocks[force_zero], 0.0)

    return lb_stocks, ub_stocks, all_pos, all_neg, force_zero


def create_sectors(config, weights_two):
    sector_weights = np.random.rand(config["n_sectors"], weights_two.shape[0])
    column_sum = sector_weights.sum(axis=0, keepdims=True)
    sector_weights = sector_weights / column_sum

    return sector_weights


def sector_constraint(sector_weights, config, x):
    sector_neutrality = [
        sector_weights @ x >= -config["sector_abs_weight"],
        sector_weights @ x <= config["sector_abs_weight"],
    ]
    return sector_neutrality


def net_exposure_constraint(all_pos, config, x):
    long_sum = [
        all_pos @ x <= 1.0 + config["eps"],
        all_pos @ x >= 1.0 - config["eps"],
    ]
    return long_sum


def force_sign_constraint(all_pos, all_neg, config, x):
    n_stocks = all_pos.shape[0]
    pos_idx = all_pos.astype(bool)
    positive_force = [x[pos_idx] >= -config["eps"]]
    neg_idx = all_neg.astype(bool)
    negative_force = [x[neg_idx] <= config["eps"]]

    return positive_force + negative_force


def force_zero_constraint(force_zero_padded, config, x):

    zero_idx = force_zero_padded.astype(bool)
    zeros_case = [
        x[zero_idx] <= config["eps"],
        x[zero_idx] >= -config["eps"],
    ]
    return zeros_case


def dollar_neutral_constraint(ones, config, x):
    dollar_neutrality = [
        ones @ x <= config["eps"],
        ones @ x >= -config["eps"],
    ]
    return dollar_neutrality


def individual_constraint(diagonal, lb_stocks, ub_stocks, x):
    individual_weight = [
        diagonal @ x >= lb_stocks,
        diagonal @ x <= ub_stocks,
    ]
    return individual_weight


def total_turnover_component_constraint(diagonal, weights_one, x, y):
    total_turnover_geq = [
        diagonal @ x + diagonal @ y >= weights_one,
        diagonal @ x + diagonal @ y <= math.inf,
    ]

    total_turnover_leq = [
        diagonal @ x - diagonal @ y >= -math.inf,
        diagonal @ x - diagonal @ y <= weights_one,
    ]
    return total_turnover_geq + total_turnover_leq


def v_constraint(ones, diagonal, config, y, z):
    sum_v = [ones @ y >= 0.0, ones @ y <= config["total_turnover"] + z[0]]

    positive_v = [
        diagonal @ y >= 0.0,
        diagonal @ y <= math.inf,
    ]
    return sum_v, positive_v


def alpha_constraint(config, z):
    alpha_gate = [z[0] >= -config["eps"]]

    return alpha_gate


def objective_a(x, weights_two, config):
    diff = x - weights_two
    return cp.sum_squares(diff)


def objective_b(x, y, z, weights_two, config):
    diff = x - weights_two
    return cp.sum_squares(diff) + config["alpha"] * (z[0]) ** 2


def print_details(
    optimize_weights, buffers, weights_two, all_pos, all_neg, sector_w, config
):
    n_stocks = optimize_weights.shape[0]
    print("---------------- VERIFICATION: --------------")
    if optimize_weights is not None and buffers is not None:
        print(f"net exposure: {optimize_weights.sum()}")
        print(f"short sum: {np.minimum(optimize_weights, 0).sum()}")
        print(f"long sum: {np.maximum(optimize_weights, 0).sum()}")

        pos1 = (optimize_weights >= 0).sum()
        print(f"force positive: {pos1 == all_pos.sum()}")
        neg1 = (optimize_weights <= 0).sum()
        print(f"force negative: {neg1 == all_neg.sum()}")
        print(
            f"sector weights: {sector_w @ optimize_weights <= config['sector_abs_weight']}"
        )
        print(f"positive v: {(buffers >= -config['eps']).sum() == n_stocks}")
        print(
            f"over total turnover cap?: {abs(buffers).sum() > config['total_turnover']}"
        )
        # print(
        #     f"turnover: {optimize_weights[n_stocks : -config['number_gates']].sum() <= config['total_turnover'] + config['eps']}"
        # )
    else:
        print("optimization failed")

    print("----------------- OPTIMIZATION DETAILS: --------------")
    print(f"n_stocks: {n_stocks}")
    # print(f"abs_starting_weights: {config['abs_starting_weights']}")
    print(f"n_sectors: {config['n_sectors']}")
    print(f"max_weight: {config['max_weight']}")
    print(f"total_turnover: {config['total_turnover']}")
    # print(f"target_std: {config['target_std']}")
    print(f"sector_abs_weight: {config['sector_abs_weight']}")
    print(f"eps: {config['eps']}")
    print(f"alpha: {config['alpha']}")
    # print(f"beta: {config['beta']}")
    print("----------------- POST RUN STATISTICS: --------------")
    print(f"no buffer objective: {((optimize_weights - weights_two) ** 2).sum()}")
    correlation = stats.pearsonr(optimize_weights, weights_two)
    print(f"correlation: {correlation.statistic}")


def run_optimize():
    config = get_config()
    weights_one, weights_two, x, y, z = generate_weights(config)
    n_stocks = weights_one.shape[0]
    zeros = np.zeros(n_stocks)
    ones = np.ones(n_stocks)
    diagonal = sp.eye(n_stocks, format="csr")

    lb_stocks, ub_stocks, all_pos_padded, all_neg_padded, force_zero_padded = (
        generate_bounds_masks(
            weights_one,
            weights_two,
            config,
        )
    )

    # Constraints
    sector_weights = create_sectors(config, weights_two)
    sector_cons = sector_constraint(sector_weights, config, x)
    long_sum = net_exposure_constraint(all_pos_padded, config, x)
    force_sign = force_sign_constraint(all_pos_padded, all_neg_padded, config, x)
    dollar_neutral = dollar_neutral_constraint(ones, config, x)
    individual_weight = individual_constraint(diagonal, lb_stocks, ub_stocks, x)
    total_turnover = total_turnover_component_constraint(diagonal, weights_one, x, y)
    sum_v, positive_v = v_constraint(ones, diagonal, config, y, z)
    alpha_cons = alpha_constraint(config, z)
    force_zero = force_zero_constraint(force_zero_padded, config, x)

    if force_zero_padded[:n_stocks].any():
        constraints = [
            sector_cons,
            long_sum,
            force_sign,
            dollar_neutral,
            individual_weight,
            total_turnover,
            sum_v,
            positive_v,
            alpha_cons,
            force_zero,
        ]
        constraints = functools.reduce(operator.add, constraints)
    else:
        constraints = [
            sector_cons,
            long_sum,
            force_sign,
            dollar_neutral,
            individual_weight,
            total_turnover,
            sum_v,
            positive_v,
            alpha_cons,
        ]
        constraints = functools.reduce(operator.add, constraints)

    TOL = 1e-10
    MAX_ITER = 100000

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

    problem = cp.Problem(
        cp.Minimize(objective_b(x, y, z, weights_two, config)),
        constraints=constraints,
    )

    start_time = time.perf_counter()
    problem.solve(**clarabel_opts)
    # problem.solve(**piqp_opts)
    end_time = time.perf_counter()
    print(f"Solver time: {end_time - start_time:.4f} seconds")
    if problem.status == "optimal":
        optimize_weights = x.value
        buffers = y.value
        gate_a = z.value
    else:
        optimize_weights = None
        buffers = None
        gate_a = None

    print("Status:", problem.status)
    print("Optimal objective value:", problem.value)
    print_details(
        optimize_weights,
        buffers,
        weights_two,
        all_pos_padded,
        all_neg_padded,
        sector_weights,
        config,
    )
    return optimize_weights, buffers, gate_a


# x, y, z = run_optimize()
