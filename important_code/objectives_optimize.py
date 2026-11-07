import argparse
import functools
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


#warnings.filterwarnings("error", category=RuntimeWarning)

random.seed(21)
BASE_DIR = Path(__file__).resolve().parent
results_folder = BASE_DIR / "results_folder"
# results_folder.mkdir(parents=True, exist_ok=True)
log_folder = BASE_DIR / "log_folder"
# log_folder.mkdir(parents=True, exist_ok=True)


file_weights = BASE_DIR / "short_arr_a.npz"
file_weights_long = BASE_DIR / "arr.npz"
file_sectors = BASE_DIR / "sector.h5"
data_file_path = BASE_DIR / "optimized_portfolio_copy.npz"
all_attributes = BASE_DIR / "hrs1.h5"
no_gating_a = BASE_DIR / "no_gating_assets.npz"

#Logger setup
logger = logging.getLogger("Optimizer")
logger.setLevel(logging.DEBUG)
logger.propagate = False

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.terminator = ""
console_handler.setFormatter(
    logging.Formatter("\r[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
)
logger.addHandler(console_handler)

_case_file_handler: logging.FileHandler | None = None

#Retiried handler and log infrastructure
# File: normal, one line per entry
# file_handler = logging.FileHandler(BASE_DIR / "optimizer_debug.log", mode="w")
# file_handler.setLevel(logging.DEBUG)
# file_handler.setFormatter(
#     logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
# )
# logger.addHandler(file_handler)

def attach_case_log_file(case_name: str) -> Path:
    global _case_file_handler

    # Remove a previously attached case file handler, if any
    if _case_file_handler is not None:
        logger.removeHandler(_case_file_handler)
        try:
            _case_file_handler.close()
        except Exception:
            pass
        _case_file_handler = None

    log_path = log_folder / f"log_{case_name}.log"
    fh = logging.FileHandler(log_path, mode="w")   # one file per run, overwrite
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(fh)
    _case_file_handler = fh
    return log_path

#normalizing weights and applying masks --> creates data file with raw turnovers, sectors, etc
def create_data_file(save : bool = False, num_attributes: int = 30):
    data = np.load(file_weights_long)
    y = data["y"].astype(bool)
    weights = data["z"]
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    num_days, num_assets, num_speeds = weights.shape
    print(f"processing {num_days} days, {num_assets} assets, {num_speeds} speeds")

    y_3d = y[:, :, np.newaxis]
    counts = np.sum(y_3d, axis=1, keepdims=True)
    counts_safe = np.where(counts == 0, 1, counts)

    w_curr = np.where(y_3d, weights, 0.0)

    sum_curr = np.sum(np.abs(w_curr), axis=1, keepdims=True)
    norm_w_curr = w_curr / np.where(sum_curr == 0, 1.0, sum_curr)

    mean_curr = np.sum(norm_w_curr, axis=1, keepdims=True) / counts_safe
    final_w_curr = np.where(y_3d, (norm_w_curr - mean_curr) * 2.0, 0)

    weights_prev = np.zeros_like(weights)
    weights_prev[1:] = weights[:-1]

    w_prev = np.where(y_3d, weights_prev, 0.0)

    sum_prev = np.sum(np.abs(w_prev), axis=1, keepdims=True)
    norm_w_prev = w_prev / np.where(sum_prev == 0, 1.0, sum_prev)

    mean_prev = np.sum(norm_w_prev, axis=1, keepdims=True) / counts_safe
    final_w_prev = np.where(y_3d, (norm_w_prev - mean_prev) * 2.0, 0)

    result_dtype = np.dtype([
        ("maximum", "f8"),
        ("minimum", "f8"),
        ("long_exposure", "f8"),
        ("count", "i8"),
        ("total turnover", "f8"),
        ("attribute exposures", "f8", (200,5)),
        ("weight_eps", "f8"),
        ("max_ind_turnover", "f8"),
        ("standard_deviation", "f8"),
    ])
    result = np.empty((num_days, num_speeds), dtype=result_dtype)

    weight_eps = np.sum(np.where(y_3d, final_w_curr, np.inf) < 1E-8, axis=1)
    result["weight_eps"] = weight_eps

    std_per_row = np.nanstd(np.where(y_3d, final_w_curr, np.nan), axis=1)

    result['standard_deviation'] = std_per_row

    has_active = counts.squeeze(axis=1) > 0

    # Min/Max over active assets for the current day
    max_vals = np.where(y_3d, final_w_curr, -np.inf).max(axis=1)
    min_vals = np.where(y_3d, final_w_curr, np.inf).min(axis=1)

    result["maximum"] = np.where(has_active, max_vals, 0.0)
    result["minimum"] = np.where(has_active, min_vals, 0.0)
    result["long_exposure"] = np.maximum(final_w_curr, 0.0).sum(axis=1)
    result["count"] = counts.squeeze(axis=1)

    turnover = np.sum(np.abs(final_w_curr - final_w_prev), axis=1)
    turnover[0] = 0.0
    result["total turnover"] = turnover

    max_turnover = np.max(np.abs(final_w_curr - final_w_prev), axis = 1)
    result["max_ind_turnover"] = max_turnover

    with h5py.File(all_attributes, "r") as f:
        arr = weights[:len(sorted(f.keys()))]
        for k, (key, value, mask) in enumerate(zip(sorted(f.keys()), arr, y)):
            renewed_weights = value[mask]
            result["attribute exposures"][k] = f[key][1:201] @ renewed_weights

    if save:
        np.savez(data_file_path, result=result)

    return result


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__}: {end_time - start_time:.4f} seconds")
        return result

    return wrapper


class portfolio:
    def __init__(
        self,
        individual_turnover: float = 0.02,
        max_weight: float = 0.04,
        total_turnover: float = 0.4,
        hard_turnover: float = 0.3,
        day: int = 1,
        speed: int = 0,
        alpha: float = 1,
        n_sectors: int = 30,
        eps: float = 1e-8,
        sector_abs_weight: float = 1e-3,
        number_gates: int = 2,
        solver: str = "clarabel",
        printouts: bool = True,
        pass_file: bool = False,
        assets_on: np.ndarray = None,
        assets: np.ndarray = None,
        alpha_on: bool = False,
        mask: np.ndarray = None,
        clip_threshold: float = None,
    ) -> None:

        # Parameters subject to change

        self.individual_turnover = cp.Parameter(nonneg=True, value=individual_turnover)
        self.max_weight = cp.Parameter(nonneg=True, value=max_weight)
        self.total_turnover = cp.Parameter(value=total_turnover)
        self.hard_turnover = cp.Parameter(value=hard_turnover)
        self.day = cp.Parameter(nonneg=True, value=day)
        self.speed = cp.Parameter(nonneg=True, value=speed)
        self.alpha = cp.Parameter(nonneg=True, value=alpha)
        self.n_sectors = cp.Parameter(nonneg=True, value=n_sectors)
        # No change parameters
        self.eps = cp.Parameter(nonneg=True, value=eps)
        self.printouts = cp.Parameter(value=printouts)
        self.sector_abs_weight = cp.Parameter(nonneg=True, value=sector_abs_weight)
        self.solver = solver
        self.number_gates = cp.Parameter(nonneg=True, value=number_gates)
        self.pass_file = cp.Parameter(value=pass_file)
        self.assets_on = assets_on
        self.assets = assets
        self.alpha_on = cp.Parameter(value=alpha_on)
        self.correlation = cp.Parameter(value = None)
        self.tracking_error = cp.Parameter(value=None)
        self.penalty = cp.Parameter(value=None)
        self.below_eps = cp.Parameter(value=None)
        self.clip_threshold = cp.Parameter(value=clip_threshold)
        self.char_name = cp.Parameter(value=None)
        self.num_stocks = cp.Parameter(value=None)

    def generate_weights(
        self,
    ):
        start = time.perf_counter()
        if not self.pass_file.value:
            temp = np.load(file_weights_long)
            on_off = temp["y"]
            delta = temp["z"]
        else:
            on_off = self.assets_on
            delta = self.assets
        # weights = temp["x"]
        weights = delta[
            self.day.value - 1: self.day.value + self.number_gates.value - 1,
            :,
            self.speed.value,
        ]
        day_on_off = on_off[self.day.value].astype(bool)
        self.mask = day_on_off                          # plain numpy array, not cp.Parameter
        day_weights = weights[:, day_on_off]
        day_weights[np.isnan(day_weights)] = 0

        row_sums = abs(day_weights).sum(axis=1, keepdims=True)
        day_weights /= np.where(row_sums == 0, 1, row_sums)
        day_weights -= day_weights.mean(axis=1, keepdims=True)
        day_weights *= 2

        if not np.isfinite(day_weights).all():
            logger.warning(
                f"Non-finite weights day={self.day.value} speed={self.speed.value}"
            )
            return None, None, None, None


        weights_one = day_weights[0, :]
        weights_two = day_weights[1, :]
        end = time.perf_counter()
        if self.printouts.value == True:
            print(f"load time: {end - start:.4f} seconds")

        # x = cp.Variable(2 * weights_one.shape[0] + 1, name="x")
        x = cp.Variable(
            weights_one.shape[0],
            name="weights",
        )

        # y = cp.Variable(weights_one.shape[0], name="buffers")
        z = cp.Variable(1, name="upper_total_gate")
        self.current_weights = cp.Parameter(weights_one.shape[0], value=weights_one)
        self.target_weights = cp.Parameter(weights_two.shape[0], value=weights_two)
        # if self.clip_threshold.value is not None:
        #     lo, hi = np.percentile(weights_one, [self.clip_threshold.value, 100.0 - self.clip_threshold.value])
        #     np.clip(weights_one, lo, hi, out=weights_one)
        #     lo, hi = np.percentile(weights_two, [self.clip_threshold.value, 100.0 - self.clip_threshold.value])
        #     np.clip(weights_two, lo, hi, out=weights_two)
        num_stocks = weights_one.shape[0]
        self.num_stocks = cp.Parameter(value=num_stocks)

        return weights_one, weights_two, x, z

    def generate_bounds_masks(
        self,
        weights_one,
        weights_two,
    ):
        # Strict limits from individual_turnover and abs_weight
        lb_stocks = np.maximum(
            -self.max_weight.value,
            weights_one - self.individual_turnover.value,
        )
        ub_stocks = np.minimum(
            self.max_weight.value,
            weights_one + self.individual_turnover.value,
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
        lb_stocks[force_zero] = -self.eps.value
        ub_stocks[force_zero] = self.eps.value
        self.force_zero = cp.Parameter(force_zero.shape, value=force_zero)
        self.all_neg = cp.Parameter(all_neg.shape, value=all_neg)
        self.all_pos = cp.Parameter(all_pos.shape, value=all_pos)
        return lb_stocks, ub_stocks, self.all_pos, self.all_neg, self.force_zero

    def create_sectors(self, weights_two):
        with h5py.File(all_attributes, "r") as f:
            dataset_name = list(f.keys())[self.day.value]

            dataset = f[dataset_name]

            sector_weights = dataset[
                :self.n_sectors.value, : self.current_weights.shape[0]
            ]
            self.char_name = list(f.keys())[self.day.value]

        column_sum = sector_weights.sum(axis=0, keepdims=True)
        sector_weights = sector_weights / column_sum
        self.sector_weights_matrix = cp.Parameter(
            sector_weights.shape, value=sector_weights
        )

        return sector_weights

    def sector_constraint(self, sector_weights, x):
        sector_neutrality = [
            sector_weights @ x >= -self.sector_abs_weight,
            sector_weights @ x <= self.sector_abs_weight,
        ]
        return sector_neutrality

    def net_exposure_constraint(self, all_pos, x):
        long_sum = [
            all_pos.value @ x <= 1.0 + self.eps,
            all_pos.value @ x >= 1.0 - self.eps,
        ]
        return long_sum

    def force_sign_constraint(self, all_pos, all_neg, x):
        pos_idx = all_pos.value.astype(bool)
        positive_force = [x[pos_idx] >= 0.0]
        neg_idx = all_neg.value.astype(bool)
        negative_force = [x[neg_idx] <= 0.0]

        return positive_force + negative_force

    def force_zero_constraint(self, force_zero, x):
        zero_idx = force_zero.value.astype(bool)
        zeros_case = [
            x[zero_idx] <= self.eps,
            x[zero_idx] >= -self.eps,
        ]
        return zeros_case

    def dollar_neutral_constraint(self, ones, x):
        dollar_neutrality = [
            ones @ x <= self.eps,
            ones @ x >= -self.eps,
        ]
        return dollar_neutrality

    def individual_constraint(self, diagonal, lb_stocks, ub_stocks, x):
        individual_weight = [
            x >= lb_stocks,
            x <= ub_stocks,
        ]
        return individual_weight

    # def total_turnover_component_constraint(self, diagonal, weights_one, x, y):
    #     total_turnover_geq = [
    #         diagonal @ x + diagonal @ y >= weights_one,
    #     ]

    #     total_turnover_leq = [
    #         diagonal @ x - diagonal @ y <= weights_one,
    #     ]
    #     return total_turnover_geq + total_turnover_leq

    # def v_constraint(self, ones, diagonal, y, z):
    #     sum_v = [ones @ y >= 0.0, ones @ y <= self.total_turnover + z[0]]
    #     positive_v = [
    #         diagonal @ y >= 0.0,
    #     ]
    #     return sum_v, positive_v

    def total_turnover_direct_constraint(self, x, z, weights_one):
        if self.alpha_on.value:
            total_turnover_constraint = [
                cp.sum(cp.abs(x - weights_one)) <= self.total_turnover + z[0],
                self.total_turnover + z[0] <= self.total_turnover.value + self.hard_turnover.value
            ]
        else:
            total_turnover_constraint = [
                cp.sum(cp.abs(x - weights_one)) <= (self.total_turnover + self.hard_turnover)
            ]
        return total_turnover_constraint

    def alpha_constraint(self, z):
        alpha_gate = [z[0] >= -self.eps]
        return alpha_gate

    # def objective_a(self, x, weights_two):
    #     diff = x - weights_two
    #     return cp.sum_squares(diff)

    def objective(self, x, z, weights_one, weights_two):
        obj = cp.sum_squares(x - weights_two)
        if self.alpha_on.value:
            obj = obj + self.alpha * z[0] ** 2
        return obj

    def generate_constraints(
        self,
        weights_one,
        weights_two,
        x,
        z,
        all_pos,
        all_neg,
        force_zero,
        lb_stocks,
        ub_stocks,
    ):
        n_stocks = weights_one.shape[0]
        ones = np.ones(n_stocks)
        diagonal = sp.eye(n_stocks, format="csr")

        # Constraints
        sector_weights = self.create_sectors(weights_two)
        sector_cons = self.sector_constraint(sector_weights, x)
        long_sum = self.net_exposure_constraint(all_pos, x)
        force_sign = self.force_sign_constraint(all_pos, all_neg, x)
        dollar_neutral = self.dollar_neutral_constraint(ones, x)
        individual_weight = self.individual_constraint(
            diagonal, lb_stocks, ub_stocks, x
        )
        # total_turnover = self.total_turnover_component_constraint(
        #     diagonal, weights_one, x, y
        # )
        # sum_v, positive_v = self.v_constraint(ones, diagonal, y, z)
        total_turnover_cons = self.total_turnover_direct_constraint(x, z, weights_one)
        alpha_cons = self.alpha_constraint(z)
        force_zero_condition = self.force_zero_constraint(force_zero, x)
        if force_zero.value[:n_stocks].any():
            constraints = [
                sector_cons,
                long_sum,
                force_sign,
                dollar_neutral,
                individual_weight,
                # total_turnover,
                # sum_v,
                # positive_v,
                total_turnover_cons,
                force_zero_condition,
            ]
        else:
            constraints = [
                sector_cons,
                long_sum,
                force_sign,
                dollar_neutral,
                individual_weight,
                # total_turnover,
                # sum_v,
                # positive_v,
                total_turnover_cons,
            ]
        if self.alpha_on.value:
            constraints.append(alpha_cons)
        constraints = functools.reduce(operator.add, constraints)

        return constraints, sector_weights

    def generate_opt(self,solver):
        TOL = 1e-7
        MAX_ITER = 200

        # Clarabel Configuration
        clarabel_opts = {
            "solver": cp.CLARABEL,
            "tol_feas": TOL,  # Primal & Dual feasibility threshold
            "tol_gap_abs": TOL,  # Absolute duality gap
            "tol_gap_rel": TOL,  # Relative duality gap
            "max_iter": MAX_ITER,
            "verbose": False,  # Disable stdout logging to ensure raw speed evaluation
        }

        piqp_opts = {
            "solver": cp.PIQP,
            "eps_abs": TOL,  # Primal & Dual feasibility threshold
            "eps_duality_gap_abs": TOL,  # Absolute duality gap
            "eps_duality_gap_rel": TOL,  # Relative duality gap
            "max_iter": MAX_ITER,
            "verbose": False,  # Disable stdout logging to ensure raw speed evaluation
        }

        if solver == "piqp":
            return piqp_opts
        else:
            return clarabel_opts

    def print_details(self, optimize_weights, weights_two, all_pos, all_neg, sector_w):
        n_stocks = optimize_weights.shape[0]
        print("---------------- VERIFICATION: --------------")
        if optimize_weights is not None:
            print(f"net exposure: {optimize_weights.value.sum()}")
            print(f"short sum: {np.minimum(optimize_weights.value, 0).sum()}")
            print(f"long sum: {np.maximum(optimize_weights.value, 0).sum()}")
            pos1 = (optimize_weights.value > 0.0).sum()
            print(f"force positive: {pos1 == all_pos.value.sum()}")
            neg1 = (optimize_weights.value < 0.0).sum()
            print(f"force negative: {neg1 == all_neg.value.sum()}")
            print(
                f"sector weights: {sector_w @ optimize_weights.value <= self.sector_abs_weight.value + self.eps.value}"
            )
            # print(
            #     f"positive v: {(optimize_weights.value >= -self.eps.value).sum() == n_stocks}"
            # )
        # print(
        #     f"turnover: {optimize_weights[n_stocks : -config['number_gates']].sum() <= config['total_turnover'] + config['eps']}"
        # )
        else:
            print("optimization failed")
        print("----------------- OPTIMIZATION DETAILS: --------------")
        print(f"n_stocks: {n_stocks}")
        # print(f"abs_starting_weights: {config['abs_starting_weights']}")
        print(f"n_sectors: {self.n_sectors.value}")
        print(f"max_weight: {self.max_weight.value}")
        print(f"total_turnover: {self.total_turnover.value}")
        # print(f"target_std: {config['target_std']}")
        print(f"sector_abs_weight: {self.sector_abs_weight.value}")
        print(f"eps: {self.eps.value}")
        print(f"alpha: {self.alpha.value}")
        # print(f"beta: {config['beta']}")
        print("----------------- POST RUN STATISTICS: --------------")
        print(
            f"total turnover: {abs(self.current_weights.value - optimize_weights.value).sum()}"
        )
        print(
            f"no buffer objective: {((optimize_weights.value - weights_two) ** 2).sum()}"
        )
        print(
            f"below eps turnover #: {(abs(self.current_weights.value - optimize_weights.value) <= self.eps.value).sum()}"
        )

        print(f"Method: {self.__class__.__name__}")
        correlation = stats.pearsonr(optimize_weights.value, weights_two)
        print(f"correlation: {correlation.statistic}")

    def run_optimize(
        self,
    ):
        weights_one, weights_two, x, z = self.generate_weights()
        if weights_one is None:
            return None, None
        lb_stocks, ub_stocks, all_pos, all_neg, force_zero = self.generate_bounds_masks(
            weights_one, weights_two
        )
        constraints, sector_weights = self.generate_constraints(
            weights_one,
            weights_two,
            x,
            z,
            all_pos,
            all_neg,
            force_zero,
            lb_stocks,
            ub_stocks,
        )
        opts = self.generate_opt(self.solver)

        problem = cp.Problem(
            cp.Minimize(self.objective(x, z, weights_one, weights_two)),
            constraints,
        )
        start_time = time.perf_counter()
        try:
            problem.solve(**opts)
        except cp.error.SolverError as e:
            logger.warning(
                f"SolverError day={self.day.value} speed={self.speed.value}: {e}"
            )
            return None, None
        end_time = time.perf_counter()
        if self.printouts.value == True:
            print(f"Solver time: {end_time - start_time:.4f} seconds")
            if problem.status == "optimal" or problem.status == "optimal_inaccurate":
                optimize_weights = x
                gate_a = z
                correlation = stats.pearsonr(optimize_weights.value, weights_two)
                self.correlation.value = correlation.statistic
                self.tracking_error.value = ((optimize_weights.value - weights_two) ** 2).sum()
                self.below_eps.value = (abs(weights_one - optimize_weights.value) <= self.eps.value).sum()
                print("Status:", problem.status)
                print("Optimal objective value:", problem.value)
                self.print_details(
                    optimize_weights,
                    weights_two,
                    all_pos,
                    all_neg,
                    sector_weights,
                )
            else:
                print("optimization failed.")
                return None, None
        else:
            if problem.status == "optimal":
                optimize_weights = x
                gate_a = z
                correlation = stats.pearsonr(optimize_weights.value, weights_two)
                self.correlation.value = correlation.statistic
                self.tracking_error.value = ((optimize_weights.value - weights_two) ** 2).sum()
                self.penalty.value = problem.value - self.tracking_error.value
                self.below_eps.value = (abs(weights_one - optimize_weights.value) <= self.eps.value).sum()
            else:
                logger.debug(
                    f"day={self.day.value} speed={self.speed.value} solver_status={problem.status}"
                )
                return None, None
        return optimize_weights, gate_a


class elastic_net(portfolio):
    def __init__(self, lambda_a: float = 0.005, lambda_b: float = 0.002, **kwargs):
        super().__init__(**kwargs)
        print("initialized elastic_net")
        self.lambda_a = cp.Parameter(nonneg=True, value=lambda_a)
        self.lambda_b = cp.Parameter(nonneg=True, value=lambda_b)
        self.penalty = cp.Parameter(value=None)
    def objective(self, x, z, weights_one, weights_two):
        l1_gate_penalty = cp.norm(x - weights_one, 1)
        l2_gate_penalty = cp.sum_squares(x - weights_one)
        obj = super().objective(x, z, weights_one, weights_two)
        objective = (
            obj
            + self.lambda_a.value * (self.lambda_b.value * l1_gate_penalty + 0.5 * (1 - self.lambda_b.value) * l2_gate_penalty)
        )

        return objective
        # return super().objective(x, y, z, weights_two)

class berhu_penalty(portfolio):
    def __init__(self, threshold: float = 1e-6, scale: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.threshold = cp.Parameter(nonneg=True, value=threshold)
        self.scale = cp.Parameter(nonneg=True, value=scale)

    def generate_weights(self):
        weights_one, weights_two, x, z = super().generate_weights()

        s = cp.Variable(
            weights_one.shape[0],
            name="capped",
        )
        t = cp.Variable(
            weights_one.shape[0],
            name="cushion",
        )
        return weights_one, weights_two, x, z, s, t

    def generate_constraints(
        self,
        weights_one,
        weights_two,
        x,
        z,
        s,
        t,
        all_pos,
        all_neg,
        force_zero,
        lb_stocks,
        ub_stocks,
    ):
        constraints, sector_weights = super().generate_constraints(
            weights_one,
            weights_two,
            x,
            z,
            all_pos,
            all_neg,
            force_zero,
            lb_stocks,
            ub_stocks,
        )
        bound_s = [0 <= s, s <= self.threshold]
        bound_t = [0 <= t]
        s_t_turnover = [s + t >= cp.abs(x - weights_one)]
        constraints.extend(bound_s)
        constraints.extend(bound_t)
        constraints.extend(s_t_turnover)
        return constraints, sector_weights

    def berhu_function(self, s: cp.Variable, t: cp.Variable) -> cp.Expression:
        denominater = 1 / (2 * self.threshold.value)
        return self.scale * (cp.sum(s + t) + cp.sum_squares(t) * denominater)

    def objective(
        self,
        x,
        z,
        s,
        t,
        weights_one,
        weights_two,
    ):
        original_obj = super().objective(x, z, weights_one, weights_two)
        return original_obj + self.berhu_function(s, t)

    def print_details(self, optimize_weights, weights_two, all_pos, all_neg, sector_w):
        n_stocks = optimize_weights.shape[0]
        print("---------------- VERIFICATION: --------------")
        if optimize_weights is not None:
            print(f"net exposure: {optimize_weights.value.sum()}")
            print(f"short sum: {np.minimum(optimize_weights.value, 0).sum()}")
            print(f"long sum: {np.maximum(optimize_weights.value, 0).sum()}")
            pos1 = (optimize_weights.value > 0.0).sum()
            print(f"force positive: {pos1 == all_pos.value.sum()}")
            neg1 = (optimize_weights.value < 0.0).sum()
            print(f"force negative: {neg1 == all_neg.value.sum()}")
            print(
                f"sector weights: {sector_w @ optimize_weights.value <= self.sector_abs_weight.value + self.eps.value}"
            )
            # print(
            #     f"positive v: {(optimize_weights.value >= -self.eps.value).sum() == n_stocks}"
            # )
        # print(
        #     f"turnover: {optimize_weights[n_stocks : -config['number_gates']].sum() <= config['total_turnover'] + config['eps']}"
        # )
        else:
            print("optimization failed")
        print("----------------- OPTIMIZATION DETAILS: --------------")
        print(f"Method: {self.__class__.__name__}")
        print(f"n_stocks: {n_stocks}")
        # print(f"abs_starting_weights: {config['abs_starting_weights']}")
        print(f"n_sectors: {self.n_sectors.value}")
        print(f"max_weight: {self.max_weight.value}")
        print(f"total_turnover: {self.total_turnover.value}")
        print(f"threshold: {self.threshold.value}")
        print(f"scale: {self.scale.value}")
        # print(f"target_std: {config['target_std']}")
        print(f"sector_abs_weight: {self.sector_abs_weight.value}")
        print(f"eps: {self.eps.value}")
        print(f"alpha: {self.alpha.value}")
        # print(f"beta: {config['beta']}")
        print("----------------- POST RUN STATISTICS: --------------")
        print(
            f"total turnover: {abs(self.current_weights.value - optimize_weights.value).sum()}"
        )
        print(
            f"no buffer objective: {((optimize_weights.value - weights_two) ** 2).sum()}"
        )
        print(
            f"below eps turnover #: {(abs(self.current_weights.value - optimize_weights.value) <= self.eps.value).sum()}"
        )

        correlation = stats.pearsonr(optimize_weights.value, weights_two)
        print(f"correlation: {correlation.statistic}")

    def run_optimize(self):
        weights_one, weights_two, x, z, s, t = self.generate_weights()
        lb_stocks, ub_stocks, self.all_pos, self.all_neg, self.force_zero = (
            self.generate_bounds_masks(weights_one, weights_two)
        )
        constraints, sector_weights = self.generate_constraints(
            weights_one,
            weights_two,
            x,
            z,
            s,
            t,
            self.all_pos,
            self.all_neg,
            self.force_zero,
            lb_stocks,
            ub_stocks,
        )
        opts = self.generate_opt(self.solver)
        problem = cp.Problem(
            cp.Minimize(self.objective(x, z, s, t, weights_one, weights_two)),
            constraints,
        )

        start_time = time.perf_counter()
        try:
            problem.solve(**opts)
        except cp.error.SolverError as e:
            logger.warning(
                f"SolverError day={self.day.value} speed={self.speed.value}: {e}"
            )
            return None, None
        end_time = time.perf_counter()
        if self.printouts.value == True:
            print(f"Solver time: {end_time - start_time:.4f} seconds")
            if problem.status == "optimal":
                optimize_weights = x
                gate_a = z
                correlation = stats.pearsonr(optimize_weights.value, weights_two)
                self.correlation.value = correlation.statistic
                self.tracking_error.value = ((optimize_weights.value - weights_two) ** 2).sum()
                self.penalty.value = problem.value - self.tracking_error.value
                self.below_eps.value = (abs(weights_one - optimize_weights.value) <= self.eps.value).sum()

                print("Status:", problem.status)
                print("Optimal objective value:", problem.value)
                self.print_details(
                    optimize_weights,
                    weights_two,
                    self.all_pos,
                    self.all_neg,
                    sector_weights,
                )
                print(
                    f"berhu penalty: {self.scale.value * np.sum(s.value + t.value + t.value**2 / (2 * self.threshold.value))}"
                )
                print("Optimal objective value:", problem.value)
                print(f"Solver: {self.solver}")
                self.s = s.value
                self.t = t.value
            else:
                print("optimization failed.")
                return None, None

        else:
            if problem.status == "optimal":
                optimize_weights = x
                gate_a = z
                correlation = stats.pearsonr(optimize_weights.value, weights_two)
                self.correlation.value = correlation.statistic
                self.tracking_error.value = ((optimize_weights.value - weights_two) ** 2).sum()
                self.penalty.value = problem.value - self.tracking_error.value
                self.below_eps.value = (abs(weights_one - optimize_weights.value) <= self.eps.value).sum()
            else:
                logger.debug(
                    f"day={self.day.value} speed={self.speed.value} solver_status={problem.status}"
                )
                return None, None
        return optimize_weights, gate_a


class huber_penalty(portfolio):
    def __init__(self, threshold: float = 1e-4, scale: float = 100, **kwargs):
        super().__init__(**kwargs)
        print("initialized huber_penalty")

        self.threshold = cp.Parameter(nonneg=True, value=threshold)
        self.scale = cp.Parameter(nonneg=True, value=scale)

    def objective(self, x, z, weights_one, weights_two):
        turnover_diff = x - weights_two
        huber_penalty = self.scale * cp.sum(
            cp.huber(turnover_diff, M=self.threshold.value)
        )

        return huber_penalty

    def print_details(self, optimize_weights, weights_two, all_pos, all_neg, sector_w):
        super().print_details(optimize_weights, weights_two, all_pos, all_neg, sector_w)

        if optimize_weights is not None:
            diff = optimize_weights.value - weights_two
            M = self.threshold.value

            abs_diff = np.abs(diff)
            quad_part = np.where(abs_diff <= M, diff**2, 0)
            lin_part = np.where(abs_diff > M, 2 * M * abs_diff - M**2, 0)
            huber_val = self.scale.value * np.sum(quad_part + lin_part)

            print(f"huber penalty: {huber_val:.6f}")


class huber_berhu(huber_penalty, berhu_penalty):
    def __init__(
        self,
        threshold_huber: float = 1e-4,
        scale_huber: float = 100,
        threshold_berhu: float = 1e-6,
        scale_berhu: float = 1e-6,
        **kwargs,
    ):
        # Initialize the base portfolio class via MRO
        super().__init__(**kwargs)
        print("initialized huber_berhu")

        # Huber specific parameters
        self.threshold_huber = cp.Parameter(nonneg=True, value=threshold_huber)
        self.scale_huber = cp.Parameter(nonneg=True, value=scale_huber)

        # Berhu specific parameters
        self.threshold_berhu = cp.Parameter(nonneg=True, value=threshold_berhu)
        self.scale_berhu = cp.Parameter(nonneg=True, value=scale_berhu)

        # Override the shared names from parent classes to avoid confusion
        self.threshold = self.threshold_huber
        self.scale = self.scale_huber

    def objective(self, x, z, s, t, weights_one, weights_two):
        huber_obj = huber_penalty.objective(self, x, z, weights_one, weights_two)
        berhu_obj = berhu_penalty.objective(self, x, z, s, t, weights_one, weights_two)
        return huber_obj + berhu_obj

    def print_details(self, optimize_weights, weights_two, all_pos, all_neg, sector_w):
        portfolio.print_details(
            self, optimize_weights, weights_two, all_pos, all_neg, sector_w
        )

        if optimize_weights is not None:
            # Calculate and print Huber penalty
            # Note: Using weights_two to be consistent with the objective function definition
            diff = optimize_weights.value - weights_two
            M = self.threshold_huber.value
            abs_diff = np.abs(diff)
            quad_part = np.where(abs_diff <= M, diff**2, 0)
            lin_part = np.where(abs_diff > M, 2 * M * abs_diff - M**2, 0)
            huber_val = self.scale_huber.value * np.sum(quad_part + lin_part)
            print(f"huber penalty: {huber_val:.6f}")

            # Calculate and print Berhu penalty
            if hasattr(self, "s") and self.s is not None:
                berhu_val = self.scale_berhu.value * np.sum(
                    self.s + self.t + self.t**2 / (2 * self.threshold_berhu.value)
                )
                print(f"berhu penalty: {berhu_val:.6f}")

#-------------------------
# ITERATING --------------
#-------------------------

#----Finding specific day index----
# Converts key into date string
def _key_to_date(key: str) -> str:
    return key.split("_")[-1]

#imput format: year-month-day
def find_day_index(day: str | datetime):
    if isinstance(day, str):
        date_str = day.replace("-", "")
    else:
        date_str = day.strftime("%Y%m%d")

    with h5py.File(all_attributes, "r") as f:
        keys = sorted(f.keys())
        dates = [_key_to_date(k) for k in keys]
    pos = bisect_right(dates, date_str) - 1
    if pos <= 0:
        logger.warning(
            f"{date_str} predates earliest feasible day {keys[0]}; "
            f"using first feasible date"
        )
        pos = 1
    matched = keys[pos]
    if dates[pos] != date_str:
        # fall back to the previous available date
        logger.warning(
            f"No exact key for {date_str}; using next available date {matched}"
        )
        pos += 1
    matched = keys[pos]
    return pos, matched

#Defaults to 11 year runner
def loop_optimization(
    start_day: str = "2016-01-04",
    end_day: str = "2025-12-31",
    log_every: int = 100,
    portfolio_class=portfolio,
    portfolio_kwargs=None,
    save_filename: str | None = None,
):
    if portfolio_kwargs is None:
        portfolio_kwargs = {}
    portfolio_kwargs = dict(portfolio_kwargs)

    temp = np.load(file_weights_long)
    on_off = temp["y"]
    delta = temp["z"]

    begin = 1
    end = delta.shape[0]

    if start_day != "beginning":
        begin = find_day_index(start_day)[0]
    if end_day != "end":
        end = find_day_index(end_day)[0] + 1

    # begin = max(1, min(begin, delta.shape[0] - 1))
    # end   = max(1, min(end,   delta.shape[0] - 1))

    num_days = end - begin

    turnover_values = np.full((num_days, 3), np.nan)
    correlation_values = np.full((num_days, 3), np.nan)
    tracking_error_values = np.full((num_days, 3), np.nan)
    penalty_values = np.full((num_days, 3), np.nan)
    num_stocks = np.full((num_days, 3), np.nan)
    below_eps_values = np.full((num_days, 3), np.nan)

    # Inject required arguments for file passing to keep the cases dictionary clean
    portfolio_kwargs.setdefault("pass_file", True)
    portfolio_kwargs.setdefault("assets_on", on_off)
    portfolio_kwargs.setdefault("assets", delta)
    portfolio_kwargs.setdefault("printouts", False)
    portfolio_kwargs.setdefault("alpha_on", False)

    # Instantiate the specific portfolio class dynamically
    port = portfolio_class(**portfolio_kwargs)

    start_time = time.perf_counter()
    total_optimizations = (num_days) * 3
    optimizations_completed = 0

    raw_weights = np.zeros((num_days, on_off.shape[1], 3))
    day_success = np.empty((num_days, 3))
    for current_day in range(begin, end):
        for current_speed in range(3):
            port.speed.value = current_speed
            port.day.value = current_day
            iter_start = time.perf_counter()

            # run_optimize() correctly returns optimized_weights, gate_a for all subclasses
            optimized_weights, gate_a = port.run_optimize()
            iter_time = time.perf_counter() - iter_start

            if optimized_weights is not None:
                turnover_values[current_day - begin, current_speed] = np.sum(
                    np.abs(optimized_weights.value - port.current_weights.value)
                )
                correlation_values[current_day - begin, current_speed] = port.correlation.value
                tracking_error_values[current_day - begin, current_speed] = port.tracking_error.value
                penalty_values[current_day - begin, current_speed] = port.penalty.value
                num_stocks[current_day - begin, current_speed] = port.num_stocks.value
                below_eps_values[current_day - begin, current_speed] = port.below_eps.value
                raw_weights[current_day - begin, port.mask, current_speed] = optimized_weights.value
                day_success[current_day - begin, current_speed] = True

            else:
                raw_weights[current_day - begin, port.mask, current_speed] = np.nan
                day_success[current_day - begin, current_speed] = False
                logger.warning(f"FAILED day={current_day} speed={current_speed} time={iter_time:.4f}s key={port.char_name}")

            optimizations_completed += 1
            if optimizations_completed % log_every == 0 or optimizations_completed == total_optimizations:
                logger.info(f"Optimizations completed: {optimizations_completed} / {total_optimizations}")

    end_time = time.perf_counter()
    console_handler.terminator = "\n"
    print()
    logger.info(f"Optimization completed over {total_optimizations} portfolios. Total time: {end_time - start_time:.4f}s")
    console_handler.terminator = "\n"

    parameters_dtype = np.dtype([
        ("total_turnover", "f8"),
        ("hard_turnover", "f8"),
        ("max_weights", "f8"),
        ("individual_turnover", "f8"),
        ("alpha", "f8"),
        ("lambda_a", "f8"),
        ("lambda_b", "f8"),
        ("scale", "f8"),
        ("threshold", "f8"),
    ])

    parameters = np.zeros(1, dtype=parameters_dtype)

    common = (
        port.total_turnover.value,
        port.hard_turnover.value,
        port.max_weight.value,
        port.individual_turnover.value,
        port.alpha.value,
    )

    if portfolio_class == elastic_net:
        parameters[0] = common + (port.lambda_a.value, port.lambda_b.value, np.nan, np.nan)
    if portfolio_class == berhu_penalty:
        parameters[0] = common + (np.nan, np.nan, port.scale.value, port.threshold.value)
    else:
        parameters[0] = common + (np.nan, np.nan, np.nan, np.nan)


    if save_filename:
        np.savez_compressed(save_filename, result=raw_weights, turnover=turnover_values, correlation=correlation_values, tracking_error=tracking_error_values, day_success=day_success, below_eps = below_eps_values, parameters=parameters, num_stocks=num_stocks, penalty_values=penalty_values)
    return raw_weights, turnover_values, correlation_values, tracking_error_values, day_success, below_eps_values, parameters


def run_case(cases_dict: dict, case_name: str):

    if case_name not in cases_dict:
        raise ValueError(f"Case '{case_name}' not found. Available cases: {list(cases_dict.keys())}")

    logger.info(f"Starting {case_name}")

    func, kwargs = cases_dict[case_name]
    raw_weights, turnover_values, correlation_values, tracking_error_values, day_success, below_eps_values, parameters = func(**kwargs)
    logger.info(f"Finished {case_name}\n")
    return raw_weights, turnover_values, correlation_values, tracking_error_values, day_success, below_eps_values, parameters

def main():
    from cases_dict import get_cases
    cases = get_cases(results_folder)

    parser = argparse.ArgumentParser(
        description="Portfolio Optimization CLI",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Case Selection
    _ = parser.add_argument(
        "-c", "--case",
        type=str,
        help="Name of the case to run (e.g., 'case_a_base')."
    )

    # Utility Flags
    _ = parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List all available cases and exit."
    )

    # Execution Overrides
    # _ = parser.add_argument(
    #     "--num-days",
    #     type=int,
    #     default=None,
    #     help="Override the number of days to run (useful for quick tests)."
    # )
    _ = parser.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="Override logging frequency."
    )
    _ = parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Override the output save path (e.g., 'my_custom_run.npz')."
    )

    # Portfolio Parameter Overrides
    _ = parser.add_argument(
        "--turnover",
        type=float,
        default=None,
        help="Override 'total_turnover' inside portfolio_kwargs."
    )

    args = parser.parse_args()

    # --- Handle --list-cases ---
    if args.list_cases:
        print("\nAvailable Cases:")
        print("-" * 60)
        for name, (_, config) in cases.items():
            p_class = config.get("portfolio_class", portfolio).__name__
            days = config.get("num_days", 2735)
            print(f" • {name:<25} | Class: {p_class:<15} | Days: {days}")
        print("-" * 60)
        sys.exit(0)

    # --- Validate case argument ---
    if not args.case:
        parser.error("the following arguments are required: -c/--case (or use --list-cases)")

    if args.case not in cases:
        print(f"\n[ERROR] Case '{args.case}' not found.")
        print("Use --list-cases to see available options.")
        sys.exit(1)

    # --- Extract function and kwargs ---
    func, kwargs = cases[args.case]
    kwargs["save_filename"] = results_folder / f"results_{args.case}.npz"
    # Per-case log file
    log_path = attach_case_log_file(args.case)
    logger.info(f"Logging this run to {log_path}")

    # --- Apply CLI overrides ---
    if args.log_every is not None:
        kwargs["log_every"] = args.log_every
    if args.output is not None:
        # Ensure the directory exists if a custom path is provided
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs["save_filename"] = str(out_path)

    # Drill down into nested portfolio_kwargs if needed
    if args.turnover is not None:
        kwargs.setdefault("portfolio_kwargs", {})["total_turnover"] = args.turnover

    # ---- Execute ----
    logger.info(f"--- CLI Execution: Starting {args.case} ---")
    start_time = time.perf_counter()

    result, turnover_values, correlation_values, tracking_error_values, day_success, below_eps_values, parameters = func(**kwargs)

    end_time = time.perf_counter()
    logger.info(f"--- CLI Execution: Finished {args.case} in {end_time - start_time:.2f}s ---")

    # Quick terminal feedback
    nan_count = np.isnan(result[:,0,:]).sum()
    logger.info(f"Run complete. Shape: {result.shape} | Failed optimizations (NaNs): {nan_count}")
    return result, turnover_values, correlation_values, tracking_error_values, day_success, below_eps_values, parameters

if __name__ == "__main__":
    main()
