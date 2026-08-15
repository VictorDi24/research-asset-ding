import functools
import math
import operator
import time

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


class portfolio:
    def __init__(
        self,
        individual_turnover: float = 0.005,
        max_weight: float = 0.008,
        total_turnover: float = 0.6,
        day: int = 0,
        speed: int = 0,
        alpha: float = 1,
        n_sectors: int = 15,
        eps: float = 1e-8,
        sector_abs_weight: float = 1e-3,
        number_gates: int = 2,
    ) -> None:

        # Parameters subject to change
        self.current_weights = cp.Parameter()
        self.target_weights = cp.Parameter()
        self.individual_turnover = cp.Parameter(nonneg=True, value=individual_turnover)
        self.max_weight = cp.Parameter(nonneg=True, value=max_weight)
        self.total_turnover = cp.Parameter(value=total_turnover)
        self.day = cp.Parameter(nonneg=True, value=day)
        self.speed = cp.Parameter(nonneg=True, value=speed)
        self.alpha = cp.Parameter(nonneg=True, value=alpha)
        self.n_sectors = cp.Parameter(nonneg=True, value=n_sectors)
        # No change parameters
        self.eps = cp.Parameter(nonneg=True, value=eps)
        self.sector_abs_weight = cp.Parameter(nonneg=True, value=sector_abs_weight)
        self.number_gates = cp.Parameter(nonneg=True, value=number_gates)

    def generate_weights(
        self,
    ):
        start = time.perf_counter()
        temp = np.load("/home/cding/IdeaProjects/cvxpy_optimize/short_arr_a.npz")

        # weights = temp["x"]
        on_off = temp["y"]
        delta = temp["z"]
        weights = delta[
            self.day.value : self.day.value + self.number_gates.value,
            :,
            self.speed.value,
        ]
        day_on_off = on_off[self.day.value + 1]
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
        self.current_weights = weights_one
        self.target_weights = weights_two
        return weights_one, weights_two, x, y, z

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
        lb_stocks[force_zero] = np.minimum(lb_stocks[force_zero], 0.0)
        ub_stocks[force_zero] = np.maximum(ub_stocks[force_zero], 0.0)

        return lb_stocks, ub_stocks, all_pos, all_neg, force_zero

    def create_sectors(self, weights_two):
        sector_weights = np.random.rand(self.n_sectors.value, weights_two.shape[0])
        column_sum = sector_weights.sum(axis=0, keepdims=True)
        sector_weights = sector_weights / column_sum

        return sector_weights

    def sector_constraint(self, sector_weights, x):
        sector_neutrality = [
            sector_weights @ x >= -self.sector_abs_weight,
            sector_weights @ x <= self.sector_abs_weight,
        ]
        return sector_neutrality

    def net_exposure_constraint(self, all_pos, x):
        long_sum = [
            all_pos @ x <= 1.0 + self.eps,
            all_pos @ x >= 1.0 - self.eps,
        ]
        return long_sum

    def force_sign_constraint(self, all_pos, all_neg, x):
        n_stocks = all_pos.shape[0]
        pos_idx = all_pos.astype(bool)
        positive_force = [x[pos_idx] >= -self.eps]
        neg_idx = all_neg.astype(bool)
        negative_force = [x[neg_idx] <= self.eps]

        return positive_force + negative_force

    def force_zero_constraint(self, force_zero, x):
        zero_idx = force_zero.astype(bool)
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
            diagonal @ x >= lb_stocks,
            diagonal @ x <= ub_stocks,
        ]
        return individual_weight

    def total_turnover_component_constraint(self, diagonal, weights_one, x, y):
        total_turnover_geq = [
            diagonal @ x + diagonal @ y >= weights_one,
        ]

        total_turnover_leq = [
            diagonal @ x - diagonal @ y <= weights_one,
        ]
        return total_turnover_geq + total_turnover_leq

    def v_constraint(self, ones, diagonal, y, z):
        sum_v = [ones @ y >= 0.0, ones @ y <= self.total_turnover + z[0]]
        positive_v = [
            diagonal @ y >= 0.0,
        ]
        return sum_v, positive_v

    def alpha_constraint(self, z):
        alpha_gate = [z[0] >= -self.eps]
        return alpha_gate

    def objective_a(self, x, weights_two):
        diff = x - weights_two
        return cp.sum_squares(diff)

    def objective(self, x, y, z, weights_one, weights_two):
        diff = x - weights_two
        return cp.sum_squares(diff) + self.alpha.value * (z[0]) ** 2

    def generate_constraints(
        self,
        weights_one,
        weights_two,
        x,
        y,
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
        total_turnover = self.total_turnover_component_constraint(
            diagonal, weights_one, x, y
        )
        sum_v, positive_v = self.v_constraint(ones, diagonal, y, z)
        alpha_cons = self.alpha_constraint(z)
        force_zero_condition = self.force_zero_constraint(force_zero, x)
        if force_zero[:n_stocks].any():
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
                force_zero_condition,
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

        return constraints, sector_weights

    def generate_opt(
        self,
    ):
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

        return clarabel_opts

    def print_details(
        self, optimize_weights, buffers, weights_two, all_pos, all_neg, sector_w
    ):
        n_stocks = optimize_weights.shape[0]
        print("---------------- VERIFICATION: --------------")
        if optimize_weights is not None and buffers is not None:
            print(f"net exposure: {optimize_weights.value.sum()}")
            print(f"short sum: {np.minimum(optimize_weights.value, 0).sum()}")
            print(f"long sum: {np.maximum(optimize_weights.value, 0).sum()}")
            pos1 = (optimize_weights.value >= 0).sum()
            print(f"force positive: {pos1 == all_pos.sum()}")
            neg1 = (optimize_weights.value <= 0).sum()
            print(f"force negative: {neg1 == all_neg.sum()}")
            print(
                f"sector weights: {sector_w @ optimize_weights.value <= self.sector_abs_weight.value}"
            )
            print(f"positive v: {(buffers.value >= -self.eps.value).sum() == n_stocks}")
            print(f"slack total turnover cap?: {abs(buffers.value).sum()}")
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
            f"no buffer objective: {((optimize_weights.value - weights_two) ** 2).sum()}"
        )
        correlation = stats.pearsonr(optimize_weights.value, weights_two)
        print(f"correlation: {correlation.statistic}")

    def run_optimize(
        self,
    ):
        weights_one, weights_two, x, y, z = self.generate_weights()
        lb_stocks, ub_stocks, all_pos, all_neg, force_zero = self.generate_bounds_masks(
            weights_one, weights_two
        )
        constraints, sector_weights = self.generate_constraints(
            weights_one,
            weights_two,
            x,
            y,
            z,
            all_pos,
            all_neg,
            force_zero,
            lb_stocks,
            ub_stocks,
        )
        opts = self.generate_opt()
        problem = cp.Problem(
            cp.Minimize(self.objective(x, y, z, weights_one, weights_two)),
            constraints,
        )

        start_time = time.perf_counter()
        problem.solve(**opts)
        end_time = time.perf_counter()
        print(f"Solver time: {end_time - start_time:.4f} seconds")
        if problem.status == "optimal":
            optimize_weights = x
            buffers = y
            gate_a = z
        else:
            optimize_weights = None
            buffers = None
            gate_a = None

        print("Status:", problem.status)
        print("Optimal objective value:", problem.value)
        self.print_details(
            optimize_weights,
            buffers,
            weights_two,
            all_pos,
            all_neg,
            sector_weights,
        )
        return optimize_weights, buffers, gate_a


class elastic_net(portfolio):
    def __init__(self, lambda_a: float = 1.0, lambda_b: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        print("initialized elastic_net")
        self.lambda_a = cp.Parameter(nonneg=True, value=lambda_a)
        self.lambda_b = cp.Parameter(nonneg=True, value=lambda_b)

    def objective(self, x, y, z, weights_one, weights_two):
        l1_gate_constraint = self.lambda_a.value * cp.norm(x - weights_one, 1)
        l2_gate_constraint = self.lambda_b.value * cp.sum_squares(x - weights_one)

        objective = (
            super().objective(x, y, z, weights_one, weights_two)
            + l1_gate_constraint
            + l2_gate_constraint
        )
        return objective
        # return super().objective(x, y, z, weights_two)

    def run_optimize(
        self,
    ):

        optimize_weights, buffers, gate_a = super().run_optimize()
        print(f"Method: {__name__}")
        return optimize_weights, buffers, gate_a


# elastic = elastic_net(0, 1)
# x, y, z = elastic.run_elastic_optimize()
