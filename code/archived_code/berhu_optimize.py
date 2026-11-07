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
        individual_turnover: float = 0.008,
        max_weight: float = 0.008,
        total_turnover: float = 0.8,
        day: int = 0,
        speed: int = 0,
        alpha: float = 1,
        n_sectors: int = 15,
        eps: float = 1e-8,
        sector_abs_weight: float = 1e-3,
        number_gates: int = 2,
    ) -> None:

        # Parameters subject to change
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

        # y = cp.Variable(weights_one.shape[0], name="buffers")
        z = cp.Variable(1, name="upper_total_gate")
        self.current_weights = cp.Parameter(weights_one.shape[0], value=weights_one)
        self.target_weights = cp.Parameter(weights_two.shape[0], value=weights_two)

        # s/t are BerHu-specific auxiliary variables. The base class has no
        # use for them, but returning a uniform 6-tuple here lets a single
        # run_optimize drive every subclass without branching or an
        # overridden solve routine in berhu_penalty.
        return weights_one, weights_two, x, z, None, None

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
        self.force_zero = cp.Parameter(force_zero.shape, value=force_zero)
        self.all_neg = cp.Parameter(all_neg.shape, value=all_neg)
        self.all_pos = cp.Parameter(all_pos.shape, value=all_pos)
        return lb_stocks, ub_stocks, self.all_pos, self.all_neg, self.force_zero

    def create_sectors(self, weights_two):
        sector_weights = np.random.rand(self.n_sectors.value, weights_two.shape[0])
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
        n_stocks = all_pos.value.shape[0]
        pos_idx = all_pos.value.astype(bool)
        positive_force = [x[pos_idx] >= -self.eps]
        neg_idx = all_neg.value.astype(bool)
        negative_force = [x[neg_idx] <= self.eps]

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

    def total_turnover_direct_constraint(self, x, z, weights_one):
        total_turnover_constraint = [
            cp.sum(cp.abs(x - weights_one)) <= self.total_turnover + z[0]
        ]
        return total_turnover_constraint

    def alpha_constraint(self, z):
        alpha_gate = [z[0] >= -self.eps]
        return alpha_gate

    def objective_a(self, x, weights_two):
        diff = x - weights_two
        return cp.sum_squares(diff)

    def objective(self, x, z, weights_one, weights_two, s=None, t=None):
        diff = x - weights_two
        return cp.sum_squares(diff) + self.alpha * (z[0]) ** 2

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
        s=None,
        t=None,
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
                total_turnover_cons,
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
                total_turnover_cons,
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
            print(
                f"total turnover: {abs(self.current_weights.value - optimize_weights.value).sum()}"
            )
        else:
            print("optimization failed")
        print("----------------- OPTIMIZATION DETAILS: --------------")
        print(f"n_stocks: {n_stocks}")
        print(f"n_sectors: {self.n_sectors.value}")
        print(f"max_weight: {self.max_weight.value}")
        print(f"total_turnover: {self.total_turnover.value}")
        print(f"sector_abs_weight: {self.sector_abs_weight.value}")
        print(f"eps: {self.eps.value}")
        print(f"alpha: {self.alpha.value}")
        print("----------------- POST RUN STATISTICS: --------------")
        print(
            f"no buffer objective: {((optimize_weights.value - weights_two) ** 2).sum()}"
        )
        correlation = stats.pearsonr(optimize_weights.value, weights_two)
        print(f"correlation: {correlation.statistic}")

    def run_optimize(
        self,
    ):
        weights_one, weights_two, x, z, s, t = self.generate_weights()
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
            s=s,
            t=t,
        )
        opts = self.generate_opt()
        problem = cp.Problem(
            cp.Minimize(self.objective(x, z, weights_one, weights_two, s=s, t=t)),
            constraints,
        )

        start_time = time.perf_counter()
        problem.solve(**opts)
        end_time = time.perf_counter()
        print(f"Solver time: {end_time - start_time:.4f} seconds")
        if problem.status == "optimal":
            optimize_weights = x
            gate_a = z
        else:
            print("optimization failed")
            breakpoint()
            optimize_weights = None
            gate_a = None

        print("Status:", problem.status)
        print("Optimal objective value:", problem.value)
        self.print_details(
            optimize_weights,
            weights_two,
            all_pos,
            all_neg,
            sector_weights,
        )
        print(f"Method: {self.__class__.__name__}")
        return optimize_weights, gate_a


class elastic_net(portfolio):
    def __init__(self, lambda_a: float = 0.005, lambda_b: float = 0.002, **kwargs):
        super().__init__(**kwargs)
        print("initialized elastic_net")
        self.lambda_a = cp.Parameter(nonneg=True, value=lambda_a)
        self.lambda_b = cp.Parameter(nonneg=True, value=lambda_b)

    def objective(self, x, z, weights_one, weights_two, s=None, t=None):
        l1_gate_penalty = self.lambda_a * cp.norm(x - weights_one, 1)
        l2_gate_penalty = self.lambda_b * cp.sum_squares(x - weights_one)

        objective = (
            super().objective(x, z, weights_one, weights_two)
            + l1_gate_penalty
            + l2_gate_penalty
        )
        return objective


class berhu_penalty(portfolio):
    def __init__(self, threshold: float = 0.000001, scale: float = 0.000001, **kwargs):
        super().__init__(**kwargs)
        print("initialized berhu")
        self.threshold = cp.Parameter(nonneg=True, value=threshold)
        self.scale = cp.Parameter(nonneg=True, value=scale)

    def generate_weights(self):
        # Base now returns a uniform 6-tuple with s/t as None placeholders;
        # discard those and swap in real BerHu auxiliary variables.
        weights_one, weights_two, x, z, _, _ = super().generate_weights()

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
        all_pos,
        all_neg,
        force_zero,
        lb_stocks,
        ub_stocks,
        s=None,
        t=None,
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
        # Chained comparisons (0 <= s <= threshold) don't work on cvxpy
        # expressions -- Python evaluates them as `(0 <= s) and (s <= thresh)`,
        # and cvxpy explicitly forbids truth-testing a Constraint. Each bound
        # needs to be its own inequality.=None
        bound_s = [s >= 0, s <= self.threshold]
        bound_t = [t >= 0]
        # `constraints` returned from super() is already a flat list of
        # Constraint objects (the base class did its own reduce). Simple
        # list concatenation is correct here -- appending bound_s/bound_t as
        # single list elements and re-reducing over operator.add fails,
        # since a bare Constraint doesn't support "+" against another one.
        constraints = constraints + bound_s + bound_t
        return constraints, sector_weights

    def berhu_function(self, s: cp.Variable, t: cp.Variable) -> cp.Expression:
        denominator = 1 / (2 * self.threshold)
        return self.scale * cp.sum(s + t + t**2 * denominator)

    def objective(self, x, z, weights_one, weights_two, s, t):
        original_obj = super().objective(x, z, weights_one, weights_two)
        return original_obj + self.berhu_function(s, t)
