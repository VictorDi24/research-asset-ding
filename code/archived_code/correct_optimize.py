import functools
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
        data_path: str = "/home/cding/IdeaProjects/cvxpy_optimize/short_arr_a.npz",
    ) -> None:

        # data_path used to be hard-coded inside generate_weights(), which only
        # worked on one machine. It's now a constructor argument.
        self.data_path = data_path
        self._data_cache = None  # populated once, lazily, by _load_data()

        # Parameters subject to change
        #
        # NOTE: self.current_weights / self.target_weights (cp.Parameter()) were
        # removed. They were declared as *scalar* Parameters, then immediately
        # overwritten with plain numpy arrays in generate_weights() -- so they
        # were never actually usable as CVXPY Parameters (a scalar Parameter
        # can't hold a vector .value; try it and CVXPY raises
        # "Invalid dimensions ... for Parameter value"), and after being
        # overwritten they were never read anywhere else in the class. They
        # were dead weight. weights_one/weights_two are threaded through as
        # plain arrays/function arguments instead, same as the rest of the
        # class already does.
        self.individual_turnover = cp.Parameter(nonneg=True, value=individual_turnover)
        self.max_weight = cp.Parameter(nonneg=True, value=max_weight)
        self.total_turnover = cp.Parameter(nonneg=True, value=total_turnover)
        self.day = cp.Parameter(nonneg=True, value=day)
        self.speed = cp.Parameter(nonneg=True, value=speed)
        # alpha multiplies a squared term in the objective (alpha * z**2). For
        # CVXPY to accept "Parameter * convex_term" as convex it needs to know
        # the Parameter's sign, so this must be nonneg=True. The original code
        # declared it with unknown sign, which fails DCP -- that's why it fell
        # back to alpha.value (a bare python float) in the objective. That
        # workaround silently defeats the whole point of using a Parameter:
        # once alpha.value is baked into the expression as a constant, changing
        # it later and re-solving the *same* Problem object has zero effect.
        self.alpha = cp.Parameter(nonneg=True, value=alpha)
        self.n_sectors = cp.Parameter(nonneg=True, value=n_sectors)
        # No change parameters
        self.eps = cp.Parameter(nonneg=True, value=eps)
        self.sector_abs_weight = cp.Parameter(nonneg=True, value=sector_abs_weight)
        self.number_gates = cp.Parameter(nonneg=True, value=number_gates)

    def _load_data(self):
        """Load the .npz file once and cache it in memory. The old code called
        np.load() from inside generate_weights(), i.e. on every single solve --
        expensive disk I/O + parsing repeated for no reason when only day/speed
        change between calls."""
        if self._data_cache is None:
            self._data_cache = np.load(self.data_path)
        return self._data_cache

    def generate_weights(
        self,
    ):
        # weights_one/weights_two below are built from day_weights[0]/[1] --
        # that hard-codes exactly 2 gates (current, target). number_gates is
        # exposed as a configurable Parameter but nothing downstream
        # generalizes past 2, so guard it explicitly instead of failing with a
        # confusing shape error deep in generate_bounds_masks.
        assert int(self.number_gates.value) == 2, (
            "generate_weights() assumes number_gates == 2 (current vs. target "
            "gate) via day_weights[0]/day_weights[1]. Generalize this method "
            "before changing number_gates."
        )

        start = time.perf_counter()
        temp = self._load_data()

        on_off = temp["y"]
        delta = temp["z"]
        day = int(self.day.value)
        gates = int(self.number_gates.value)
        speed = int(self.speed.value)

        weights = delta[day : day + gates, :, speed]
        day_on_off = on_off[day + 1]
        day_weights = weights[:, day_on_off]
        day_weights[np.isnan(day_weights)] = 0

        day_weights /= np.abs(day_weights).sum(axis=1, keepdims=True)
        day_weights -= day_weights.mean(axis=1, keepdims=True)
        day_weights *= 2

        weights_one = day_weights[0, :]
        weights_two = day_weights[1, :]
        end = time.perf_counter()
        print(f"load time: {end - start:.4f} seconds")

        x = cp.Variable(
            weights_one.shape[0],
            name="weights",
        )
        y = cp.Variable(weights_one.shape[0], name="buffers")
        z = cp.Variable(1, name="upper_total_gate")

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

    def create_sectors(self, weights_two, sector_labels):
        """Build a sparse one-hot sector-membership matrix from real sector
        classifications.

        The previous implementation called np.random.rand(n_sectors, n_stocks)
        -- i.e. it fabricated a brand new *random* sector map on every call.
        The sector-neutrality constraints were therefore being checked against
        pure noise: not reproducible between runs, and structurally
        meaningless (a real sector map is close to one-hot -- each stock
        belongs to ~1 sector -- not a dense random loading across all 15).

        sector_labels: int array, shape (n_stocks,). sector_labels[i] is the
            sector id (0 .. n_sectors-1) of stock i. You need to supply your
            real classification here (e.g. GICS sector mapped to an int id).
        """
        n_stocks = weights_two.shape[0]
        sector_labels = np.asarray(sector_labels)
        if sector_labels.shape[0] != n_stocks:
            raise ValueError(
                f"sector_labels has length {sector_labels.shape[0]}, expected "
                f"{n_stocks} (one sector id per stock)."
            )
        n_sectors = int(self.n_sectors.value)
        if sector_labels.min() < 0 or sector_labels.max() >= n_sectors:
            raise ValueError(
                f"sector_labels must be in [0, {n_sectors}); got range "
                f"[{sector_labels.min()}, {sector_labels.max()}]."
            )

        rows = sector_labels
        cols = np.arange(n_stocks)
        data = np.ones(n_stocks)
        # Sparse one-hot membership -- also faster to canonicalize than the
        # old dense random matrix, in the same spirit as the sparse selector /
        # risk-term work already done elsewhere in this project.
        sector_weights = sp.csr_matrix(
            (data, (rows, cols)), shape=(n_sectors, n_stocks)
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
            all_pos @ x <= 1.0 + self.eps,
            all_pos @ x >= 1.0 - self.eps,
        ]
        return long_sum

    def force_sign_constraint(self, all_pos, all_neg, x):
        pos_idx = all_pos.astype(bool)
        positive_force = [x[pos_idx] >= -self.eps]
        neg_idx = all_neg.astype(bool)
        negative_force = [x[neg_idx] <= self.eps]

        return positive_force + negative_force

    def force_zero_constraint(self, force_zero_padded, x):
        zero_idx = force_zero_padded.astype(bool)
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

    def objective_b(self, x, y, z, weights_two):
        diff = x - weights_two
        # Using self.alpha (the Parameter) instead of self.alpha.value (a
        # bare float) -- see the comment on self.alpha in __init__.
        return cp.sum_squares(diff) + self.alpha * (z[0]) ** 2

    def generate_constraints(
        self,
        weights_one,
        weights_two,
        x,
        y,
        z,
        all_pos_padded,
        all_neg_padded,
        force_zero_padded,
        lb_stocks,
        ub_stocks,
        sector_labels,
    ):
        n_stocks = weights_one.shape[0]
        ones = np.ones(n_stocks)
        diagonal = sp.eye(n_stocks, format="csr")

        # Constraints
        sector_weights = self.create_sectors(weights_two, sector_labels)
        sector_cons = self.sector_constraint(sector_weights, x)
        long_sum = self.net_exposure_constraint(all_pos_padded, x)
        force_sign = self.force_sign_constraint(all_pos_padded, all_neg_padded, x)
        dollar_neutral = self.dollar_neutral_constraint(ones, x)
        individual_weight = self.individual_constraint(
            diagonal, lb_stocks, ub_stocks, x
        )
        total_turnover = self.total_turnover_component_constraint(
            diagonal, weights_one, x, y
        )
        sum_v, positive_v = self.v_constraint(ones, diagonal, y, z)
        alpha_cons = self.alpha_constraint(z)
        force_zero = self.force_zero_constraint(force_zero_padded, x)

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
        if force_zero_padded.any():
            constraints.append(force_zero)
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

        # PIQP Configuration
        piqp_opts = {
            "solver": cp.PIQP,
            "eps_abs": TOL,  # Absolute tolerance (Primal, Dual, and Gap)
            "eps_rel": 0.0,  # Set relative to 0.0 or 1e-9 to enforce absolute alignment
            "max_iter": MAX_ITER,
            "verbose": False,
        }

        return clarabel_opts, piqp_opts

    def print_details(
        self, optimize_weights, buffers, weights_two, all_pos, all_neg, sector_w
    ):
        print("---------------- VERIFICATION: --------------")
        if optimize_weights is not None and buffers is not None:
            n_stocks = optimize_weights.shape[0]
            print(f"net exposure: {optimize_weights.sum()}")
            print(f"short sum: {np.minimum(optimize_weights, 0).sum()}")
            print(f"long sum: {np.maximum(optimize_weights, 0).sum()}")
            pos1 = (optimize_weights >= 0).sum()
            print(f"force positive: {pos1 == all_pos.sum()}")
            neg1 = (optimize_weights <= 0).sum()
            print(f"force negative: {neg1 == all_neg.sum()}")
            sector_exposure = sector_w @ optimize_weights
            print(
                "sector weights: "
                f"{(np.abs(sector_exposure) <= self.sector_abs_weight.value).all()}"
            )
            print(f"positive v: {(buffers >= -self.eps.value).sum() == n_stocks}")
            print(
                f"over total turnover cap?: {abs(buffers).sum() > self.total_turnover.value}"
            )
            print("----------------- POST RUN STATISTICS: --------------")
            print(
                f"no buffer objective: {((optimize_weights - weights_two) ** 2).sum()}"
            )
            correlation = stats.pearsonr(optimize_weights, weights_two)
            print(f"correlation: {correlation.statistic}")
        else:
            # NOTE: the old code computed `n_stocks = optimize_weights.shape[0]`
            # unconditionally at the top of this function, before this None
            # check -- so a failed solve (optimize_weights = None) crashed with
            # AttributeError instead of printing "optimization failed".
            print("optimization failed")

        print("----------------- OPTIMIZATION DETAILS: --------------")
        print(f"n_sectors: {self.n_sectors.value}")
        print(f"max_weight: {self.max_weight.value}")
        print(f"total_turnover: {self.total_turnover.value}")
        print(f"sector_abs_weight: {self.sector_abs_weight.value}")
        print(f"eps: {self.eps.value}")
        print(f"alpha: {self.alpha.value}")

    def run_optimize(
        self,
        weights_two,
        x,
        y,
        z,
        all_pos_padded,
        all_neg_padded,
        sector_weights,
        problem,
        opts,
    ):

        start_time = time.perf_counter()
        problem.solve(**opts)
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
        self.print_details(
            optimize_weights,
            buffers,
            weights_two,
            all_pos_padded,
            all_neg_padded,
            sector_weights,
        )
        return optimize_weights, buffers, gate_a


class elastic_net(portfolio):
    def __init__(
        self,
        lambda_a: float = 1.0,
        lambda_b: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # lambda_a/lambda_b multiply convex terms (an L1 norm and a sum of
        # squares) in the objective -- same reasoning as self.alpha above:
        # they must be nonneg=True for CVXPY to accept them as Parameters
        # there, otherwise you're forced back to baking in .value as a float.
        self.lambda_a = cp.Parameter(nonneg=True, value=lambda_a)
        self.lambda_b = cp.Parameter(nonneg=True, value=lambda_b)
        print("initialized elastic_net")

    def elastic_generate_weights(self):
        return super().generate_weights()

    def elastic_generate_bounds_masks(self, weights_one, weights_two):
        return super().generate_bounds_masks(weights_one, weights_two)

    def elastic_generate_constraints(
        self,
        weights_one,
        weights_two,
        x,
        y,
        z,
        all_pos_padded,
        all_neg_padded,
        force_zero_padded,
        lb_stocks,
        ub_stocks,
        sector_labels,
    ):
        # The old version of this method did:
        #     constraints = super().generate_constraints(...)
        # but generate_constraints() returns a *tuple*, (constraints, sector_weights).
        # That tuple got wrapped in another list and run through
        # functools.reduce(operator.add, ...), which raises
        # `TypeError: can only concatenate tuple (not "list") to tuple`.
        # And even if it hadn't crashed there, this method only returned a
        # single value while run_elastic_optimize() unpacks two
        # (`constraints, sector_weights = self.elastic_generate_constraints(...)`),
        # which fails with its own unpacking error. In short: this was the
        # actual reason the file as given can't run at all.
        constraints, sector_weights = super().generate_constraints(
            weights_one,
            weights_two,
            x,
            y,
            z,
            all_pos_padded,
            all_neg_padded,
            force_zero_padded,
            lb_stocks,
            ub_stocks,
            sector_labels,
        )

        # Also removed the extra hard constraint that used to live here:
        #     net = [cp.norm(x - weights_one, 1) <= self.individual_turnover]
        # self.individual_turnover defaults to 0.005 and is meant as a
        # *per-stock* box-turnover bound (see generate_bounds_masks). Using it
        # to cap cp.norm(x - weights_one, 1) -- the *portfolio-wide* sum of
        # absolute turnover -- caps total turnover at 0.5%, which is almost
        # certainly infeasible for any real rebalance and duplicates the
        # turnover control that already exists via y / z / self.total_turnover
        # (total_turnover_component_constraint + v_constraint already make
        # sum(y) an L1-turnover proxy, soft-capped at self.total_turnover via
        # the alpha-penalized z gate). If you want a *hard* ceiling on top of
        # that soft gate, add it back with self.total_turnover, not
        # self.individual_turnover.

        return constraints, sector_weights

    def objective_elastic_net(self, x, y, z, weights_one, weights_two):
        l1_gate_constraint = self.lambda_a * cp.norm(x - weights_one, 1)
        l2_gate_constraint = self.lambda_b * cp.sum_squares(x - weights_one)
        objective = (
            super().objective_b(x, y, z, weights_two)
            + l1_gate_constraint
            + l2_gate_constraint
        )
        return objective

    def run_elastic_optimize(
        self,
        individual_turnover: float | None = None,
        max_weight: float | None = None,
        total_turnover: float | None = None,
        day: int | None = None,
        speed: int | None = None,
        alpha: float | None = None,
        n_sectors: int | None = None,
        lambda_a: float | None = None,
        lambda_b: float | None = None,
    ):
        if lambda_a is not None:
            self.lambda_a.value = lambda_a
        if lambda_b is not None:
            self.lambda_b.value = lambda_b
        if individual_turnover is not None:
            self.individual_turnover.value = individual_turnover
        if max_weight is not None:
            self.max_weight.value = max_weight
        if total_turnover is not None:
            self.total_turnover.value = total_turnover
        if day is not None:
            self.day.value = day
        if speed is not None:
            self.speed.value = speed
        if alpha is not None:
            self.alpha.value = alpha
        if n_sectors is not None:
            self.n_sectors.value = n_sectors

        weights_one, weights_two, x, y, z = self.elastic_generate_weights()
        lb_stocks, ub_stocks, all_pos, all_neg, force_zero = (
            self.elastic_generate_bounds_masks(weights_one, weights_two)
        )
        constraints, sector_weights = self.elastic_generate_constraints(
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
        clarabel_opts, piqp_opts = super().generate_opt()
        problem = cp.Problem(
            cp.Minimize(self.objective_elastic_net(x, y, z, weights_one, weights_two)),
            constraints,
        )

        # NOTE: the old code called problem.solve(**clarabel_opts) here, then
        # called it *again* inside run_optimize() below -- i.e. every call to
        # run_elastic_optimize solved the identical problem twice, and the
        # timing measured around the first solve was never even printed.
        # run_optimize() below is now the single solve call.
        (
            optimize_weights,
            buffers,
            gate_a,
        ) = super().run_optimize(
            weights_two,
            x,
            y,
            z,
            all_pos,
            all_neg,
            sector_weights,
            problem,
            clarabel_opts,
        )
        return optimize_weights, buffers, gate_a


elastic = elastic_net(0, 1)
elastic.run_elastic_optimize()
