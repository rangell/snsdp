"""
Solve MaxCut sdp fast.
"""

import argparse
import json
import logging
import pickle
import time
from typing import Callable, Dict, Tuple

import cvxpy as cp
import numpy as np
from IPython import embed  # type: ignore
from numpy.polynomial import Polynomial
from scipy import io  # type: ignore
from scipy.linalg import eigh_tridiagonal  # type: ignore
from scipy.sparse import csc_matrix, csgraph  # type: ignore

from logger import create_logger

APPROX_EPS = 1e-4  # for numerical operations
CONVERGE_EPS = 1e-1  # for measuring convergence
FIXED_POINT_TOLERANCE = 1e-1


def solve_maxcut_slow(L: csc_matrix) -> np.ndarray:
    """
    Baseline method for solving maxcut.
    """
    n = L.shape[0]
    X = cp.Variable((n, n), PSD=True)
    prob = cp.Problem(cp.Maximize(cp.trace(L @ X)), [cp.diag(X) == np.ones((n,))],)
    prob.solve(solver=cp.SCS, verbose=True)
    return X.value


def approx_extreme_eigen(
    objective_primitive: Callable[[np.ndarray], np.ndarray],
    adjoint_constraint_primitive_closure: Callable[[np.ndarray], np.ndarray],
    n: int,
    num_iters: int,
) -> Tuple[float, np.ndarray, float, np.ndarray]:

    V = np.empty((num_iters, n))
    omegas = np.empty((num_iters,))
    rhos = np.empty((num_iters - 1,))

    v_0 = np.random.normal(size=(n,))
    V[0] = v_0 / np.linalg.norm(v_0)

    for i in range(num_iters):
        transformed_v = objective_primitive(
            V[i]
        ) + adjoint_constraint_primitive_closure(V[i])
        omegas[i] = np.dot(V[i], transformed_v)
        if i == num_iters - 1:
            break  # we have all we need
        V[i + 1] = transformed_v - (omegas[i] * V[i])
        if i > 0:
            V[i + 1] -= rhos[i - 1] * V[i - 1]
        rhos[i] = np.linalg.norm(V[i + 1])
        if rhos[i] < APPROX_EPS:
            break
        V[i + 1] = V[i + 1] / rhos[i]

    min_eigen_val, u = eigh_tridiagonal(
        omegas[: i + 1], rhos[:i], select="i", select_range=(0, 0)
    )
    min_eigen_vec = (u.T @ V[: i + 1]).squeeze()
    # renormalize for stability
    min_eigen_vec = min_eigen_vec / np.linalg.norm(min_eigen_vec)

    max_eigen_val, u = eigh_tridiagonal(
        omegas[: i + 1], rhos[:i], select="i", select_range=(i, i)
    )
    max_eigen_vec = (u.T @ V[: i + 1]).squeeze()
    # renormalize for stability
    max_eigen_vec = max_eigen_vec / np.linalg.norm(max_eigen_vec)

    return (
        min_eigen_val.squeeze(),
        min_eigen_vec,
        max_eigen_val.squeeze(),
        max_eigen_vec,
    )


def reconstruct(Omega: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = Omega.shape[0]
    rho = np.sqrt(n) * APPROX_EPS * np.linalg.norm(S, ord=2)
    S_rho = S + rho * Omega
    B = Omega.T @ S_rho
    B = 0.5 * (B + B.T)
    L = np.linalg.cholesky(B)
    U, Rho, _ = np.linalg.svd(
        np.linalg.lstsq(L, S_rho.T, rcond=-1)[0].T,
        full_matrices=False,  # this compresses the output to be rank `R`
    )
    Lambda = np.clip(Rho ** 2 - rho, 0, np.inf)
    return U, Lambda


def get_step_estimate_1(
    aug_infeas_norm: float,
    lagrangian_gap: float,
    sigma_init: float,
    best_curve_est: float,
) -> float:
    # solve order 3 polynomial for sigma_est
    poly = Polynomial(
        [
            -2 * (sigma_init ** 2) * lagrangian_gap,
            -2 * (sigma_init ** 2) * aug_infeas_norm - 2 * best_curve_est * sigma_init,
            lagrangian_gap,
            aug_infeas_norm,
        ]
    )
    sigma_est = np.max(poly.roots())
    assert sigma_est > 0
    step_est = (sigma_est / sigma_init) ** 2 - 2
    return step_est


def get_step_estimate_2(
    aug_infeas_norm: float,
    lagrangian_gap: float,
    sigma_init: float,
    best_curve_est: float,
) -> float:
    # solve order 3 polynomial for sigma_est
    poly = Polynomial(
        [
            -2 * (sigma_init ** 2) * lagrangian_gap - 2 * best_curve_est * sigma_init,
            -2 * (sigma_init ** 2) * aug_infeas_norm,
            lagrangian_gap,
            aug_infeas_norm,
        ]
    )
    sigma_est = np.max(poly.roots())
    assert sigma_est.imag == 0.0
    sigma_est = sigma_est.real
    assert sigma_est > 0
    step_est = (sigma_est / sigma_init) ** 2 - 2
    return step_est


def get_step_estimate_4(
    aug_infeas_norm: float,
    lagrangian_gap: float,
    sigma_init: float,
    best_curve_est: float,
) -> float:
    # solve order 5 polynomial for sigma_est
    poly = Polynomial(
        [
            4 * (sigma_init ** 4) * lagrangian_gap,
            4 * (sigma_init ** 4) * aug_infeas_norm,
            -4 * best_curve_est * (sigma_init ** 4)
            - 4 * (sigma_init ** 2) * lagrangian_gap,
            -4 * (sigma_init ** 2) * aug_infeas_norm,
            lagrangian_gap,
            aug_infeas_norm,
        ]
    )
    sigma_est = np.max(poly.roots())
    assert sigma_est.imag == 0.0
    sigma_est = sigma_est.real
    assert sigma_est > 0
    step_est = (sigma_est / sigma_init) ** 2 - 2
    return step_est


def sketchy_cgal(
    logger: logging.Logger,
    S: np.ndarray,
    Omega: np.ndarray,
    z: np.ndarray,
    b: np.ndarray,
    y: np.ndarray,
    objective_primitive: Callable[[np.ndarray], np.ndarray],
    adjoint_constraint_primitive: Callable[[np.ndarray, np.ndarray], np.ndarray],
    constraint_primitive: Callable[[np.ndarray], np.ndarray],
    obj_val: float,
    R: int,
    scale_C: float,
    scale_X: float,
    T: int,
    soln_quality_callback: Callable[[np.ndarray, np.ndarray], int],
    eval_freq: int,
    step_size_mode: str,
    warm_start: bool,
) -> Dict:

    assert step_size_mode in ["std", "static", "dynamic"]
    assert not (step_size_mode == "static" and not warm_start)

    n = z.shape[0]

    best_obj_lb = -np.inf
    best_curve_est = 0.0
    sigma_init = 1.0

    # for computing `aug_prob_gap`
    sigma_final = 1 / (scale_C * CONVERGE_EPS)

    start_time = time.time()

    if not warm_start:
        step_num = 0.0
    else:
        step_num = 1.0

    for t in range(T):
        sigma = sigma_init * np.sqrt(step_num + 2)
        eta = 2.0 / (step_num + 2.0)
        infeas = z - b

        # need initial `best_obj_lb` and `best_curve_est` for either
        # "static" or "dynamic" warm-start
        if t == 0 and warm_start and step_size_mode in ["static", "dynamic"]:
            state_vec = y + sigma * infeas

            def adjoint_closure(u: np.ndarray) -> np.ndarray:
                return adjoint_constraint_primitive(u, state_vec)

            num_iters = int((np.ceil((step_num + 1) ** (1 / 4)) * np.log(n)))

            # compute primal update direction via randomized Lanczos
            min_eigen_val, min_eigen_vec, max_eigen_val, _ = approx_extreme_eigen(
                objective_primitive, adjoint_closure, n, num_iters
            )
            h = tau * constraint_primitive(min_eigen_vec)
            obj_update = tau * np.dot(min_eigen_vec, objective_primitive(min_eigen_vec))

            # compute objective lower bound
            aug_lagrangian = (
                obj_val
                + np.dot(y, infeas)
                + 0.5 * sigma * (np.linalg.norm(infeas) ** 2)
            )
            obj_lb = (
                aug_lagrangian
                + obj_update
                - obj_val
                + np.dot(y + sigma * (infeas), h - z)
                + 0.5 * sigma * (np.linalg.norm(infeas) ** 2)
            )
            best_obj_lb = np.max([obj_lb, best_obj_lb])

            # compute curvature estimate
            next_infeas = (1 - eta) * z + eta * h - b
            next_aug_lagrangian = (
                (1 - eta) * obj_val
                + eta * obj_update
                + np.dot(y, next_infeas)
                + 0.5 * sigma * (np.linalg.norm(next_infeas) ** 2)
            )
            linear_update_mag = eta * (
                obj_update - obj_val + np.dot(y + sigma * (infeas), h - z)
            )

            curr_curve_est = (
                next_aug_lagrangian - aug_lagrangian - linear_update_mag
            ) / (0.5 * eta ** 2 * sigma)
            if curr_curve_est > best_curve_est:
                best_curve_est = curr_curve_est

        # compute accelerated pseudo step num
        if (t == 0 and step_size_mode == "static") or (
            (t > 1 or warm_start) and step_size_mode == "dynamic"
        ):
            rescale_infeas = infeas / scale_X
            rescale_y = y / scale_X
            rescale_obj_val = obj_val / (scale_X * scale_C)
            rescale_best_obj_lb = best_obj_lb / (scale_X * scale_C)

            aug_infeas_norm = 0.5 * (np.linalg.norm(rescale_infeas) ** 2)
            lagrangian_gap = (
                rescale_obj_val
                + np.dot(rescale_y, rescale_infeas)
                - rescale_best_obj_lb
            )

            step_num = get_step_estimate_4(
                aug_infeas_norm,
                lagrangian_gap,
                sigma_init,
                curr_curve_est / (scale_X ** 2),
            )
            sigma = sigma_init * np.sqrt(step_num + 2.0)
            eta = 2.0 / (step_num + 2.0)

        # compute primal update direction via randomized Lanczos
        state_vec = y + sigma * infeas

        def adjoint_closure(u: np.ndarray) -> np.ndarray:
            return adjoint_constraint_primitive(u, state_vec)

        num_iters = int((np.ceil((step_num + 1) ** (1 / 4)) * np.log(n)))

        min_eigen_val, min_eigen_vec, max_eigen_val, _ = approx_extreme_eigen(
            objective_primitive, adjoint_closure, n, num_iters
        )
        h = tau * constraint_primitive(min_eigen_vec)
        obj_update = tau * np.dot(min_eigen_vec, objective_primitive(min_eigen_vec))

        aug_lagrangian = (
            obj_val + np.dot(y, infeas) + 0.5 * sigma * (np.linalg.norm(infeas) ** 2)
        )
        obj_lb = (
            aug_lagrangian
            + obj_update
            - obj_val
            + np.dot(y + sigma * (infeas), h - z)
            + 0.5 * sigma * (np.linalg.norm(infeas) ** 2)
        )
        best_obj_lb = np.max([obj_lb, best_obj_lb])

        # compute curvature estimate
        if step_size_mode == "dynamic":
            next_infeas = (1 - eta) * z + eta * h - b
            next_aug_lagrangian = (
                (1 - eta) * obj_val
                + eta * obj_update
                + np.dot(y, next_infeas)
                + 0.5 * sigma * (np.linalg.norm(next_infeas) ** 2)
            )
            linear_update_mag = eta * (
                obj_update - obj_val + np.dot(y + sigma * (infeas), h - z)
            )

            curr_curve_est = (
                next_aug_lagrangian - aug_lagrangian - linear_update_mag
            ) / (0.5 * eta ** 2 * sigma)
            if curr_curve_est > best_curve_est:
                best_curve_est = curr_curve_est

        # compute metrics and check for convergence
        if t > 0 and t % eval_freq == 0:
            rescale_infeas = infeas / scale_X
            rescale_obj_val = obj_val / (scale_X * scale_C)
            rescale_best_obj_lb = best_obj_lb / (scale_X * scale_C)

            aug_prob_gap = (
                rescale_obj_val
                + 0.5 * sigma_final * (np.linalg.norm(rescale_infeas) ** 2)
                - rescale_best_obj_lb
            )

            U, Lambda = reconstruct(Omega, S)
            Lambda_tr_correct = Lambda + (tau - np.sum(Lambda)) / R
            soln_quality = soln_quality_callback(U, Lambda_tr_correct)

            metrics = {
                "step_num": step_num,
                "infeas": np.linalg.norm(rescale_infeas, 2),
                "obj_val": rescale_obj_val,
                "best_obj_lb": rescale_best_obj_lb,
                "aug_prob_gap": aug_prob_gap,
                "soln_quality": soln_quality,
            }

            metric_str = "; ".join(
                [
                    k + " = " + ("{:.4f}".format(v) if isinstance(v, float) else str(v))
                    for k, v in metrics.items()
                ]
            )
            logger.info("Round %d metrics - " + metric_str, t)

            # check convergence
            if aug_prob_gap < CONVERGE_EPS:
                break

        # update state variables
        z = (1 - eta) * z + eta * h
        gamma = np.clip(
            (4 * tau ** 2)
            / (((step_num + 2) ** (3 / 2)) * (np.linalg.norm(z - b) ** 2)),
            0,
            1,
        )
        y = y + gamma * (z - b)
        S = (1 - eta) * S + eta * tau * (
            min_eigen_vec[:, None] @ (min_eigen_vec[None, :] @ Omega)
        )
        # update running objective
        obj_val = (1 - eta) * obj_val + eta * obj_update

        # increment step number
        step_num += 1

    result_dict = {
        "U": U,
        "Lambda": Lambda_tr_correct,
        "S": S,
        "z": z,
        "y": y,
        "num_iters": t,
        "time": time.time() - start_time,
        "infeas": np.linalg.norm(infeas, 2),
        "soln_quality": soln_quality,
        "sigma": sigma,
        "obj_val": obj_val,
    }

    return result_dict


def maxcut_quality(L: csc_matrix, R: int, U: np.ndarray, Lambda: np.ndarray):
    maxcut_size = 0
    for i in range(R):
        pred_cut = 2 * (U[:, i] > 0).astype(float) - 1
        cut_size = get_cut_size(L, pred_cut)
        if cut_size > maxcut_size:
            maxcut_size = cut_size
    return int(maxcut_size)


def get_cut_size(L: csc_matrix, pred_cut: np.ndarray) -> int:
    cut_size = (pred_cut[None, :] @ L @ pred_cut[:, None]).item() / 4
    return int(cut_size)


def soln_quality_callback_closure(
    laplacian: csc_matrix, R: int
) -> Callable[[np.ndarray, np.ndarray], int]:
    def soln_quality_callback(U: np.ndarray, Lambda: np.ndarray) -> int:
        return maxcut_quality(laplacian, R, U, Lambda)

    return soln_quality_callback


def objective_primitive_closure(C: csc_matrix) -> Callable[[np.ndarray], np.ndarray]:
    def objective_primitive(u: np.ndarray) -> np.ndarray:
        return C @ u

    return objective_primitive


def adjoint_constraint_primitive(u: np.ndarray, z: np.ndarray) -> np.ndarray:
    return u * z


def constraint_primitive(u: np.ndarray) -> np.ndarray:
    return u * u


def get_hparams():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enables and disables certain opts for debugging",
    )
    parser.add_argument(
        "--output_dir", default=None, type=str, help="output directory for this run"
    )
    parser.add_argument(
        "--data_path", default=None, type=str, required=True, help="path to matfile"
    )

    parser.add_argument(
        "--warm_start_data_frac",
        type=float,
        default=0.8,
        help="fraction of data to warm start with",
    )
    parser.add_argument(
        "--test_data_frac",
        type=float,
        default=1.0,
        help="fraction of data to use to test warm start",
    )

    parser.add_argument(
        "--warm_start",
        action=argparse.BooleanOptionalAction,
        help="whether or not to warm start",
    )
    parser.add_argument(
        "--step_size_mode",
        type=str,
        default="std",
        choices=["std", "static", "dynamic"],
        help="step size mode for optimization",
    )

    hparams = parser.parse_args()
    return hparams


if __name__ == "__main__":
    # get commandline arguments
    hparams = get_hparams()

    # create logger
    logger = create_logger(hparams.output_dir, hparams.debug)

    # TODO: move these to hparams
    R = 25
    T = int(1e6)
    eval_freq = 100

    logger.info(
        "Experiment args:\n{}".format(
            json.dumps(vars(hparams), sort_keys=True, indent=4)
        )
    )

    np.random.seed(hparams.seed)

    matobj = io.loadmat(hparams.data_path)
    graph = matobj["Problem"][0][0][1]

    n = graph.shape[0]
    laplacian = csgraph.laplacian(graph, normed=False).tocsc()

    # X = solve_maxcut_slow(laplacian)

    Omega = np.random.normal(size=(n, R))

    # prepare warm start
    warm_start_n = int(hparams.warm_start_data_frac * n)
    warm_start_graph = graph[:warm_start_n, :warm_start_n]
    warm_start_laplacian = csgraph.laplacian(warm_start_graph, normed=False).tocsc()

    C = -1.0 * warm_start_laplacian
    b = np.ones((warm_start_n,))
    tau = float(warm_start_n)

    # scaling params
    warm_start_scale_C = 1.0 / np.linalg.norm(C.data)  # equivalent to frobenius norm
    warm_start_scale_X = 1.0 / warm_start_n

    C *= warm_start_scale_C
    b *= warm_start_scale_X
    tau *= warm_start_scale_X

    # initialize everything
    S = np.zeros((warm_start_n, R))
    z = np.zeros((warm_start_n,))
    y = np.zeros((warm_start_n,))

    obj_val = 0.0

    logger.warning('USING "dynamic" FOR COMPUTING WARM-START RESULT')

    # get warm-start result
    warm_start_result_dict = sketchy_cgal(
        logger,
        S,
        Omega[:warm_start_n, :],
        z,
        b,
        y,
        objective_primitive_closure(C),
        adjoint_constraint_primitive,
        constraint_primitive,
        obj_val,
        R,
        scale_C=warm_start_scale_C,
        scale_X=warm_start_scale_X,
        T=T,
        soln_quality_callback=soln_quality_callback_closure(warm_start_laplacian, R),
        eval_freq=100,
        step_size_mode="dynamic",
        warm_start=False,
    )

    embed()
    exit()

    # with open("SCS_G22_X.pkl", "rb") as f:
    #    X_opt = pickle.load(f)
    # S_opt = X_opt @ Omega
    # U_opt, Lambda_opt = reconstruct(Omega, S_opt)
    # Lambda_tr_correct = Lambda_opt + (tau - np.sum(Lambda_opt)) / R
    # opt_soln_quality = soln_quality_callback_closure(warm_start_laplacian, R)(
    #    U_opt, Lambda_tr_correct
    # )
    # logger.info("Optimal sketched solution quality: %d", opt_soln_quality)

    # with open('G63_warm_start_6999_R-2.pkl', 'rb') as f:
    #    warm_start_result_dict = pickle.load(f)
    # with open('G63_warm_start_6979_R-2.pkl', 'rb') as f:
    #    warm_start_result_dict = pickle.load(f)
    # with open('G63_warm_start_6930_R-2.pkl', 'rb') as f:
    #    warm_start_result_dict = pickle.load(f)
    # with open('G63_warm_start_6930_R-100.pkl', 'rb') as f:
    #    warm_start_result_dict = pickle.load(f)

    # test with warm start mode
    test_n = int(hparams.test_data_frac * n)
    test_graph = graph[:test_n, :test_n]
    test_laplacian = csgraph.laplacian(test_graph, normed=False).tocsc()

    C = -1.0 * test_laplacian
    b = np.ones((test_n,))
    tau = test_n

    # scaling params
    test_scale_C = 1.0 / np.linalg.norm(C.data)  # equivalent to frobenius norm
    test_scale_X = 1.0 / test_n

    C *= test_scale_C
    b *= test_scale_X
    tau *= test_scale_X

    # initialize everything
    if not hparams.warm_start:
        S = np.zeros((test_n, R))
        z = np.zeros((test_n,))
        y = np.zeros((test_n,))
        obj_val = 0.0
    else:
        S = np.concatenate(
            [
                warm_start_result_dict["S"] * (test_scale_X / warm_start_scale_X),
                test_scale_X * Omega[warm_start_n:test_n, :],
            ]
        )
        z = np.concatenate(
            [
                warm_start_result_dict["z"] * (test_scale_X / warm_start_scale_X),
                np.ones((test_n - warm_start_n,)) * test_scale_X,
            ]
        )
        y = np.concatenate(
            [
                warm_start_result_dict["y"] * (test_scale_X / warm_start_scale_X),
                np.zeros((test_n - warm_start_n,)),
            ]
        )
        obj_val = (
            warm_start_result_dict["obj_val"]
            * (test_scale_C / warm_start_scale_C)
            * (test_scale_X / warm_start_scale_X)
        )

    # get test result
    test_result_dict = sketchy_cgal(
        logger,
        S,
        Omega[:test_n, :],
        z,
        b,
        y,
        objective_primitive_closure(C),
        adjoint_constraint_primitive,
        constraint_primitive,
        obj_val,
        R,
        scale_C=warm_start_scale_C,
        scale_X=warm_start_scale_X,
        T=T,
        soln_quality_callback=soln_quality_callback_closure(test_laplacian, R),
        eval_freq=100,
        step_size_mode=hparams.step_size_mode,
        warm_start=hparams.warm_start,
    )
