import argparse
import copy
import pickle
import time
from typing import Callable, Tuple

import cvxpy as cp
import numpy as np
from scipy import io
from scipy.linalg import eigh_tridiagonal, pinv
from scipy.sparse import csgraph, csc_matrix, coo_matrix
#from tqdm import trange

from IPython import embed


APPROX_EPS = 1e-4   # for numerical operations
CONVERGE_EPS = 1e-3 # for measuring convergence


def solve_maxcut_slow(L: csc_matrix) -> np.array:
    n = L.shape[0]
    X = cp.Variable((n, n), PSD=True)
    prob = cp.Problem(cp.Maximize(cp.trace(L @ X)),
                      [cp.diag(X) == np.ones((n,))])
    prob.solve(solver=cp.SCS, verbose=True)
    return X.value


def approx_extreme_eigen(
        objective_primitive: Callable[[np.array], np.array],
        adjoint_constraint_primitive_closure: Callable[[np.array], np.array],
        n: int,
        num_iters: int) -> Tuple[float, np.array, float, np.array]:

    V = np.empty((num_iters, n))
    omegas = np.empty((num_iters,))
    rhos = np.empty((num_iters-1,))

    v_0 = np.random.normal(size=(n,))
    V[0] = v_0 / np.linalg.norm(v_0)

    for i in range(num_iters):
        transformed_v = (objective_primitive(V[i])
                         + adjoint_constraint_primitive_closure(V[i]))
        omegas[i] = np.dot(V[i], transformed_v)
        if i == num_iters-1:
            break # we have all we need
        V[i+1] = transformed_v - (omegas[i] * V[i])
        if i > 0:
            V[i+1] -= (rhos[i-1] * V[i-1])
        rhos[i] = np.linalg.norm(V[i+1])
        if rhos[i] < np.sqrt(n) * APPROX_EPS: 
            break
        V[i+1] = V[i+1] / rhos[i]

    min_eigen_val, u = eigh_tridiagonal(
            omegas[:i+1], rhos[:i], select='i', select_range=(0, 0))
    min_eigen_vec = (u.T @ V[:i+1]).squeeze()
    # renormalize for stability
    min_eigen_vec = min_eigen_vec / np.linalg.norm(min_eigen_vec)

    max_eigen_val, u = eigh_tridiagonal(
            omegas[:i+1], rhos[:i], select='i', select_range=(i, i))
    max_eigen_vec = (u.T @ V[:i+1]).squeeze()
    # renormalize for stability
    max_eigen_vec = max_eigen_vec / np.linalg.norm(max_eigen_vec)

    return (min_eigen_val.squeeze(),
            min_eigen_vec,
            max_eigen_val.squeeze(),
            max_eigen_vec)


def reconstruct(Omega: np.array, S: np.array) -> Tuple[np.array, np.array]:
    n = Omega.shape[0]
    rho = np.sqrt(n) * APPROX_EPS * np.linalg.norm(S, ord=2)
    S_rho = S + rho * Omega
    B = Omega.T @ S_rho
    B = 0.5 * (B + B.T)
    L = np.linalg.cholesky(B)
    U, Rho, _ = np.linalg.svd(
            np.linalg.lstsq(L, S_rho.T, rcond=-1)[0].T,
            full_matrices=False # this compresses the output to be rank `R`
    )
    Lambda = np.clip(Rho**2 - rho, 0, np.inf)
    return U, Lambda


def sketchy_cgal(
        S: np.ndarray,
        Omega: np.ndarray,
        z: np.array,
        b: np.array,
        y: np.array,
        objective_primitive: Callable[[np.array], np.array],
        adjoint_constraint_primitive: Callable[[np.array, np.array], np.array],
        constraint_primitive: Callable[[np.array], np.array],
        obj_val: float,
        R: int,
        T: int,
        soln_quality_callback: Callable[[np.ndarray, np.array], int],
        eval_freq: int,
        warm_start_mode: str) -> np.array:

    assert warm_start_mode in ['none', 'just_data', 'static', 'dynamic']

    n = z.shape[0]

    best_obj_lb = -np.inf
    sigma_init = 1.0

    start_time = time.time()

    if warm_start_mode == 'none':
        step_num = 0.0
    elif warm_start_mode == 'just_data':
        step_num = 1.0
    else:
        step_num = 2.0

    for t in range(T):
        sigma = sigma_init * np.sqrt(step_num + 2)

        # determine eta and sigma (step-size and smoothing parameters)
        while True:
            state_vec = y + sigma * (z-b)
            adjoint_closure = lambda u: adjoint_constraint_primitive(u, state_vec)
            num_iters = int(np.ceil((step_num+1)**(1/4) * np.log(n)))

            # compute primal update direction via randomized Lanczos
            min_eigen_val, min_eigen_vec, max_eigen_val, _ = approx_extreme_eigen(
                    objective_primitive, adjoint_closure, n, num_iters
            )
            h = tau * constraint_primitive(min_eigen_vec)
            obj_update = tau * np.dot(min_eigen_vec,
                                      objective_primitive(min_eigen_vec))

            # compute objective tracking metrics
            infeas = z - b
            dual_gap = (obj_val + np.dot(y + sigma * (infeas), z).squeeze()
                        - min_eigen_val).squeeze()
            sub_opt = dual_gap - np.dot(y, infeas) \
                    - 0.5*sigma*(np.linalg.norm(infeas)**2)
            aug_lagrangian = (obj_val + np.dot(y, infeas)
                              + 0.5*sigma*(np.linalg.norm(infeas)**2))
            obj_lb = (
                    aug_lagrangian
                    + obj_update - obj_val + np.dot(y + sigma * (infeas), h - z)
                    + 0.5*sigma*(np.linalg.norm(infeas)**2)
                    - tau * (sigma_init / sigma)
                        * np.max([np.abs(min_eigen_val), np.abs(max_eigen_val)])
            )
            #obj_lb = obj_val - sub_opt
            if t > 0:
                best_obj_lb = np.max([obj_lb, best_obj_lb])
            else:
                best_obj_lb = obj_lb
            lb_gap = aug_lagrangian - best_obj_lb

            if warm_start_mode in ['none', 'just_data']:
                break
            if warm_start_mode == 'static' and t > 0:
                break

            # binary search for sigma here
            approx_step_num = 4 / lb_gap

            # can never go back in time
            if approx_step_num < step_num:
                break
            
            approx_sigma = sigma_init * np.sqrt(approx_step_num + 2)
            sigma_gap = np.abs(approx_sigma - sigma)
            sigma += (approx_sigma - sigma) / 2

            if warm_start_mode in ['static', 'dynamic'] and sigma_gap < 1e-2:
                step_num = approx_step_num
                break

        if (t > 0 and t % eval_freq == 0) or (lb_gap < CONVERGE_EPS
                and np.linalg.norm(infeas, 2) < CONVERGE_EPS):
            print('t = ', t)
            print('step_num = ', step_num)
            print('infeas = ', np.linalg.norm(infeas, 2))
            print('obj val = ', obj_val)
            print('dual gap = ', dual_gap)
            print('sub opt = ', sub_opt)
            print('aug lagrangian = ', aug_lagrangian)
            print('obj lb = ', obj_lb)
            print('best obj lb = ', best_obj_lb)
            print('lb gap = ', lb_gap)

            U, Lambda = reconstruct(Omega, S)
            Lambda_tr_correct = Lambda + (tau - np.sum(Lambda)) / R
            soln_quality = soln_quality_callback(U, Lambda_tr_correct)

            print('soln_quality = ', soln_quality)

            print()

            if (lb_gap < CONVERGE_EPS
                    and np.linalg.norm(infeas, 2) < CONVERGE_EPS):
                break

        eta = 2 / (step_num + 2.0)

        # update state variables
        z = (1-eta) * z + eta * h 
        gamma = np.clip((4 * tau**2) / (((step_num + 2)**(3/2)) * (np.linalg.norm(z - b)**2)), 0, 1)
        y = y + gamma * (z-b)
        S = (1-eta) * S + eta * tau * (min_eigen_vec[:,None]
                                         @ (min_eigen_vec[None,:] @ Omega))
        # update running objective
        obj_val = (1-eta) * obj_val + eta * obj_update 

        # increment step number
        step_num += 1


    result_dict = {
            'U': U,
            'Lambda': Lambda_tr_correct,
            'S': S,
            'z': z,
            'y': y,
            'num_iters': t,
            'time': time.time() - start_time,
            'lb_gap': lb_gap,
            'infeas': np.linalg.norm(infeas, 2),
            'soln_quality': soln_quality,
            'sigma': sigma,
            'obj_val': obj_val,
    }

    return result_dict


def get_hparams():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--debug', action='store_true',
                        help="enables and disables certain opts for debugging")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="output directory for this run")
    parser.add_argument("--data_path", default=None, type=str, required=True,
                        help="path to matfile")

    parser.add_argument('--warm_start_data_frac', type=float, default=0.8,
                        help="fraction of data to warm start with")
    parser.add_argument('--test_data_frac', type=float, default=1.0,
                        help="fraction of data to use to test warm start")

    parser.add_argument('--warm_start_mode', type=str, default='just_data',
                        choices=['none', 'just_data', 'static', 'dynamic'],
                        help="fraction of data to use to test warm start")

    hparams = parser.parse_args()
    return hparams


def maxcut_quality(L: csc_matrix, R: int, U: np.ndarray, Lambda: np.array):
    maxcut_size = 0
    for i in range(R):
        pred_cut = 2 * (U[:, i] > 0).astype(float) - 1
        cut_size = get_cut_size(L, pred_cut)
        if cut_size > maxcut_size:
            maxcut_size = cut_size
    return int(maxcut_size)
    

def get_cut_size(L: csc_matrix, pred_cut: np.array) -> int:
    cut_size = (pred_cut[None,:] @ L @ pred_cut[:, None]).item() / 4
    return int(cut_size)


if __name__ == '__main__':

    hparams = get_hparams()

    R = 25
    T = int(1e4)
    eval_freq = 100

    np.random.seed(hparams.seed)

    matobj = io.loadmat(hparams.data_path)
    graph = matobj['Problem'][0][0][1]

    n = graph.shape[0]

    Omega = np.random.normal(size=(n, R))

    # prepare warm start 
    warm_start_n = int(hparams.warm_start_data_frac * n)
    warm_start_graph = graph[:warm_start_n, :warm_start_n]
    warm_start_laplacian = csgraph.laplacian(
            warm_start_graph, normed=False).tocsc()

    soln_quality_callback = lambda U, Lambda: maxcut_quality(
            warm_start_laplacian, R, U, Lambda)  

    C = -1.0 * warm_start_laplacian
    b = np.ones((warm_start_n,))
    tau = warm_start_n

    # scaling params
    scale_C = 1.0/np.linalg.norm(C.data) # equivalent to frobenius norm
    scale_X = 1.0/warm_start_n

    C *= scale_C
    b *= scale_X
    tau *= scale_X

    # define the primitives
    objective_primitive = lambda u: (C @ u)
    adjoint_constraint_primitive = lambda u, z: u * z
    constraint_primitive = lambda u: u * u

    # initialize everything
    S = np.zeros((warm_start_n, R))
    z = np.zeros((warm_start_n,))
    y = np.zeros((warm_start_n,))

    obj_val = 0.0

    # get warm-start result
    warm_start_result_dict = sketchy_cgal(
            S,
            Omega[:warm_start_n,:],
            z,
            b,
            y,
            objective_primitive,
            adjoint_constraint_primitive,
            constraint_primitive,
            obj_val,
            R,
            T,
            soln_quality_callback,
            eval_freq,
            warm_start_mode='none'
    )

    if hparams.test_data_frac == hparams.warm_start_data_frac:
        exit()

    # test with warm start mode
    test_n = int(hparams.test_data_frac * n)
    test_graph = graph[:test_n, :test_n]
    test_laplacian = csgraph.laplacian(
            test_graph, normed=False).tocsc()

    soln_quality_callback = lambda U, Lambda: maxcut_quality(
            test_laplacian, R, U, Lambda)  

    C = -1.0 * test_laplacian
    b = np.ones((test_n,))
    tau = test_n

    # scaling params
    scale_C = 1.0/np.linalg.norm(C.data) # equivalent to frobenius norm
    scale_X = 1.0/test_n

    C *= scale_C
    b *= scale_X
    tau *= scale_X

    # define the primitives
    objective_primitive = lambda u: (C @ u)
    adjoint_constraint_primitive = lambda u, z: u * z
    constraint_primitive = lambda u: u * u

    # initialize everything
    if hparams.warm_start_mode == 'none':
        S = np.zeros((test_n, R))
        z = np.zeros((test_n,))
        y = np.zeros((test_n,))
        obj_val = 0.0
    else:
        # expand U randomly
        warm_start_U = warm_start_result_dict['U']
        random_indices = np.random.choice(warm_start_U.shape[0],
                                          test_n - warm_start_n,
                                          replace=True)
        U = np.concatenate([warm_start_U, warm_start_U[random_indices]])
        U = U / np.linalg.norm(U, axis=0)[None,:]

        Lambda = warm_start_result_dict['Lambda']

        X_factorized = U * np.sqrt(Lambda)[None, :]

        S = X_factorized @ (X_factorized.T @ Omega[:test_n,:])
        z = np.sum(X_factorized * X_factorized, axis=1)
        y = np.zeros((test_n,))
        y[:warm_start_n] = warm_start_result_dict['y']

        obj_val = np.trace(X_factorized.T @ (C @ X_factorized))

    # get test result
    test_result_dict = sketchy_cgal(
            S,
            Omega[:test_n,:],
            z,
            b,
            y,
            objective_primitive,
            adjoint_constraint_primitive,
            constraint_primitive,
            obj_val,
            R,
            T,
            soln_quality_callback,
            eval_freq,
            warm_start_mode=hparams.warm_start_mode
    )
