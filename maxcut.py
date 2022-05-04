import copy
from typing import Callable, Tuple
import pickle

import cvxpy as cp
import numpy as np
from scipy import io
from scipy.linalg import eigh_tridiagonal, pinv
from scipy.sparse import csgraph, csc_matrix, coo_matrix
#from tqdm import trange

from IPython import embed


APPROX_EPS = 1e-4   # for numerical operations
CONVERGE_EPS = 1e-3 # for measuring convergence


def get_cut_size(L: csc_matrix, pred_cut: np.array) -> int:
    cut_size = (pred_cut[None,:] @ L @ pred_cut[:, None]).item() / 4
    return int(cut_size)


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


def solve_maxcut_sketchyfast(laplacian: csc_matrix, R: int, T: int
        ) -> np.array:

    C = -1.0 * laplacian
    n = laplacian.shape[0]
    b = np.ones((n,))
    tau = n

    ## scaling params
    scale_C = 1.0/np.linalg.norm(C.data) # equivalent to frobenius norm
    scale_X = 1.0/n

    C *= scale_C
    b *= scale_X
    tau *= scale_X

    # define the primitives
    objective_primitive = lambda u: (C @ u)
    adjoint_constraint_primitive = lambda u, z: u * z
    constraint_primitive = lambda u: u * u

    # initialize everything
    Omega = np.random.normal(size=(n, R))
    S = np.zeros((n, R))
    z = np.zeros((n,))
    y = np.zeros((n,))
    obj_val = 0.0
    best_obj_lb = -np.inf
    sigma_init = 1.0

    for t in range(T):
        sigma = sigma_init * np.sqrt(t + 2)
        eta = 2 / (t + 2.0)

        state_vec = y + sigma * (z-b)
        adjoint_closure = lambda u: adjoint_constraint_primitive(u, state_vec)
        num_iters = int(np.ceil((t+1)**(1/4) * np.log(n)))

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
                    - min_eigen_val * tau).squeeze()
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
        best_obj_lb = np.max([obj_lb, best_obj_lb])
        lb_gap = aug_lagrangian - best_obj_lb

        if t % 100 == 0:
            print('t = ', t)
            print('infeas = ', np.linalg.norm(infeas, 1))
            print('obj val = ', obj_val)
            print('dual gap = ', dual_gap)
            print('sub opt = ', sub_opt)
            print('aug lagrangian = ', aug_lagrangian)
            print('obj lb = ', obj_lb)
            print('best obj lb = ', best_obj_lb)
            print('lb gap = ', lb_gap)
            print()

        if lb_gap < CONVERGE_EPS and np.linalg.norm(infeas, 1) < CONVERGE_EPS:
            break

        # update state variables
        z = (1-eta) * z + eta * h 
        gamma = np.clip((4 * tau**2) / (((t + 2)**(3/2)) * (np.linalg.norm(z - b)**2)), 0, 1)
        y = y + gamma * (z-b)
        S = (1-eta) * S + eta * tau * (min_eigen_vec[:,None]
                                         @ (min_eigen_vec[None,:] @ Omega))
        # update running objective
        obj_val = (1-eta) * obj_val + eta * obj_update 

    # reconstruct matrix
    U, Lambda = reconstruct(Omega, S)

    # trace correction
    Lambda_tr_correct = Lambda + (tau - np.sum(Lambda)) / R

    return U
    

if __name__ == '__main__':

    R = 25
    T = int(1e4)

    np.random.seed(42)

    MATFILENAME = 'G1.mat'
    matobj = io.loadmat(MATFILENAME)
    graph = matobj['Problem'][0][0][1]

    ## create synthetic example for testing
    #row = np.array([0, 0, 1, 1, 2, 3])
    #col = np.array([1, 2, 2, 3, 4, 4])
    ##row = np.array([0, 0])
    ##col = np.array([1, 2])
    #data = np.ones_like(row)
    #graph = coo_matrix((data, (row, col)), shape=(5, 5)).tocsc()
    #graph = graph + graph.T

    laplacian = csgraph.laplacian(graph, normed=False).tocsc()

    ## slow maxcut
    #X_slow = solve_maxcut_slow(laplacian)

    with open('X_slow.pkl', 'rb') as f:
        X_slow = pickle.load(f)

    U_slow, Lambda_slow, _ = np.linalg.svd(X_slow)

    # fast maxcut
    U_fast = solve_maxcut_sketchyfast(laplacian, R, T)

    R = 5

    for i in range(R):
        pred_cut = 2 * (U_slow[:, i] > 0).astype(float) - 1
        cut_size = get_cut_size(laplacian, pred_cut)
        print('slow cut size: ', cut_size)

    print()

    for i in range(R):
        pred_cut = 2 * (U_fast[:, i] > 0).astype(float) - 1
        cut_size = get_cut_size(laplacian, pred_cut)
        print('fast cut size: ', cut_size)

    embed()
    exit()

    #cut_size = (pred_cut[None,:] @ laplacian @ pred_cut[:, None]).item() / 4
