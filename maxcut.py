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


EPS = 1e-4


def solve_maxcut_slow(L: csc_matrix) -> np.array:
    n = L.shape[0]
    X = cp.Variable((n, n), PSD=True)
    prob = cp.Problem(cp.Maximize(cp.trace(L @ X)),
                      [cp.diag(X) == np.ones((n,))])
    prob.solve(solver=cp.SCS, verbose=True)
    return X.value


def approx_min_eigen(
        objective_primitive: Callable[[np.array], np.array],
        adjoint_constraint_primitive_closure: Callable[[np.array], np.array],
        n: int,
        num_iters: int) -> np.array:

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
        if rhos[i] < np.sqrt(n) * EPS: 
            break
        V[i+1] = V[i+1] / rhos[i]

    eigen_val, u = eigh_tridiagonal(
            omegas[:i+1], rhos[:i], select='i', select_range=(0, 0))
    eigen_vec = (u.T @ V[:i+1]).squeeze()

    # renormalize for stability
    eigen_vec = eigen_vec / np.linalg.norm(eigen_vec)

    return eigen_val, eigen_vec


def reconstruct(Omega: np.array, S: np.array) -> Tuple[np.array, np.array]:
    n = Omega.shape[0]
    sigma = np.sqrt(n) * EPS * np.linalg.norm(S, ord=2)
    S_sigma = S + sigma * Omega
    B = Omega.T @ S_sigma
    B = 0.5 * (B + B.T)
    L = np.linalg.cholesky(B)
    U, Sigma, _ = np.linalg.svd(
            np.linalg.lstsq(L, S_sigma.T, rcond=-1)[0].T,
            full_matrices=False # this compresses the output to be rank `R`
    )
    Lambda = np.clip(Sigma**2 - sigma, 0, np.inf)
    return U, Lambda


def solve_maxcut_sketchyfast(laplacian: csc_matrix, R: int, T: int
        ) -> np.array:

    C = -1.0 * laplacian
    n = laplacian.shape[0]
    b = np.ones((n,))
    alpha = n

    ### scaling params
    #scale_C = 1.0/np.linalg.norm(C.data) # equivalent to frobenius norm
    #scale_X = 1.0/n

    #b *= scale_X

    # define the primitives
    objective_primitive = lambda u: (C @ u)
    adjoint_constraint_primitive = lambda u, z: u * z
    constraint_primitive = lambda u: u * u

    # initialize everything
    Omega = np.random.normal(size=(n, R))
    S = np.zeros((n, R))
    z = np.zeros((n,))
    y = np.zeros((n,))
    obj_val = 0

    for t in range(T):
        beta = np.sqrt(t + 2)
        eta = 2 / (t + 2.0)

        state_vec = y + beta * (z-b)
        adjoint_closure = lambda u: adjoint_constraint_primitive(u, state_vec)
        num_iters = int(np.ceil((t+1)**(1/4) * np.log(n)))

        # compute primal update direction via randomized Lanczos
        eigen_val, eigen_vec = approx_min_eigen(
                objective_primitive, adjoint_closure, n, num_iters
        )

        # update state variables
        z = (1-eta) * z + eta * alpha * constraint_primitive(eigen_vec)
        gamma = np.clip((4 * alpha**2) / (((t + 2)**(3/2)) * (np.linalg.norm(z - b)**2)), 0, 1)
        y = y + gamma * (z-b)
        S = (1-eta) * S + eta * alpha * (eigen_vec[:,None]
                                         @ (eigen_vec[None,:] @ Omega))

        #print(eigen_vec)
        #print(constraint_primitive(eigen_vec))
        #print(z)
        #print(obj_val)
        #print()

        # for tracking
        infeas = z - b
        obj_val = (1-eta) * obj_val \
                + eta * alpha * np.dot(eigen_vec, objective_primitive(eigen_vec))
        dual_gap = (obj_val + np.dot(y + beta * (infeas), z).squeeze() - eigen_val * alpha).squeeze()
        sub_opt = dual_gap - np.dot(y, infeas) \
                - 0.5*beta*(np.linalg.norm(infeas)**2)

        if t % 100 == 0:
            print('t = ', t)
            print('infeas = ', np.linalg.norm(infeas))
            print('obj val = ', obj_val)
            print('dual gap = ', dual_gap)
            print('sub opt = ', sub_opt)
            print()

    # reconstruct matrix
    U, Lambda = reconstruct(Omega, S)

    # trace correction
    Lambda_tr_correct = Lambda + (alpha - np.sum(Lambda)) / R

    X_hat_1 = (U * Lambda_tr_correct[None, :]) @ U.T

    X_hat_2 = S @ pinv(Omega.T @ S) @ S.T
    
    embed()
    exit()

    return X_hat_2
    

if __name__ == '__main__':

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

    # fast maxcut
    R = 250
    T = int(2e4)
    X_fast = solve_maxcut_sketchyfast(laplacian, R, T)

    embed()
    exit()

    #cut_size = (pred_cut[None,:] @ laplacian @ pred_cut[:, None]).item() / 4
