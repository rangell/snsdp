import cvxpy as cp
import numpy as np
from scipy import io
from scipy.sparse import csgraph, csc_matrix, coo_matrix

from IPython import embed


def solve_maxcut_slow(L: csc_matrix) -> np.array:
    n = L.shape[0]
    X = cp.Variable((n, n), PSD=True)
    prob = cp.Problem(cp.Maximize(cp.trace(L @ X)),
                      [cp.diag(X) == np.ones((n,))])
    prob.solve(solver=cp.SCS, verbose=True)
    return 2*(np.linalg.svd(X.value)[0][:,0] > 0).astype(float) - 1


if __name__ == '__main__':

    #MATFILENAME = 'G1.mat'
    #matobj = io.loadmat(MATFILENAME)
    #graph = matobj['Problem'][0][0][1]

    # create synthetic example for testing
    row = np.array([0, 0, 1, 1, 2, 3])
    col = np.array([1, 2, 2, 3, 4, 4])
    #row = np.array([0, 0])
    #col = np.array([1, 2])
    data = np.ones_like(row)
    graph = coo_matrix((data, (row, col)), shape=(5, 5)).tocsc()
    graph = graph + graph.T
    laplacian = csgraph.laplacian(graph, normed=False).tocsc()

    pred_cut = solve_maxcut_slow(laplacian)
    cut_size = (pred_cut[None,:] @ laplacian @ pred_cut[:, None]).item() / 4

    embed()
    exit()

