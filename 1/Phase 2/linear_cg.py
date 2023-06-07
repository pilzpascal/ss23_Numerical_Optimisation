import numpy as np
from typing import Callable
from hilbert_matrix import hilbert_mat


def linear_cg(
        residual: Callable,
        x_0: np.ndarray,
        x_star: np.ndarray,
        dim: int,
        eps=1e-6,
        max_iter=1e4,
) -> tuple:
    """This function implements the linear conjugate gradient algorithm using backtracking line search.

    :param residual: gradient of the function
    :param x_0: starting point
    :param x_star: solution of the objective
    :param dim: dimension used for the hilbert matrix
    :param eps: necessary stopping criterion
    :param max_iter: failsafe if criterion is never met
    :return: final x_k, norm of grad_k, norm of x_k - x_star, the iteration count i
    """

    x_k = x_0
    r_k = residual(x_k)
    p_k = -r_k
    Q = hilbert_mat(dim)

    i = 0
    while np.linalg.norm(r_k) > eps:

        if i == max_iter:
            print("Max iteration reached!")
            return x_k, np.array(np.linalg.norm(r_k)), np.array(np.linalg.norm(x_k - x_star)), np.array(i)

        alpha_k = (r_k.T @ r_k) / (p_k.T @ Q @ p_k)
        x_k = x_k + alpha_k * p_k
        r_k_new = r_k + alpha_k * Q @ p_k
        beta_k = (r_k_new.T @ r_k_new) / (r_k.T @ r_k)
        p_k = -r_k_new + beta_k * p_k
        r_k = r_k_new

        i += 1

    return x_k, np.array(np.linalg.norm(r_k)), np.array(np.linalg.norm(x_k - x_star)), np.array(i)


if __name__ == '__main__':
    import funcs as fc

    print(linear_cg(fc.l_f1_grad, np.zeros(shape=5), np.array([5., -120., 630., -1120.00000001, 630.]), 5))
    print(linear_cg(fc.l_f2_grad, np.zeros(shape=8), np.array(
        [-8.00000052e+00, 5.04000025e+02, -7.56000030e+03, 4.62000015e+04, -1.38600004e+05, 2.16216005e+05,
         -1.68168004e+05, 5.14800010e+04]), 8))
    print(linear_cg(fc.l_f3_grad, np.zeros(shape=12),
                    np.array([-1.28758021e+01, 1.82700490e+03, -6.35497518e+04, 9.48423728e+05,
                              -7.55539458e+06, 3.58352883e+07, -1.07176657e+08, 2.07239527e+08,
                              -2.58446725e+08, 2.00601048e+08, -8.81030636e+07, 1.67194346e+07]), 12))
    print(linear_cg(fc.l_f4_grad, np.zeros(shape=20),
                    np.array([-3.06198188e+01, 5.79631108e+03, -2.67656855e+05, 5.30590627e+06,
                              -5.62934738e+07, 3.56180768e+08, -1.41454296e+09, 3.54689896e+09,
                              -5.26881741e+09, 3.31570124e+09, 2.23492076e+09, -4.62144407e+09,
                              -4.79904240e+08, 5.96741325e+09, -4.19120773e+09, -4.68358103e+08,
                              1.17802999e+09, 4.13333400e+08, -7.15516924e+08, 1.98562765e+08]), 20))
    print(linear_cg(fc.l_f5_grad, np.zeros(shape=30),
                    np.array([-9.82482439e+00, 1.83726227e+03, -8.40808229e+04, 1.63655657e+06,
                              -1.67132563e+07, 9.84342322e+07, -3.42619774e+08, 6.54384034e+08,
                              -3.64246837e+08, -1.23608428e+09, 3.33110919e+09, -4.11395369e+09,
                              3.61640475e+09, -3.04010905e+09, 2.03383571e+09, -6.12820729e+08,
                              5.54105091e+08, -1.97634656e+09, 2.17304903e+09, 2.29862444e+08,
                              -1.61735957e+09, -5.59274698e+08, 1.42743422e+09, 9.15511460e+08,
                              -4.43132977e+08, -2.73444446e+09, 2.35525831e+09, 5.29596440e+08,
                              -1.25096008e+09, 3.87527105e+08]), 30))
