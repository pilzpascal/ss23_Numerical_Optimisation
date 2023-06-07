import numpy as np
from typing import Callable


def backtracking(f: Callable,
                 grad_k: np.ndarray,
                 x_k: np.ndarray,
                 p_k: np.ndarray,
                 max_iter: float = 1e5,
                 alpha_k: float = 1,
                 rho: float = 0.5,
                 c: float = 1e-4) -> float:
    """This function implements the backtracking algorithm 3.1.

    :param f: function to use when checking for stopping criterion
    :param grad_k: gradient of the function at a certain point
    :param x_k: point where line search is performed
    :param p_k: descent direction
    :param max_iter: failsafe if the stopping criterion is not met
    :param alpha_k: step length to be updated
    :param rho: scaling factor for each update step
    :param c: constant needed for stopping criterion
    :return: optimal alpha_k
    """

    i = 0

    while f(x_k + alpha_k * p_k) > (f(x_k) + c*alpha_k*(grad_k.T @ p_k)):
        if i == max_iter:
            print("Max iteration reached!")
            return alpha_k
        alpha_k *= rho
        i += 1

    return alpha_k
