import numdifftools as nd
import numpy as np

import utils


def steepest_descent(f: callable(np.ndarray),
                     start_point: np.ndarray,
                     stop_crit: float = 1e-6) -> (np.ndarray, np.ndarray, np.ndarray):

    grad_f = nd.Gradient(f)
    x_list = [start_point]
    x_k = x_list[-1]
    p_list = list()
    alpha_list = list()

    while not (abs(grad_f(x_k)) < stop_crit).all():

        p_k = -grad_f(x_k)
        p_list.append(p_k)

        alpha = utils.get_alpha(f, x_k, p_k, grad_f)
        alpha_list.append(alpha)

        x_k = x_k + alpha * p_k
        x_list.append(x_k)

    return np.array(x_list), np.array(p_list), np.array(alpha_list)


def newtons_method(f: callable(np.ndarray),
                   start_point: np.ndarray,
                   stop_crit: float = 1e-6) -> (np.ndarray, np.ndarray, np.ndarray):

    grad_f = nd.Gradient(f)
    hess_f = nd.Hessian(f)
    x_list = [start_point]
    x_k = x_list[-1]
    p_list = list()
    alpha_list = list()

    while not (abs(grad_f(x_k)) < stop_crit).all():

        p_k = np.linalg.solve(hess_f(x_k), -grad_f(x_k))
        p_list.append(p_k)

        alpha = utils.get_alpha(f, x_k, p_k, grad_f)
        alpha_list.append(alpha)

        x_k = x_k + alpha * p_k
        x_list.append(x_k)

    return np.array(x_list), np.array(p_list), np.array(alpha_list)
