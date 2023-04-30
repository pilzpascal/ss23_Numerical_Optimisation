import numpy as np
from numpy.linalg import norm
from numpy.linalg import solve
from utils import get_alpha


def steepest_descent(start_point: np.ndarray,
                     f: callable,
                     grad_f: callable,
                     stop_crit: float = 1e-3,
                     print_interval: int = False,
                     start_a: np.longfloat = 1) -> (np.ndarray, np.ndarray, np.ndarray):

    x_list = [start_point]
    x_k = start_point
    p_list = list()
    a_list = list()
    i = 0

    while norm(grad_f(x_k)) > stop_crit:
        p_k = -grad_f(x_k)
        a_k = get_alpha(f, grad_f, x_k, p_k, start_a)
        x_k += a_k * p_k.reshape(x_k.shape)

        if print_interval and i % print_interval == 0:
            try:
                print(f'Iteration {i:7d}: alpha={a_k:8.7f}, norm_grad={norm(p_k):8.7f}, '
                      f'change_norm_grad={norm(p_list[-1]) - norm(p_k):10.7f}')
            except IndexError:
                print(f'Iteration {i:7d}: alpha={a_k:8.7f}, norm_grad={norm(p_k):8.7f}')
            p_list.append(p_k)
            a_list.append(a_k)
            x_list.append(x_k)
        i += 1

    return np.array(x_list), np.array(p_list), np.array(a_list), i


def newtons_method(start_point: np.ndarray,
                   f: callable,
                   grad_f: callable,
                   hess_f: callable,
                   stop_crit: float = 1e-3,
                   print_interval: int = False) -> (np.ndarray, np.ndarray, np.ndarray):

    x_list = [start_point]
    x_k = start_point
    p_list = list()
    a_list = list()
    i = 0

    while norm(grad_f(x_k)) > stop_crit:
        p_k = solve(hess_f(x_k).astype(float), (-grad_f(x_k).astype(float))).reshape(-1)
        # a_k = get_alpha(f, grad_f, x_k, p_k)
        a_k = 1
        x_k += a_k * p_k

        if print_interval and i % print_interval == 0:
            try:
                print(f'Iteration {i:7d}: alpha={a_k:8.7f}, norm_grad={norm(p_k):8.7f}, '
                      f'change_norm_grad={norm(p_list[-1]) - norm(p_k):10.7f}')
            except IndexError:
                print(f'Iteration {i:7d}: alpha={a_k:8.7f}, norm_grad={norm(p_k):8.7f}')
            p_list.append(p_k)
            a_list.append(a_k)
            x_list.append(x_k)
        i += 1

    return np.array(x_list), np.array(p_list), np.array(a_list), i
