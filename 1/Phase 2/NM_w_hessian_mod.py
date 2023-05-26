import numpy as np
from numpy.linalg import norm
from numpy.linalg import solve


def function_1() -> (callable, callable, callable):
    """
    Call this function to get the first function, its gradient, and its hessian.
    :return: func, grad, hess
    """
    def func(x: np.ndarray) -> np.ndarray:
        return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

    def grad(x: np.ndarray) -> np.ndarray:
        return np.array([400*x[0]**3 + (2 - 400*x[1])*x[0] - 2, 200*(x[1] - x[0]**2)])

    def hess(x: np.ndarray) -> np.ndarray:
        return np.array([[1200*x[0]**2 - 400*x[1] + 2, -400*x[0]],
                         [-400*x[0], 200]])

    return func, grad, hess


def function_2() -> (callable, callable, callable):
    """
    Call this function to get the first function, its gradient, and its hessian.
    :return: func, grad, hess
    """
    def func(x: np.ndarray) -> np.ndarray:
        return 150*(x[0]*x[1])**2 + (0.5*x[0] + 2*x[1] - 2)**2

    def grad(x: np.ndarray) -> np.ndarray:
        return np.array([300*x[1]**2*x[0] + 0.5*x[0] + 2*x[1] - 2, (300*x[0]**2 + 8)*x[1] + 2*x[0] - 8])

    def hess(x: np.ndarray) -> np.ndarray:
        return np.array([[300*x[1]**2 + 0.5, 600*x[0]*x[1] + 2],
                         [600*x[0]*x[1] + 2, 300*x[0]**2 + 8]])

    return func, grad, hess


def newtons_method_hessian_mod(start_point: np.ndarray,
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


if __name__ == "__main__":
    start_points = np.array([[[1.2, 1.2], [-1.2, 1], [0.2, 0.8]],
                             [[-0.2, 1.2], [3.8, 0.1], [1.9, 0.6]]])
