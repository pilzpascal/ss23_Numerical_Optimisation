import numpy as np
from numpy import exp
from methods import steepest_descent, newtons_method


def task_1() -> (callable, callable, callable):
    def func(x: np.ndarray):
        return (x+1)**3 - (x+1)**2

    def grad(x: np.ndarray):
        return np.array([(x+1) * (x*3 + 1)])

    def hess(x: np.ndarray):
        return np.array([[6*x + 4]])

    return func, grad, hess


def task_2() -> (callable, callable, callable):
    def func(x: np.ndarray):
        return (x - 9) * (3 * (x - 9) ** 3 + 8 * (x - 9) ** 2 + 6 * (x - 9) - 24)

    def grad(x: np.ndarray):
        return np.array([12*x**3 - 300*x**2 + 2496*x - 6936])

    def hess(x: np.ndarray):
        return np.array([[36*x**2 - 600*x + 2496]])

    return func, grad, hess


def task_3() -> (callable, callable, callable):
    def func(x: np.ndarray):
        return -exp(-(x - 0.5) ** 2)

    def grad(x: np.ndarray):
        return np.array([(2*x - 1) * exp(-(x - 0.5) ** 2)])

    def hess(x: np.ndarray):
        return np.array([[-(4*x**2 - 4*x - 1) * exp(-(x - 0.5) ** 2)]])

    return func, grad, hess


def task_4() -> (callable, callable, callable):
    def func(x: np.ndarray):
        return exp(-(x**2 - (x-1)**4))

    def grad(x: np.ndarray):
        return np.array([(4*(x-1)**3 - 2*x) * exp(-(x**2 - (x-1)**4))])

    def hess(x: np.ndarray):
        return np.array([[((12*(x-1)**2 - 2) + (4*(x-1)**3 - 2*x)) * exp(-(x**2 - (x-1)**4))]])

    return func, grad, hess


def task_5() -> (callable, callable, callable):
    def func(x: np.ndarray):
        return exp(-(x-0.4)**4 + (x-0.45)**2)

    def grad(x: np.ndarray):
        return np.array([(2*(x-0.45) - 4*(x-0.4)**3) * exp(-(x-0.4)**4 + (x-0.45)**2)])

    def hess(x: np.ndarray):
        return np.array([[((2 - 12*(x-0.4)**2) + (2*(x-0.45) - 4*(x-0.4)**3)) * exp(-(x-0.4)**4 + (x-0.45)**2)]])

    return func, grad, hess


if __name__ == '__main__':
    start = np.array([0]).astype(np.longfloat)
    stop = 1e-6

    f, grad_f, hess_f = task_2()

    # x_fin, p_fin, a_fin, n_iter = steepest_descent(start, f, grad_f, print_interval=10)
    x_fin, p_fin, a_fin, n_iter = newtons_method(start, f, grad_f, hess_f, print_interval=10)
    result = x_fin[-1][0]
    true_min = -1/3     # TODO: change this for every task!!

    print(f'final x: {result:10.7f}')
    print(f'norm f(x): {np.linalg.norm(f(result))}')
    print(f'distance to true min: {np.format_float_scientific(np.linalg.norm(true_min - result), precision=6)}')
    print(f'iterations: {n_iter}')
