import numpy as np
from methods import newtons_method


def task_1() -> (callable, callable, callable):
    def func(x: np.ndarray):
        return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2

    def grad(x: np.ndarray):
        return np.array([400*x[0]*(x[0]**2-x[1]) + 2*(x[0]-1), 200*(x[1]-x[0]**2)])

    def hess(x: np.ndarray):
        return np.array([[1200*x[0]**2 - 400*x[1] + 2, -400*x[0]], [-400*x[0], 200]])

    return func, grad, hess


def task_2() -> (callable, callable, callable):
    def func(x: np.ndarray):
        return (2*x[0]**2 - 8*x[1])**2 + (x[0] - 1)**2

    def grad(x: np.ndarray):
        return np.array([16*x[0]**3 + x[0]*(2-64*x[1]) - 2,
                         128*x[1] - 32*x[0]**2])

    def hess(x: np.ndarray):
        return np.array([[48*x[0]**2 - 64*x[1] + 2, -64*x[0]],
                         [-200*x[0]*x[1], 128]])

    return func, grad, hess


def task_3() -> (callable, callable, callable):
    def func(x: np.ndarray):
        return (10*x[0]**2 - ((5/2)*x[1]))**2 + (x[0] + 1)**2

    def grad(x: np.ndarray):
        return np.array([400*x[0]**3 + (2 - 100*x[1])*x[0] + 2,
                         (25*x[1] - 100*x[0]**2)/2])

    def hess(x: np.ndarray):
        return np.array([[1200*x[0]**2 - 100*x[1] + 2, -100*x[0]],
                         [-100*x[0], 12.5]])

    return func, grad, hess


def task_4() -> (callable, callable, callable):
    def func(x: np.ndarray):
        return (10*x[0] - 0.01*x[1]**2)**2 + (x[1] - 100)**2

    def grad(x: np.ndarray):
        return np.array([(1_000*x[0] - x[1]**2)/5,
                         (x[1]**3 + (5_000 - 1000*x[0])*x[1] - 500_000)/2_500])

    def hess(x: np.ndarray):
        return np.array([[200, -(2*x[1])/5],
                         [-(2*x[1])/5, (3*x[1]**2 - 1_000*x[0] + 5_000)/2_500]])

    return func, grad, hess


def task_5() -> (callable, callable, callable):
    def func(x: np.ndarray):
        return (100 - x[0])**2 + (x[1] - x[0]**2)**2

    def grad(x: np.ndarray):
        return np.array([4*x[0]**3 + (2-4*x[1])*x[0] - 200, 2*(x[1] - x[0]**2)])

    def hess(x: np.ndarray):
        return np.array([[12*x[0]**2 - 4*x[1] + 2, -4*x[0]],
                         [-4*x[0], 2]])

    return func, grad, hess


if __name__ == '__main__':
    start = np.array([0, 0]).astype(np.longfloat)
    stop = 1e-6

    f, grad_f, hess_f = task_5()

    x_fin, p_fin, a_fin, n_iter = newtons_method(start, f, grad_f, hess_f, stop_crit=1e-18, print_interval=1000)
    result = x_fin[-1]
    true_min = np.array([100, 10_000])     # TODO: change this for every task!!

    print(f'final x: {result}')
    print(f'norm f(x): {np.linalg.norm(f(result))}')
    print(f'distance to true min: {np.format_float_scientific(np.linalg.norm(true_min - result), precision=6)}')
    print(f'iterations: {n_iter}')

