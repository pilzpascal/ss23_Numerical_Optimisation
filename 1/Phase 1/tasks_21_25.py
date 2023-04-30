import numpy as np
import scipy
from scipy.linalg import hilbert

from methods import steepest_descent


def task_21(n: int):
    q = hilbert(n).astype(np.longfloat)
    b = np.ones(n, dtype=np.longfloat)

    def func(x: np.ndarray):
        return 0.5 * x @ q @ x - b @ x

    def grad(x: np.ndarray):
        return x @ q - b

    return func, grad


if __name__ == "__main__":
    sizes = [5, 8, 12, 20, 30]

    for size in sizes:
        start = np.zeros(size, dtype=np.longfloat)
        f, grad_f = task_21(size)
        x_fin, p_fin, a_fin, n_iter = steepest_descent(start, f, grad_f,
                                                       start_a=np.longfloat(10),
                                                       stop_crit=1e-2)

        print('=' * 70)
        print()
        print(f'n={size}')
        print(f'It took {n_iter} iterations.')
        true_min = scipy.optimize.minimize(f, start).x.astype(int)
        my_min = x_fin[-1]
        print(f'true min: {true_min}')
        print(f'my endpoint: {my_min}')
        print(f'norm of fin: {np.linalg.norm(my_min)}')
        print(f'distance to true min: {np.linalg.norm(true_min - my_min)}')
        eigen_vals, eigen_vec = np.linalg.eig(hilbert(size))
        eigen_vals_sorted = sorted(eigen_vals, reverse=True)
        print(f'eigenvalues: {eigen_vals_sorted}')
        print(f'{(eigen_vals_sorted[-1]-eigen_vals_sorted[0])/(eigen_vals_sorted[-1]+eigen_vals_sorted[0])}')
        print(f'condition number: {eigen_vals_sorted[-1]/eigen_vals_sorted[0]}')
