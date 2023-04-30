import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import approximate_taylor_polynomial as taylor

from methods import steepest_descent, newtons_method


def target_1(t):
    return np.sin(t * 2 * np.pi)


def target_2(t):
    return np.sin(t * 2 * np.pi)


def target_3(t):
    return np.sin(t * 2 * np.pi) + np.sin(t * np.pi)


def target_4(t):
    return np.sin(t * 5 * np.pi) + np.sin(t * 2 * np.pi) + np.sin(t * np.pi)


def target_5(t):
    return np.sin(t * 2 * np.pi) + np.exp(t) * np.cos(t * 3 * np.pi)


if __name__ == '__main__':
    target = target_1
    q = 1
    m = 10
    n = 3
    start = np.zeros(n + 1).astype(np.longfloat)
    x_points = np.linspace(-q, q, m).astype(np.longfloat)
    y_points = target(x_points).astype(np.longfloat)
    c = np.vander(x_points, N=n+1, increasing=True).astype(np.longfloat)

    def f(x: np.ndarray):
        r = c @ x - y_points
        return 0.5 * np.sum(r**2)

    def grad_f(x: np.ndarray):
        r = c @ x - y_points
        return r @ c

    def hess_f(x: np.ndarray):
        return c.T @ c


    x_fin, p_fin, a_fin, n_iter = steepest_descent(start, f, grad_f, print_interval=500, stop_crit=1e-2)
    # x_fin, p_fin, a_fin, n_iter = newtons_method(start, f, grad_f, hess_f, print_interval=10)
    coefficients = x_fin[-1][::-1]
    final_func = np.poly1d(coefficients)

    print()
    print('='*30)
    print()
    print(f'It took {n_iter} iterations.')
    print(f'Final squared error is {0.5 * np.sum((final_func(x_points) - y_points)**2)}.')

    plt.plot(np.linspace(-q, q, 1000), final_func(np.linspace(-q, q, 1000)), color='orange',
             label='approximation', linewidth=1.5)
    plt.plot(np.linspace(-q, q, 1000), taylor(target, 0, n, q)(np.linspace(-q, q, 1000)), color='green',
             label='taylor expansion according to scipy', linewidth=0.5, alpha=0.7)
    plt.scatter(x_points, y_points, s=8)
    plt.title(f'q={q}, m={m}, n={n}')
    plt.legend()
    plt.savefig('plot')
    plt.show()
