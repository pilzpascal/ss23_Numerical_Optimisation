import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
import scipy


def get_local_minima_1d(f: sp.Function, which: str = 'min') -> np.ndarray:
    """
    Takes a sympy function and returns all local minima
    :param f: sympy function
    :param which: is in ['min', 'max', 'all']. Determines whether to return minima, maxima, or all stationary points
    :return: np.ndarray of local minima
    """
    x = sp.symbols('x')
    diff = sp.diff(f, x)
    sec_diff = sp.diff(diff, x)
    try:
        zeros = sp.solve(diff, x)
    except NotImplementedError as e:
        raise NotImplementedError('The function cannot be solved using sympy', e)
    zeros = [elem for elem in zeros if elem.is_real]
    if which == 'min':
        output = [elem for elem in zeros if sec_diff.evalf(subs={x: elem}) > 0]
    elif which == 'max':
        output = [elem for elem in zeros if sec_diff.evalf(subs={x: elem}) < 0]
    elif which == 'all':
        output = zeros
    else:
        raise ValueError("Argument 'which' is not in ['min', 'max', 'all']")
    return np.array(output)


def get_alpha(f: callable(np.ndarray),
              x_k: np.ndarray,
              p_k: np.ndarray,
              grad_f) -> float:
    """
    performs backtracking line search to find an admissible alpha
    :param f: function in question
    :param x_k: current iterate
    :param p_k: chosen direction
    :param grad_f: the gradient of the function in question
    :return: scalar alpha_k; how far to go along the direction p_k
    """
    c = 0.5
    rho = 0.9

    alpha = np.longfloat(1)
    while f(x_k + alpha * p_k) > f(x_k) + c * alpha * np.dot(grad_f(x_k), p_k):
        alpha *= rho
    return alpha


def plot_iterations(f: callable(float),
                    points: np.ndarray,
                    true_min: np.ndarray,
                    directions: np.ndarray = None,
                    show_dir: bool = False,
                    show_path: bool = False,
                    save: bool = False,
                    save_name: str = 'fig') -> None:
    """
    Function that takes a numeric function going from R^1 to R^1.
    It produces a line plot showing the given points (usually iterates).

    :param f: the numeric function from R^1 to R^1
    :param points: list of points of the shape (m,)
    :param true_min: the true minimum
    :param directions: the direction taken at each iterate; shape is (m-1,)
    :param show_dir: whether to show the directions taken at each iterate. This is bugged, not recommended
    :param show_path: whether to plot or not to plot the path
    :param save: whether to save the figure or not. Would be saved as 'fig.PNG'
    :param save_name: what name to save the figure under, default is 'fig'. If save is not true then this does nothing
    :return:
    """
    mul = 0.4   # this defines many 'ranges' (of the data) outside the range of the data to plot
    num = 100   # this defines how many points to use for plotting the function

    if points.shape[1] == 1:  # case of 1d function
        x_min = min(points)
        x_max = max(points)
        x_range = max(abs(x_max - x_min), 2)    # max() makes sure something gets plotted in case starting value is goal

        x_list = np.linspace(x_min - mul * x_range, x_max + mul * x_range, num)
        y = np.apply_along_axis(f, 0, x_list)

        fig, ax = plt.subplots()
        ax.plot(x_list, y)

        plt.title('Contour plot and iterates')
        func_y = np.apply_along_axis(f, 0, points)
        ax.scatter(points, func_y, color='orange', s=15, label='iterates')
        ax.scatter(points[0], func_y[0], color='blue', marker='x', s=80, label='starting point')
        ax.scatter(points[-1], func_y[-1], color='green', marker='|', s=400, label='end point')
        ax.scatter(true_min, f(true_min), color='red', marker='x', s=80, label='true minimum', alpha=0.7)

    elif points.shape[1] == 2:  # case of 2d function
        x_min = min(points[:, 0])
        x_max = max(points[:, 0])
        x_range = max(abs(x_max - x_min), 2)    # max() makes sure something gets plotted in case starting value is goal
        y_min = min(points[:, 1])
        y_max = max(points[:, 1])
        y_range = max(abs(y_max - y_min), 2)    # max() makes sure something gets plotted in case starting value is goal

        x_list = np.linspace(x_min - mul * x_range, x_max + mul * x_range, num)
        y_list = np.linspace(y_min - mul * y_range, y_max + mul * y_range, num)
        x, y = np.meshgrid(x_list, y_list)
        data = np.stack([x, y])
        z = np.apply_along_axis(f, 0, data)

        fig, ax = plt.subplots()
        cp = ax.contourf(x, y, z)
        fig.colorbar(cp)

        if show_path:
            for i in range(len(points)-1):
                plt.plot(points[i:i + 2, 0], points[i:i + 2, 1], 'k-')

        if show_dir:
            # the scaling here is off and I don't want to invest the time figure out a solution
            ax.quiver(points[:-1, 0], points[:-1, 1], directions[:, 0], directions[:, 1])

        plt.title('Contour plot and iterates')
        ax.scatter(points[:, 0], points[:, 1], color='orange', s=10, label='iterates')
        ax.scatter(points[0, 0], points[0, 1], color='blue', marker='x', s=80, label='starting point')
        ax.scatter(points[-1, 0], points[-1, 1], color='green', marker='x', s=80, label='end point')
        ax.scatter(true_min[0], true_min[1], color='red', marker='x', s=80, label='true minimum', alpha=0.7)

    else:   # not 1d or 2d
        raise ValueError('Given list of points is not 1d or 2d.')

    plt.legend()
    ax.grid()
    plt.show()
    if save:
        plt.savefig(save_name)


def print_output(f: callable(np.ndarray),
                 x: np.ndarray,
                 local_minima_precise: np.ndarray,
                 p: np.ndarray = None,
                 alpha: np.ndarray = None,
                 full: bool = False) -> np.ndarray:
    """
    Prints the given arrays
    :param f: the function under investigation
    :param x: list of points
    :param local_minima_precise: list of local minima
    :param p: list of directions
    :param alpha: list of step sizes
    :param full: whether to print full list of points or only the last
    :return:
    """
    # this expression returns the nearest value in a numpy array to a certain given value
    if local_minima_precise is None:
        local_minima_precise = [0]
    local_minima = [np.longfloat(elem) for elem in local_minima_precise]
    idx = (np.abs(local_minima - x[-1])).argmin()
    minimum = local_minima[idx]

    print('='*70)
    print()
    print(f'Function value at endpoint {f(x[-1])} (euclidian norm of {np.linalg.norm(f(x[-1])):.4f}).')
    print(f'It took {len(x)-1} iterations.')
    print()
    print(f'Precise endpoint:    {x[-1]}')
    print(f'True value would be: {minimum}')
    print(f'All local minima are {local_minima}')
    print(f'(Precise local minima are: {local_minima_precise})')
    print(f'Distance from last iterate to true value: {np.format_float_scientific(np.linalg.norm(minimum - x[-1]), precision=6)}')
    print()
    print('='*70)
    print()

    if full:
        print(f'List of x: {x}')
    if p:
        print(f'List of p: {p}')
    if alpha:
        print(f'List of alpha: {alpha}')

    return np.array(minimum)
