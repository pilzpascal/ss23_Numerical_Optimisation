import numpy as np
import sympy as sp

import utils
import tasks
from methods import steepest_descent as sd
from methods import newtons_method as nm


if __name__ == '__main__':
    start = np.array([0], dtype=np.longfloat)
    stop = 1e-6

    func_sym = tasks.task_1()
    x_sym = sp.Symbol('x')
    func = sp.lambdify(x_sym, func_sym, 'numpy')

    x, p, alpha = sd(func, start_point=start, stop_crit=stop)
    # x, p, alpha = nm(func, start_point=start, stop_crit=stop)

    local_minima = utils.get_local_minima_1d(func_sym)
    utils.print_output(func, x, local_minima)
    utils.plot_iterations(func, x, show_path=True)
