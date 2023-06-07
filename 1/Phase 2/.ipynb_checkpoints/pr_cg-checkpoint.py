import numpy as np
from typing import Callable
from backtracking import backtracking


def pr_cg(
        f: Callable,
        grad: Callable,
        x0: np.ndarray,
        x_star: np.ndarray,
        eps=1e-6,
        max_iter=1e4,
        alpha_0: float = 1,
        rho: float = 0.5,
        c: float = 1e-4
) -> tuple:
    """This function implements the linear conjugate gradient algorithm using backtracking line search.

    :param f: function used as objective
    :param grad: gradient of the function
    :param x0: starting point
    :param x_star: solution of the objective
    :param eps: necessary stopping criterion
    :param max_iter: failsafe if criterion is never met
    :param alpha_0: step length used for backtracking
    :param rho: used for backtracking
    :param c: used for backtracking

    :return: final x_k, norm of grad_k, norm of x_k - x_star, the iteration count i
    """
    x_k = x0
    grad_k = grad(x_k)
    p_k = -grad_k
    grad_k_norm = np.linalg.norm(grad_k)

    i = 0
    while grad_k_norm > eps:

        if i == max_iter:
            print("Max iteration reached!")
            return x_k, np.array(grad_k_norm), np.array(np.linalg.norm(x_k - x_star)), np.array(i)

        alpha = backtracking(f, grad_k, x_k, p_k, alpha_k=alpha_0, rho=rho, c=c)
        x_k = x_k + alpha * p_k
        grad_k_next = grad(x_k)
        beta_k = (grad_k_next.T @ (grad_k_next - grad_k)) / np.linalg.norm(grad_k)**2
        p_k = -grad_k_next + beta_k * p_k
        grad_k = grad_k_next

        i += 1
        grad_k_norm = np.linalg.norm(grad_k)

    return x_k, np.array(grad_k_norm), np.array(np.linalg.norm(x_k - x_star)), np.array(i)


if __name__ == '__main__':
    import funcs as fc
    'Problems 1-3)'
    print(pr_cg(fc.rosenbrock, fc.rosenbrock_gradient, np.array([1.2, 1.2]), np.array([1., 1.])))
    print(pr_cg(fc.rosenbrock, fc.rosenbrock_gradient, np.array([-1.2, 1.]), np.array([1., 1.])))
    print(pr_cg(fc.rosenbrock, fc.rosenbrock_gradient, np.array([0.2, 0.8]), np.array([1., 1.])))

    'Problems 4-6)'
    print(pr_cg(fc.nl_f, fc.nl_f_grad, np.array([-0.2, 1.2]), np.array([0., 1.])))
    print(pr_cg(fc.nl_f, fc.nl_f_grad, np.array([3.8, 0.1]), np.array([4., 0.])))
    print(pr_cg(fc.nl_f, fc.nl_f_grad, np.array([1.9, 0.6]), np.array([0., 1.])))
