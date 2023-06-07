import numpy as np
from numpy.linalg import norm, eigh, solve, cholesky
from numpy import maximum, diag


def function_1() -> (callable, callable, callable):
    """
    Call this function to get the first function, its gradient, and its hessian.
    :return: func, grad, hess
    """
    def func(x: np.ndarray) -> np.ndarray:
        """100*(x[1] - x[0]**2)**2 + (1 - x[0])**2"""
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
        """150*(x[0]*x[1])**2 + (0.5*x[0] + 2*x[1] - 2)**2"""
        return 150*(x[0]*x[1])**2 + (0.5*x[0] + 2*x[1] - 2)**2

    def grad(x: np.ndarray) -> np.ndarray:
        return np.array([300*x[1]**2*x[0] + 0.5*x[0] + 2*x[1] - 2, (300*x[0]**2 + 8)*x[1] + 2*x[0] - 8])

    def hess(x: np.ndarray) -> np.ndarray:
        return np.array([[300*x[1]**2 + 0.5, 600*x[0]*x[1] + 2],
                         [600*x[0]*x[1] + 2, 300*x[0]**2 + 8]])

    return func, grad, hess


def eig_val_mod(hessian):
    """
    Modifying the hessian by doing an eigenvalue decomposition and modifying the eigenvalues until it is positive
    definite. This is done by simply flipping the negative eigenvalues

    :param hessian: hessian to be modified
    :return: mod_hess, the modified hessian
    """
    eig_vals, eig_vecs = eigh(hessian)
    mod_eig_vals = maximum(eig_vals, -eig_vals)
    mod_hess = (eig_vecs @ diag(mod_eig_vals)) @ eig_vecs.T
    return mod_hess


def add_id(hessian, beta: float = 1e-3):
    """
    Modifying the hessian by adding a small multiple of the identity. As seen in Algorithm 3.3
    :param hessian: the hessian to be modified
    :param beta: 'hyperparamter' of this algorithm
    :return: L@L.T, where L is the result of the successful cholesky decomposition
    """
    n = len(hessian)
    identity = np.eye(n)

    if min(diag(hessian)) > 0:
        tau = 0
    else:
        tau = -min(diag(hessian)) + beta

    while True:
        try:
            L = cholesky(hessian + identity * tau)
            return L @ L.T.conj()
        except np.linalg.LinAlgError:
            tau = max(2 * tau, beta)
            

def no_mod(hessian):
    """
    This just returns the hessian as it is. Like an identity function.
    :param hessian:
    :return: hessian
    """
    return hessian


def newtons_method_hessian_mod(start_point: np.ndarray,
                               f: callable,
                               grad_f: callable,
                               hess_f: callable,
                               stop_crit: float = 1e-6,
                               print_interval: int = False,
                               hess_modifier: callable = add_id) -> (np.ndarray, np.ndarray, np.ndarray):

    
    x_k = start_point.copy()
    x_list = [x_k]
    p_list = list()
    a_list = list()
    i = 0

    while norm(grad_f(x_k)) > stop_crit:
        mod_hess = hess_modifier(hess_f(x_k))
        p_k = solve(mod_hess.astype(float), (-grad_f(x_k).astype(float))).reshape(-1)
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
    start_points = {'function_1': np.array([[1.2, 1.2], [-1.2, 1], [0.2, 0.8]]),
                    'function_2': np.array([[-0.2, 1.2], [3.8, 0.1], [1.9, 0.6]])}
    functions = {'function_1': function_1(),
                 'function_2': function_2()}

    for function in functions:
        f, grad_f, hess_f = functions[function]
        print('='*70)
        print(f'{function}: {f.__doc__}')
        print()
        for x_0 in start_points[function]:
            print('-'*60)
            print(f'starting point: {x_0}')
            print()
            x_fin, p_fin, a_fin, n_iter = newtons_method_hessian_mod(
                x_0, f, grad_f, hess_f, stop_crit=1e-6, print_interval=False, hess_modifier=eig_val_mod)
            result = x_fin[-1]
            print(f'iterations:   {n_iter}')
            print(f'norm of grad: {norm(grad_f(result))}')
            print(f'endpoint:     {result}')
            print()
