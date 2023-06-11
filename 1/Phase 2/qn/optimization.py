import numpy as np
from numpy.linalg import norm
from backtracking import backtracking


def bfgs_method(f, grad, x0, max_iter=10000, epsilon=1e-5):
    """Quasi-Newton method implementation."""
    xk = np.array(x0)
    gk = grad(xk)

    Hk = np.eye(len(xk))  # Initial inverse Hessian approximation as identity matrix

    iter_count = 0
    while np.linalg.norm(gk) > epsilon and iter_count < max_iter:

        pk = -np.dot(Hk, gk)

        # Perform line search
        alpha = backtracking(f, gk, xk, pk, alpha_k=1)

        # Update point, gradient, and function value
        xk_prev, gk_prev = xk.copy(), gk.copy()
        xk = xk + alpha * pk
        gk = grad(xk)

        # Compute difference in x and gradient
        sk = xk - xk_prev
        yk = gk - gk_prev

        if iter_count == 0:
            Hk = ((yk.T @ sk) / (yk.T @ sk)) * Hk

        div = np.dot(yk, sk)
        # Check for linear dependence
        if np.abs(div) < 1e-100:
            break

        # BFGS update for Hessian approximation
        rho = 1.0 / div
        A = np.eye(len(xk)) - rho * np.outer(sk, yk)
        B = np.eye(len(xk)) - rho * np.outer(yk, sk)

        Hk = np.dot(A, np.dot(Hk, B)) + rho * np.outer(sk, sk)

        iter_count += 1
    
    return xk, norm(grad(xk)), iter_count


def sr1_method(f, grad, x0, max_iter=10000, epsilon=1e-6, _rho=0.75, c=0.0001):
    """Quasi-Newton method implementation with SR1 update."""
    xk = np.array(x0)
    gk = grad(xk)

    Hk = np.eye(len(xk))  # Initial inverse Hessian approximation as identity matrix

    iter_count = 0
    while np.linalg.norm(gk) > epsilon and iter_count < max_iter:

        pk = -Hk @ gk

        alpha = backtracking(f, gk, xk, pk, rho=_rho, c=c)

        # Update point, gradient, and function value
        xk_prev, gk_prev = xk.copy(), gk.copy()
        xk = xk + alpha * pk
        gk = grad(xk)

        # Compute difference in x and gradient
        sk = xk - xk_prev
        yk = gk - gk_prev

        intermediate = sk - np.dot(Hk, yk)
        # SR1 update for Hessian approximation
        if np.abs(np.dot(yk, intermediate)) < (1e-8 * np.linalg.norm(yk) * np.linalg.norm(intermediate)):
            pass
        elif intermediate.T @ sk != 0:
            Hk = Hk + (np.outer(intermediate, intermediate) / np.dot(intermediate, yk))
        elif (sk == (Hk@yk)).all():
            pass

        iter_count += 1

    return xk, norm(grad(xk)), iter_count
