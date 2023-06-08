import numpy as np
from numpy.linalg import norm
from scipy.optimize import line_search
import warnings

def backtracking_line_search(f, grad, xk, pk, alpha=1.0, rho=0.9, c=0.1):
    while f(xk + alpha * pk) > f(xk) + c * alpha * np.dot(grad(xk), pk):
        alpha *= rho
    return alpha

def bfgs_method(f, grad, x0, max_iter=100, epsilon=1e-6, _rho=0.75, c=0.0001):
    """Quasi-Newton method implementation."""
    xk = np.array(x0)
    gk = grad(xk)

    Hk = np.eye(len(xk))  # Initial Hessian approximation as identity matrix

    iter_count = 0
    while np.linalg.norm(gk) > epsilon and iter_count < max_iter:

        pk = -np.dot(Hk, gk)
        
        # Perform line search
        alpha = backtracking_line_search(f, grad, xk, pk, rho=_rho, c=c)
        
        # Update point, gradient, and function value
        xk_prev, gk_prev = xk.copy(), gk.copy()
        xk = xk + alpha * pk
        gk = grad(xk)
        
        # Compute difference in x and gradient
        sk = xk - xk_prev
        yk = gk - gk_prev
        
        # Check for linear dependence
        if np.abs(np.dot(yk, sk)) < 1e-10: break

        # BFGS update for Hessian approximation
        rho = 1.0 / np.dot(yk, sk)
        A = np.eye(len(xk)) - rho * np.outer(sk, yk)
        B = np.eye(len(xk)) - rho * np.outer(yk, sk)
        
        # Check for zero denominator in BFGS update
        if np.isnan(A).any() or np.isnan(B).any(): break
        
        Hk = np.dot(A, np.dot(Hk, B)) + rho * np.outer(sk, sk)
        
        iter_count += 1
    
    return xk, norm(grad(xk)), iter_count

def sr1_method(f, grad, x0, max_iter=100, epsilon=1e-6, _rho=0.75, c=0.0001):
    """Quasi-Newton method implementation with SR1 update."""
    xk = np.array(x0)
    gk = grad(xk)

    Hk = np.eye(len(xk))  # Initial Hessian approximation as identity matrix

    iter_count = 0
    alpha = 0
    while np.linalg.norm(gk) > epsilon and iter_count < max_iter:

        pk = -np.dot(Hk, gk)

        # Perform line search
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alpha_old = alpha
            alpha = backtracking_line_search(f, grad, xk, pk, rho=_rho, c=c)

        if alpha is None:
            alpha = alpha_old

        # Update point, gradient, and function value
        xk_prev, gk_prev = xk.copy(), gk.copy()
        xk = xk + alpha * pk
        gk = grad(xk)

        # Compute difference in x and gradient
        sk = xk - xk_prev
        yk = gk - gk_prev
        
        # Check for linear dependence
        if np.abs(np.dot(yk, sk)) < 1e-10: break

        # SR1 update for Hessian approximation
        if np.abs(np.dot(sk, yk)) > np.finfo(float).eps:
            A = sk - np.dot(Hk, yk)
            
            # Check for zero denominator in BFGS update
            if np.isnan(A).any(): break
            
            Hk += np.outer(A, A) / np.dot(A, yk)

        iter_count += 1

    return xk, norm(grad(xk)), iter_count
    
