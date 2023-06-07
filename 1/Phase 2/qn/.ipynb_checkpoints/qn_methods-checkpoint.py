import numpy as np
from scipy.optimize import minimize

from optimization import bfgs_method, sr1_method
from problems import rosenbrock, rosenbrock_grad, rosenbrock_starting_points
from problems import function_2, function_2_grad, function_2_starting_points

def perform_QN(method, f, grad, starting_points, max_iter=100, epsilon=1e-6):
    for i, sp in enumerate(starting_points):
        print(f"Solving Problem-{i+1}, Starting Point: {sp}")
        x0 = np.array(sp)
        
        x_opt, gnorm, num_iters = method(f, grad, x0, max_iter, epsilon)
        
        print("Optimal solution:")
        print(f"x = {x_opt}")
        print(f"norm of grad = {gnorm}")
        print(f"Number of iterations: {num_iters}\n")
        
def perform_QN_scipy(f, starting_points, max_iter=100, epsilon=1e-6):
    for i, sp in enumerate(starting_points):
        print(f"Solving Problem-{i+1}, Starting Point: {sp}")
        x0 = np.array(sp)
        
        result = minimize(f, x0, method='BFGS', options={'maxiter': max_iter, 'gtol': epsilon})

        print("Optimal solution:")
        print(f"x = {result.x}")
        print(f"f(x) = {result.fun}")
        print(f"Number of iterations: {result.nit}\n")

print("<Quasi Newton Method for the Rosenbrock function>")
print("BFGS"+"-"*50)
perform_QN(bfgs_method, rosenbrock, rosenbrock_grad, rosenbrock_starting_points)
print("SR1-" + "-"*50)
perform_QN(sr1_method, rosenbrock, rosenbrock_grad, rosenbrock_starting_points)

print("\n\n")

print("<Quasi Newton Method for the second Function>")
print("BFGS"+"-"*50)
perform_QN(bfgs_method, function_2, function_2_grad, function_2_starting_points)
print("SR1-" + "-"*50)
perform_QN(sr1_method, function_2, function_2_grad, function_2_starting_points)

#perform_QN_scipy(rosenbrock, rosenbrock_starting_points)