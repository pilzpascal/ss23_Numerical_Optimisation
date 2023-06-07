import numpy as np 

# Define the functions
def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    return np.array([400*x[0]**3 + (2 - 400*x[1])*x[0] - 2, 200*(x[1] - x[0]**2)])

def function_2(x):
    return 150 * (x[0] * x[1]) ** 2 + (0.5 * x[0] + 2 * x[1] - 2) ** 2

def function_2_grad(x: np.ndarray) -> np.ndarray:
        return np.array([300*x[1]**2*x[0] + 0.5*x[0] + 2*x[1] - 2, (300*x[0]**2 + 8)*x[1] + 2*x[0] - 8])

# Define the starting points
rosenbrock_starting_points = [(1.2, 1.2), (-1.2, 1), (0.2, 0.8)]
function_2_starting_points = [(-0.2, 1.2), (3.8, 0.1), (1.9, 0.6)]

# Define solutions
rosenbrock_solutions = [(1,1), (1,1), (1,1)]
function_2_solutions = [(4,0), (4,0), (4,0)]
