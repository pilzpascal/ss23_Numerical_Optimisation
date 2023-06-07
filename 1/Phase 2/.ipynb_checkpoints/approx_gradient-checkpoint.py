import numpy as np

def rosenbrock_function(x: np.ndarray) -> float:
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def function1(x: np.ndarray) -> float:
    return 150 * (x[0] * x[1])**2 + (0.5 * x[0] + 2 * x[1] - 2)**2

def approximate_gradient(f: callable, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    gradient = []
    for i in range(len(x)):
        unit_vector = np.zeros_like(x)
        unit_vector[i] = 1
        grad_i = (f(x + eps * unit_vector) - f(x)) / eps
        gradient.append(grad_i)
    return np.array(gradient)

def approximate_hessian(f: callable, x:np.ndarray, eps: float = (1e-6)) -> np.ndarray:
    hessian = []
    p = np.ones_like(x)
    p[0] = 0
    p = eps * p

    O_p = (f(x + eps * p) - f(x)) / eps

    for i in range(len(x)):
        row = []
        for j in range(len(x)):
            unit_vector_i, unit_vector_j = np.zeros_like(x), np.zeros_like(x)
            unit_vector_i[i] = 1
            unit_vector_j[j] = 1
            grad_ij = (f(x + eps * unit_vector_i + eps * unit_vector_j) - f(x + eps * unit_vector_i - eps * unit_vector_j) -\
                       f(x - eps * unit_vector_i + eps * unit_vector_j) + f(x - eps * unit_vector_i - eps * unit_vector_j)) / (4 * eps**2)
            row.append(grad_ij)
        hessian.append(row)
    return np.array(hessian)


if __name__ == '__main__':
    rosenbrock_function_sp = np.array([[1.2, 1.2], [-1.2, 1], [0.2, 0.8]])
    function1_sp = np.array([[-0.2, 1.2], [3.8, 0.1], [1.9, 0.6]])

    for i in range(3):
        print(f"Rosenbrock-Function at {rosenbrock_function_sp[i]}: {rosenbrock_function(rosenbrock_function_sp[i])}")
        print(f"Gradient of Rosenbrock-Function at {rosenbrock_function_sp[i]}: {approximate_gradient(rosenbrock_function, rosenbrock_function_sp[i])}")
        print(f"Hessian of Rosenbrock-Function at {rosenbrock_function_sp[i]}:\n{approximate_hessian(rosenbrock_function, rosenbrock_function_sp[i])}")
        print(f"Function1 at {function1_sp[i]}: {function1(function1_sp[i])}")
        print(f"Gradient of Function1 at {function1_sp[i]}: {approximate_gradient(function1, function1_sp[i])}")
        print(f"Hessian of Function1 at {function1_sp[i]}:\n{approximate_hessian(function1, function1_sp[i])}")
        print()