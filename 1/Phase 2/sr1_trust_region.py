import numpy as np
import scipy
import funcs as fc


def dogleg(H, g, B, trust_radius):
    pb = -H @ g  # full newton step

    # full newton step lies inside the trust region
    if np.linalg.norm(pb) <= trust_radius:
        return pb
    # step along the steepest descent direction
    pu = - (np.dot(g, g) / np.dot(g, B @ g)) * g
    dot_pu = np.dot(pu, pu)
    norm_pu = np.sqrt(dot_pu)
    if norm_pu >= trust_radius:
        return trust_radius * pu / norm_pu

    # solve ||pu**2 +(tau-1)*(pb-pu)**2|| = trusr_radius**2
    pb_pu = pb - pu
    pb_pu_sq = np.dot(pb_pu, pb_pu)
    pu_pb_pu_sq = np.dot(pu, pb_pu)
    d = pu_pb_pu_sq ** 2 - pb_pu_sq * (dot_pu - trust_radius ** 2)
    tau = (-pu_pb_pu_sq + np.sqrt(d)) / pb_pu_sq + 1

    # 0<tau<1
    if tau < 1:
        return pu * tau
    # 1<tau<2
    return pu + (tau - 1) * pb_pu


def trust_region(f, grad, hess, x0, eta=0.15, tol=1e-4, max_trust_radius=100.0):
    xx = []
    # initial point
    x = x0
    r = []
    # initial radius
    trust_radius = 1
    r.append(trust_radius)
    xx.append(x)
    while True:
        g = grad(x)  # gradient
        B = hess(x)  # hessian
        H = np.linalg.inv(B)
        p = dogleg(H, g, B, trust_radius)
        rho = (f(x) - f(x + p)) / (-(np.dot(g, p) + 0.5 * np.dot(p, B @ p)))
        norm_p = np.linalg.norm(p)
        if rho < 0.25:
            trust_radius = 0.25 * norm_p
        else:
            if rho > 0.75 and norm_p == trust_radius:
                trust_radius = min(2.0 * trust_radius, max_trust_radius)
            else:
                trust_radius = trust_radius
        r.append(trust_radius)
        if rho > eta:
            x = x + p
        xx.append(x)
        if np.linalg.norm(g) < tol:
            break
    return np.asarray(xx), r


def sr1_trust_region(f, grad, hess, x_0, max_iter=1e4, eps=1e-3, eta=1e-6, r=1e-8):
    def ineq_constraint(s: np.ndarray):
        return - np.linalg.norm(s) + trust_k

    def subproblem(s):
        return grad_k.T @ s + 0.5 * s.T @ B_k @ s

    x_k = x_0
    grad_k = grad(x_k)
    B_k = np.eye(len(x_k))
    trust_k = trust_region(f, grad, hess, x_k)[1][-1]

    i = 0
    while np.linalg.norm(grad_k) > eps:
        if i == max_iter:
            return x_k, np.array(np.linalg.norm(grad_k)), np.array(i)

        s_0 = np.array([0., 0.])
        constraint1 = {"type": "ineq", "fun": ineq_constraint}
        s_k = scipy.optimize.minimize(subproblem, s_0, constraints=constraint1).x

        y_k = grad(x_k + s_k) - grad_k
        ared = f(x_k) - f(x_k + s_k)
        pred = -(grad_k.T @ s_k + 0.5 * s_k.T @ B_k @ s_k)

        if ared == 0 and pred == 0:
            rho = 1e4
        else:
            rho = ared / pred

        if rho > eta:
            x_k = x_k + s_k
            grad_k = grad(x_k)

        if rho > 0.75:
            if np.linalg.norm(s_k) <= 0.8 * trust_k:
                trust_k = trust_k
            else:
                trust_k = 2 * trust_k
        elif 0.1 <= rho <= 0.75:
            trust_k = trust_k
        else:
            trust_k = 0.5 * trust_k

        intermediate = y_k - B_k @ s_k
        if np.abs(np.dot(s_k, intermediate)) < (r * np.linalg.norm(s_k) * np.linalg.norm(intermediate)):
            pass
        elif intermediate.T @ s_k != 0:
            B_k = B_k + (np.outer(intermediate, intermediate) / np.dot(intermediate, s_k))
        elif (y_k == B_k @ s_k).all():
            pass

        i += 1

    return x_k, np.array(np.linalg.norm(grad_k)), np.array(i)


if __name__ == "__main__":
    from approx_gradient import approximate_gradient, approximate_hessian
    import time
    import math
    # from qn.problems import rosenbrock, rosenbrock_grad
    from funcs import rosenbrock, rosenbrock_gradient, rosenbrock_hessian, nl_f_hess
    from qn.problems import rosenbrock_starting_points
    # print(sr1_trust_region(fc.rosenbrock, fc.rosenbrock_gradient, fc.rosenbrock_hessian, np.array([1.2, 1.2]),
    #                        np.array([1., 1.])))
    #
    # print(sr1_trust_region(fc.rosenbrock, fc.rosenbrock_gradient, fc.rosenbrock_hessian, np.array([-1.2, 1.]),
    #                        np.array([1., 1.])))
    #
    # print(sr1_trust_region(fc.nl_f, fc.nl_f_grad, fc.nl_f_hess, np.array([3.8, 0.1]), np.array([4, 0]), max_iter=10000))
    #

    sol_points_1 = np.array([(1., 1.), (1., 1.), (1., 1.)])
    sol_points_2 = np.array([(4., 0.), (4., 0.), (4., 0.)])
    backtrack_params_rosen = [
        [(0.16, 0.9), (0.5, 0.99), (0.35, 0.99)],
        [(0.03, 0.99), (0.03, 0.99), (0.005, 0.9)]
    ]

    backtrack_params_f2 = [
        [(0.09, 0.99), (0.09, 0.99), (0.01, 0.99)],
        [(0.01, 0.99), (0.04, 0.99), (0.01, 0.99)]
    ]


    def perform_QN(method, f, grad_orig, start, sol, hess_orig=None, max_iter=100, epsilon=1e-6, symbolic_only=False,
                   numeric_only=False, print_dataframe=True, count_time=False, backtrack_params=None):
        qn_dicts = []
        for i, sp in enumerate(start):
            x0 = np.array(sp)

            grad_approx = lambda x: approximate_gradient(f, x)
            hess_approx = lambda x: approximate_hessian(f, x)
            grad_methods = [(grad_orig, False), (grad_approx, True)]
            hess_methods = [(hess_orig, False), (hess_approx, True)]

            if symbolic_only:
                grad_methods = grad_methods[:1]
                hess_methods = hess_methods[:1]

            if numeric_only:
                grad_methods = grad_methods[1:]
                hess_methods = hess_methods[1:]

            for j, ((grad, is_approx), (hess, _)) in enumerate(zip(grad_methods, hess_methods)):
                if backtrack_params is not None:
                    c, rho = backtrack_params[j][i]
                    start_time = time.perf_counter_ns()
                    x_opt, gnorm, num_iters = method(f, grad, x0, max_iter, epsilon, c=c, _rho=rho)
                    end_time = time.perf_counter_ns()
                else:
                    if hess_orig is not None:
                        start_time = time.perf_counter_ns()
                        x_opt, gnorm, num_iters = method(f=f, grad=grad, hess=hess, x_0=x0, max_iter=max_iter, eps=epsilon)
                        end_time = time.perf_counter_ns()
                        print(x_opt)
                    else:
                        start_time = time.perf_counter_ns()
                        x_opt, gnorm, num_iters = method(f, grad, x0, max_iter, epsilon)
                        end_time = time.perf_counter_ns()

                qn_dicts.append({
                    'Gradient': 'Actual Gradient' if not is_approx else 'Approx. Gradient',
                    '$$x_0$$': r'$$\begin{pmatrix}' + r' \\ '.join([str(round(d, 3)) for d in x0]) + r'\end{pmatrix}$$',
                    '$$x_k$$': r'$$\begin{pmatrix}' + r' \\ '.join(
                        [str(round(d, 3)) for d in x_opt]) + r'\end{pmatrix}$$',
                    '$$x*$$': r'$$\begin{pmatrix}' + r' \\ '.join(
                        [str(round(float(d), 3)) for d in sol[i]]) + r'\end{pmatrix}$$',
                    '$$\Vert \nabla f(x_k)\ \Vert$$': np.array(gnorm),
                    '$$\Vert x^* - x_k \Vert$$': np.array(math.dist(x_opt, sol[i])),
                    'Iterations': np.array(num_iters),
                })

                if count_time:
                    qn_dicts[-1]['Execution time'] = f'{str(end_time - start_time)}ns'

        # if print_dataframe:
        #     display(pd.DataFrame(qn_dicts))


    # perform_QN(sr1_trust_region, rosenbrock, rosenbrock_gradient, rosenbrock_starting_points, sol_points_1,
    #            hess_orig=rosenbrock_hessian, epsilon=1e-3)

    print(sr1_trust_region(fc.rosenbrock, fc.rosenbrock_gradient, fc.rosenbrock_hessian, np.array([-1.2, 1]),
                            max_iter=1e4))

    print(sr1_trust_region(fc.nl_f, fc.nl_f_grad, fc.nl_f_hess, np.array([-0.2, 1.2]),
                           max_iter=1e4))
