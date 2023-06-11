import numpy as np
import scipy
import funcs as fc


def dogleg(H, g, B, trust_radius):
    pb = -H @ g  # full newton step
    norm_pb = np.linalg.norm(pb)

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


def sr1_trust_region(f, grad, hess, x_0, x_star, max_iter=1e4, eps=1e-3, eta=1e-6, r=1e-8):
    def ineq_constraint(x: np.ndarray):
        return - np.linalg.norm(x) + trust_k

    def subproblem(s):
        return grad_k.T @ s + 0.5 * s.T @ B_k @ s

    x_k = x_0
    f_k = f(x_k)
    grad_k = grad(x_k)
    B_k = hess(x_k)
    trust_k = trust_region(f, grad, hess, x_k)[1][-1]

    i = 0
    while np.linalg.norm(grad_k) > eps:
        if i == max_iter:
            print("Max iteration reached!")
            return x_k, np.array(np.linalg.norm(grad_k)), np.array(np.linalg.norm(x_k - x_star)), np.array(i)

        s_0 = np.zeros_like(x_0)
        constraint1 = {"type": "ineq", "fun": ineq_constraint}
        s_k = scipy.optimize.minimize(subproblem, s_0, constraints=constraint1).x

        y_k = grad(x_k + s_k) - grad_k
        ared = f_k - f(x_k + s_k)
        pred = -(grad_k.T @ s_k + 0.5 * s_k.T @ B_k @ s_k) + 1e-200

        if (ared / pred) > eta:
            x_k = x_k + s_k
            grad_k = grad(x_k)

        if (ared / pred) > 0.75:
            if np.linalg.norm(s_k) <= 0.8 * trust_k:
                trust_k = trust_k
            else:
                trust_k = 2 * trust_k
        elif 0.1 <= (ared / pred) <= 0.75:
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

    return x_k, np.array(np.linalg.norm(grad_k)), np.array(np.linalg.norm(x_k - x_star)), np.array(i)


if __name__ == "__main__":
    print(sr1_trust_region(fc.rosenbrock, fc.rosenbrock_gradient, fc.rosenbrock_hessian, np.array([1.2, 1.2]),
                           np.array([1., 1.])))

    print(sr1_trust_region(fc.rosenbrock, fc.rosenbrock_gradient, fc.rosenbrock_hessian, np.array([-1.2, 1.]),
                           np.array([1., 1.])))

    print(sr1_trust_region(fc.nl_f, fc.nl_f_grad, fc.nl_f_hess, np.array([3.8, 0.1]), np.array([4, 0]), max_iter=10000))
