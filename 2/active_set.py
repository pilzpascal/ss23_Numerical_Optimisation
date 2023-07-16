"""
This implementation of the active set algorithm is heavily inspired by Jim Varanelli's (GitHub: JimVaranelli)
implementation found at https://github.com/JimVaranelli/ActiveSet
Most of the methods have been adapted to our specific use.
"""

import numpy as np
import warnings
from problems import get_problems
from utils import rnd_vec_l1_norm
from utils import transform_problem_to_qp_form


class ActiveSet:
    def __init__(self, tol=1e-8):
        self.tol = tol

    def _init_vars(self, G, c, A, b, neq, nineq, x0):
        self.neq = neq              # number of equality constraints
        self.nineq = nineq          # number of inequality constraints
        self.G = G                  # objective function quadratic coefficient matrix
        self.c = c                  # objective function linear coefficient vector
        self.A = A                  # constraint coefficient matrix
        self.b = b                  # constraint target vector
        self._init_active_set(x0)

    def _init_active_set(self, x0):
        # equality constraints always in active set.
        # save the indices into the full matrix.
        # add any inequality constraints to the
        # active working set that are equal to the
        # constraint value, i.e.
        #   x>=0 is active if x=0.
        cx = np.dot(self.A, x0)
        self.act_idx = np.ndarray((0, 1))     # creates an empty array of shape (0,1)
        for j in range(cx.shape[0]):
            if j < self.neq or np.isclose(cx[j], self.b[j], rtol=0, atol=self.tol):
                self._add_active_constraint(j)

    def _calc_objective(self, x):
        return 0.5 * x @ self.G @ x + x @ self.c

    def _calc_grad(self, x):
        return np.dot(self.G, x) + self.c

    def _is_feasible(self, x):
        Ax = np.dot(self.A, x)
        # check equality constraints
        if self.neq:
            if not np.allclose(Ax[:self.neq], self.b[:self.neq], rtol=0, atol=self.tol):
                return 0
        # check inequality constraints
        icm = Ax[self.neq:] <= self.b[self.neq:] + self.tol
        if len(icm[icm == 0]):
            return 0
        return 1

    def _add_active_constraint(self, idx):
        if idx == -1:
            return
        # save the constraint index
        self.act_idx = np.vstack((self.act_idx, np.asarray(idx).reshape(1, 1)))

    def _remove_active_constraint(self, idx):
        # remove the constraint index
        self.act_idx = np.delete(self.act_idx, idx, axis=0)

    def _build_kkt_system(self, x):
        n_active = len(self.act_idx)
        g = self._calc_grad(x)
        # construct the KKT system
        active_cons = self.A[self.act_idx.astype(int)].reshape(-1, self.A.shape[1])
        # constructing the left side
        left_side = np.zeros(shape=(self.G.shape[0] + n_active, self.G.shape[1] + n_active))
        left_side[:self.G.shape[0], :self.G.shape[1]] = self.G
        left_side[self.G.shape[0]:, :self.A.shape[1]] = active_cons
        left_side[:self.A.shape[1], self.G.shape[0]:] = active_cons.T
        # constructing the right side
        right_side = np.zeros(shape=(left_side.shape[0], 1))
        right_side[:self.G.shape[1]] = -g.reshape(-1, 1)
        return left_side, right_side

    def _solve_as_kkt(self, x):
        g = self._calc_grad(x)
        if len(self.act_idx) == 0:
            p = np.linalg.solve(self.G, -g)
            return p, []
        # check if there are no constraints active
        left_side, right_side = self._build_kkt_system(x)
        # solve the KKT system
        pk_lambda = np.linalg.solve(left_side, right_side)
        # return new direction and lagrangian multipliers
        return pk_lambda[:self.G.shape[1]].flatten(), pk_lambda[self.G.shape[1]:]

    def _null_space_as_kkt(self, x):
        # check if there are no constraints active
        g = self._calc_grad(x)
        m = len(self.act_idx)
        if m == 0:
            p = np.linalg.solve(self.G, -g)
            return p, []
        left_side, right_side = self._build_kkt_system(x)
        # number of variables
        n = self.A.shape[1]
        # active constraint vector is set to zero
        b = np.zeros(shape=(m, 1))
        # calculate active set Hessian vector
        h = self._calc_as_Hessian_vector(x)
        # use QR decomposition of active constraint matrix (transpose)
        Q, R = np.linalg.qr(self.AC_KKT[:self.H.shape[0], self.H.shape[1]:], 'complete')
        # trim full R to get n x n upper diagonal form
        R = R[:m]
        # calculate the step direction vector:
        #   p = Qr * py + Qn * pz
        py = np.linalg.solve(R, b)
        # split Q into null and range spaces
        Qr = Q[:, :m]
        p = np.dot(Qr, py)
        # compute null space portion if the number of variables
        # is greater than the number of active constraints
        if n > m:
            Qn = Q[:, m:]
            Qnt = Qn.T
            # compute Cholesky decomposition for reduced Hessian
            Hz = np.linalg.multi_dot([Qnt, self.H, Qn])
            L = np.linalg.cholesky(Hz)
            # compute null space target vector
            hz = np.dot(Qnt, np.linalg.multi_dot([self.H, Qr, py]) + h)
            pz = np.linalg.solve(L.T, np.linalg.solve(L, hz))
            # update step direction
            p += np.dot(Qn, pz)
        # compute Lagrangians
        l = np.linalg.solve(R, np.dot(Qr.T, np.dot(self.H, -p) + h))
        return p, l

    def _solve_using_numpy(self, x):
        left_side, right_side = self._build_kkt_system(x)
        pk_lambda, res, rank, s = np.linalg.lstsq(left_side, right_side)
        return pk_lambda[:self.G.shape[1]].flatten(), pk_lambda[self.G.shape[1]:]

    def _calc_dir(self, x):
        try:
            p, l = self._solve_as_kkt(x)
        except:
            p, l = self._solve_using_numpy(x)
            # p, l = self._null_space_as_kkt(x)
        return p, l

    def _calc_step_length(self, x, p):
        # constraint indices
        cons_idx = np.arange(self.A.shape[0])
        # get inactive constraints
        mask = np.isin(cons_idx, self.act_idx, assume_unique=True, invert=True)
        inactive_cons = self.A[mask]
        inactive_targ = self.b[mask]
        # keep track of constraint matrix indices
        inactive_cons_idx = cons_idx[mask]
        # denominator
        den = np.dot(inactive_cons, p).flatten()
        # suppress division-by-zero warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            alphas = (inactive_targ - np.dot(inactive_cons, x)) / den
        # keep quotient when den > 0
        alphas = alphas[den > self.tol]
        # check if empty
        if len(alphas) == 0:
            return 1, -1
        inactive_cons_idx = inactive_cons_idx[den > self.tol]
        min_idx = np.argmin(alphas)
        return min(1, alphas[min_idx]), inactive_cons_idx[min_idx]

    def run(self, G, c, A, b, x0, neq, nineq, stop_crit=1e-6):
        """
        Performs the active set method to find the minima of a quadratic program with linear constraints.
        :param G: coefficient matrix of quadratic part of problem
        :param c: coefficient vector of linear part of problem
        :param A: coefficient matrix for (linear) constraints
        :param b: target vector for constraints
        :param x0: beginning point
        :param neq: number of equality constraints
        :param nineq: number of inequality constraints
        :param stop_crit: stopping criterion for the gradient of the objective function
        :return:
        """
        self._init_vars(G, c, A, b, neq, nineq, x0)
        if not self._is_feasible(x0):
            raise ValueError(f'ActiveSet: supplied x0 {x0} is infeasible')

        cur_x = x0
        n_iterations = 0
        while np.any(abs(self._calc_grad(cur_x)) > stop_crit):
            p, l = self._calc_dir(cur_x)
            len_p = np.linalg.norm(p)
            if np.isclose(len_p, 0, rtol=0, atol=self.tol):
                if l.shape[0] == self.neq:
                    break
                min_l_ineq = np.amin(l[self.neq:])
                if min_l_ineq >= -self.tol:
                    break
                min_l_ineq_idx = np.where(l == min_l_ineq)[0][0]
                self._remove_active_constraint(min_l_ineq_idx)       # remove constraint with lowest lagrange multiplier
            else:
                alpha, const_idx = self._calc_step_length(cur_x, p)
                if np.isclose(alpha, 0, rtol=0, atol=self.tol):
                    break
                self._add_active_constraint(const_idx)
                cur_x = cur_x + alpha * p
            n_iterations += 1
        return cur_x, self._calc_objective(cur_x), n_iterations


if __name__ == '__main__':
    active_set = ActiveSet()
    problem = get_problems()[0]
    G, c, A, b = transform_problem_to_qp_form(problem['M'], problem['y'])
    results = active_set.run(G=G,
                             c=c,
                             A=A,
                             b=b,
                             x0=np.array([0, 0, 0, 0]),
                             neq=0,
                             nineq=4)
    print(results)
