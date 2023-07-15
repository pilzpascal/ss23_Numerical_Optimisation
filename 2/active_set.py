"""
This implementation of the active set algorithm is heavily inspired by Jim Varanelli's (GitHub: JimVaranelli)
implementation found at https://github.com/JimVaranelli/ActiveSet
Most of the methods have been adapted to our specific use.
"""

import numpy as np
import warnings
from problems import get_problems
from utils import rnd_vec_l1_norm


class ActiveSet:
    def __init__(self, tol=1e-8):
        self.tol = tol

    def _init_vars(self):
        self.neq = 0                    # number of equality constraints
        self.nineq = 0                  # number of inequality constraints
        self.G = np.ndarray             # objective function quadratic coefficient matrix
        self.c = np.ndarray             # objective function linear coefficient vector
        self.A = np.ndarray             # constraint coefficient matrix
        self.b = np.ndarray             # constraint target vector
        self.act_idx = np.ndarray         # indices of active constraints

    def _prep_inputs(self, G, c, A, b, neq, nineq):
        # initialize instance variables
        self._init_vars()
        # objective coefficients
        self.G = G
        self.c = c
        # constraint coefficients
        self.A = A
        self.b = b
        self.neq = neq
        self.nineq = nineq

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

    def _solve_as_kkt(self, x):
        # check if there are no constraints active
        g = self._calc_grad(x)
        n_active = len(self.act_idx)
        if n_active == 0:
            p = np.linalg.solve(self.G, -g)
            return p, []
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
        # solve the KKT system
        pk_lambda = np.linalg.solve(left_side, right_side)
        # return new direction and lagrangian multipliers
        return pk_lambda[:self.G.shape[1]].flatten(), pk_lambda[self.G.shape[1]:]

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
        self._prep_inputs(G, c, A, b, neq, nineq)
        if not self._is_feasible(x0):
            raise ValueError(f'ActiveSet: supplied x0 {x0} is infeasible')
        self._init_active_set(x0)
        cur_x = x0
        n_iterations = 0
        while np.any(abs(self._calc_grad(cur_x)) > stop_crit):
            p, l = self._solve_as_kkt(cur_x)
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
    """
    problems = get_problems()
    for problem in problems:
        M = problem['M']
        y = problem['y']
        m, n = M.shape
        G_prob = M.T @ M
        c_prob = M.T @ y
        A_prob = -np.ones(n)
        b_prob = -np.ones(m)

        x0_inp = rnd_vec_l1_norm(n)
        beta_plus = np.maximum(x0_inp, 0)
        beta_minus = np.macimum(-x0_inp, 0)
    """
    active_set = ActiveSet()
    results = active_set.run(G=np.array([[2, 0], [0, 2]]),
                             c=np.array([-4, -4]),
                             A=np.array([[1, 1], [1, -2], [-1, -1], [-2, 1]]),
                             b=np.array([2, 2, 1, 2]),
                             x0=np.array([0.5, 1]),
                             neq=0,
                             nineq=4)
    print(results)
