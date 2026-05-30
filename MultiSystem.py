import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import inv
import matplotlib.pyplot as plt
# Class to store a multi-DOF system of springs, masses, and dampers

class MultiSystem:
    def __init__(self, n, M, B, K, x0, v0):
        # M, B, and K represent system property matrices (n-by-n)
        # M is the mass matrix (probably diagonal)
        # B is the damping matrix (probably 0)
        # K is the stiffness matrix (probably symmetric)
        # x0 and v0 represent initial conditions
        # x0 is the initial position (n-by-1)
        # v0 is the initial velocity (n-by-1)
        self.n = n
        self.M = M
        self.B = B
        self.K = K
        self.x0 = x0
        self.v0 = v0

    def get_response(self, t_start, t_stop, t=None):
        # Get the behavior of the system from t_start to t_stop
        # If t is not None, calculate for those points specifically
        # Automatically decides the best step size
        # Returns times (1-by-t vector) and relative positions of masses over time (n-by-t matrix)

        # Mx** + Bx* + Kx = 0
        # x* = x*
        # x** = -B/Mx* - K/Mx

        # x will be structured so that the first n entries are positions and the last n entries are the derivatives
        """
        [  dx1 ]   [     0     0     1     0 ]   [  x1 ]
        [  dx2 ]   [     0     0     0     1 ]   [  x2 ]
        [ ddx1 ] = [ [-K/M -K/M] [-B/M -B/M] ] @ [ dx1 ]
        [ ddx2 ]   [ [-K/M -K/M] [-B/M -B/M] ]   [ dx2 ]
        """

        # trans = np.zeros((self.n*2, self.n*2))
        # trans[0:self.n, self.n:] = np.eye(self.n)
        # trans[self.n:, 0:self.n] = -self.K @ inv(self.M, assume_a='diagonal')
        # trans[self.n:, self.n:] = -self.B @ inv(self.M, assume_a='diagonal')
        # print(trans)
        # # TODO: Be more smart about assuming diagonal M matrix. Are we always assuming that?
        # #  Should we change internal rep to reflect for data savings?
        # # fun = lambda t, x: trans @ x
        # TODO: replace this with a nicer transition matrix so it's faster maybe


        fun = lambda t, x: np.vstack([np.reshape(x[self.n:], [self.n, 1]), -inv(self.M, assume_a='diagonal') @ self.K @ np.reshape(x[:self.n], [self.n, 1])]).flatten()
        sol = solve_ivp(fun, [t_start, t_stop], np.vstack([self.x0, self.v0]).flatten(), method="RK45", dense_output=True, t_eval=t, rtol=1e-8, atol=1e-10)
        print(sol.message)
        return sol.t, sol.y


if __name__ == "__main__":
    # Quick test of storing & solving a system

    sys = MultiSystem(
        2,
        np.array([[2, 0], [0, 1]]),
        np.array([[0.1, 0], [0, 0.1]]),
        np.array([[2, -1], [-1, 2]]),
        np.array([1, 2]).T,
        np.array([0, 0]).T
    )

    t, y = sys.get_response(0, 10)
    plt.plot(t, y[0,:])
    plt.plot(t, y[1,:])

    plt.show()