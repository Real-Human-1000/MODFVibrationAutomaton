import os
import cma
from matplotlib import pyplot as plt
import numpy as np
from MultiSystem import MultiSystem

# Find a 1D system with the desired natural frequencies

# Mass values are generally 2 orders of magnitude smaller than spring constants
# It might help the optimizer if we adjusted the masses in the optimization function to match sensitivity
mass_adj_val = 0.1

def sum_squares(x):
    return np.inner(x, x)

def construct_M(masses):
    # Construct a mass matrix for a 1D linear system
    # Looks like |-w-0-w-0-w-0-w-| (wall-spring-mass-spring-mass-spring-mass-spring-wall)
    # masses should be a normal list or 1D numpy array
    return np.diag(masses)

def construct_Minv(masses):
    # Construct a mass matrix for a 1D linear system
    # Looks like |-w-0-w-0-w-0-w-| (wall-spring-mass-spring-mass-spring-mass-spring-wall)
    # masses needs to be a 1D numpy array
    return np.diag(1/masses)

def construct_K(constants):
    # Construct a stiffness matrix for a 1D linear system
    # Looks like |-w-0-w-0-w-0-w-| (wall-spring-mass-spring-mass-spring-mass-spring-wall)
    # constants should be a normal list or 1D numpy array
    return np.diag(constants[:-1]) + np.diag(constants[1:]) - np.diag(constants[1:-1], k=1) - np.diag(constants[1:-1], k=-1)

def calc_freqs(N, sys_var):
    global mass_adj_val
    # Construct a system and evaluate its natural frequencies
    # sys_var is a 1D numpy array
    # The first N entries of sys_var are the mass values
    # The next N+1 entries of sys_var are the spring constants

    # See the nifty tricks
    # https://cma-es.github.io/cmaes_sourcecode_page.html#practical

    # I don't have a great methodology for this, but since I don't know how to use bound constraints I'm just gonna abs everything
    Minv = construct_Minv(mass_adj_val * np.abs(sys_var[:N]))
    K = construct_K(np.abs(sys_var[N:]))
    eigvals = np.linalg.eigvals(Minv @ K)
    # eigvals is *probably* sorted, but numpy doesn't make any guarantees
    # Might be good to not use quicksort then because it can have poor performance on sorted lists...
    return np.sort(eigvals)


def evaluate_system(N, sys_var, goal):
    # Compare the goal natural frequencies to those of the system defined by sys_var
    # The order of the natural frequencies doesn't really matter
    # Need this to be as fast as possible
    # sys_var is a 1D numpy array
    # The first N entries of sys_var are the mass values
    # The next N+1 entries of sys_var are the spring constants

    candidate = calc_freqs(N, sys_var)

    error = sum_squares(candidate - np.sort(np.abs(goal)))
    return error


def findSystem1D(freqs):
    # Find a 1D system with the specified natural frequencies
    N = len(freqs)

    es = cma.CMAEvolutionStrategy((N+N+1) * [0], 0.5)
    while not es.stop():
        solutions = es.ask()  # Get state of variables (list of flat np arrays)
        es.tell(solutions, [evaluate_system(N, x, freqs) for x in solutions])  # Provide the evaluations to the optimizer
        es.logger.add()  # write data to disc to be plotted
        es.disp()
    es.result_pretty()
    cma.plot()  # shortcut for es.logger.plot()
    plt.savefig(os.path.join("outcmaes", "plot.png"))

    print("\nBest result:")
    final_sys_var = es.result[0]
    print(calc_freqs(N, final_sys_var))
    M = construct_M(mass_adj_val * np.abs(final_sys_var[:N]))
    K = construct_K(np.abs(final_sys_var[N:]))
    return M, K


def calcInitial1D(sys, waves):
    # Calculate the initial conditions required for a specific mass in a 1D system to trace a specified trajectory
    # The trajectory must be formatted as a Fourier series
    # sys is a MultiSystem with (hopefully) enough frequencies to match waves
    # waves is a nx3 numpy array with columns (amplitude, frequency, phase)
    # The frequencies required by waves MUST be present in the system already to get a good result!!!

    # Calculate modes
    D, P = np.linalg.eig(np.linalg.inv(sys.M) @ sys.K)
    # D is the eigenvalues, P is the eigenvector matrix

    # Remap waves to match system frequencies
    remap = np.zeros(sys.n, dtype=np.int16)
    for i in range(sys.n):
        diff = np.abs(np.pow(waves[:, 1], 2) - D[i])
        I = np.argmin(diff)
        if diff[I] > 1e-2:
            print("Response might not be possible for this system")
        remap[i] = I

    remapped_amps = waves[remap, 0]
    remapped_freqs = waves[remap, 1]
    remapped_phases = waves[remap, 2]
    """
                                           [ a1 ]
    c1*a1 + c2*a2 + c3*a3 = [ c1 c2 c3 ] * [ a2 ]
                                           [ a3 ]
    """
    scaled_amps = 1 / P[3, :].T * remapped_amps

    # Solving for required initial conditions of the decoupled system
    # x0 = A * cos(phase) (function at t=0)
    # v0 = -w * A * sin(phase) (derivative at t=0)
    y0 = np.zeros((sys.n, 1))
    ydot0 = np.zeros((sys.n, 1))
    for i in range(sys.n):
        y0[i, 0] = scaled_amps[i] * np.cos(remapped_phases[i])
        ydot0[i, 0] = -remapped_freqs[i] * scaled_amps[i] * np.sin(remapped_phases[i])

    # Compute true system ICs
    x0 = P @ y0
    v0 = P @ ydot0
    return x0, v0


if __name__ == "__main__":
    # Amogus something
    # freqs = np.pi * np.array([
    #     2, 6, 4, 8, 10, 14, 12
    # ])

    # 1% Magnitude Amogus X
    # freqs = np.pi * np.array([
    #     2, 4, 6, 8, 10, 12, 14, 18, 20, 22, 24, 26, 28, 30, 34, 36, 38,
    # ])

    # 1% Magnitude Amogus Y
    # freqs = np.pi * np.array([
    #     2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 34,
    # ])

    x_waves = np.array([
        [0.3322, 2 * np.pi, 0.5650],
        [0.1233, 6 * np.pi, -0.4538],
        [0.1025, 4 * np.pi, -1.6065],
        [0.0778, 8 * np.pi, 0.9244],
        [0.0706, 10 * np.pi, 2.2807],
        [0.0332, 14 * np.pi, -2.6604],
        [0.0178, 12 * np.pi, -2.0449],
        [0.0106, 28 * np.pi, -0.4678],
        [0.0104, 30 * np.pi, -1.4298]
    ])

    sysX = MultiSystem(
        n=7,
        M=np.diag([3.0978, 4.7058, 0.2062, 0.3439, 18.8495, 3.2583, 4.9459]),
        B=np.zeros([7, 7]),
        K=10 ** 3 * np.array([
            [3.6019, -1.9407, 0, 0, 0, 0, 0],
            [-1.9407, 1.9760, -0.0353, 0, 0, 0, 0],
            [0, -0.0353, 0.1149, -0.0796, 0, 0, 0],
            [0, 0, -0.0796, 0.2657, -0.1861, 0, 0],
            [0, 0, 0, -0.1861, 2.2605, -2.0743, 0],
            [0, 0, 0, 0, -2.0743, 4.6377, -2.5634],
            [0, 0, 0, 0, 0, -2.5634, 5.2983]
        ]),
        x0=np.zeros([1,7]).T, #np.array([0.2263, -0.1049, 0.2320, 0.3510, 0.4655, -1.3546, 0.0236]).T,
        v0=np.zeros([1,7]).T #np.array([-13.7577, 12.4859, 12.6717, -0.7816, -6.5906, 40.1957, 7.8644]).T
    )

    x0, v0 = calcInitial1D(sysX, x_waves)
    print("x0 and v0:")
    print(x0)
    print(v0)
