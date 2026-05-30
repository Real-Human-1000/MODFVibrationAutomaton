import os
import cma
from matplotlib import pyplot as plt
import numpy as np

# Find a system with the desired natural frequencies

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
    # Construct a system and evaluate its natural frequencies
    # sys_var is a 1D numpy array
    # The first N entries of sys_var are the mass values
    # The next N+1 entries of sys_var are the spring constants

    # See the nifty tricks
    # https://cma-es.github.io/cmaes_sourcecode_page.html#practical

    # Mass values are generally 2 orders of magnitude smaller than spring constants
    # It might help the optimizer if we adjusted the masses in the optimization function to match sensitivity
    mass_adj_val = 0.1

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


if __name__ == "__main__":
    # Amogus something
    # freqs = np.array([
    #     2, 6, 4, 8, 10, 14, 12
    # ])

    # 1% Magnitude Amogus X
    freqs = np.array([
        2, 4, 6, 8, 10, 12, 14, 18, 20, 22, 24, 26, 28, 30, 34, 36, 38,
    ])
    # Perfect solution: (masses need to be adjusted)
    # [-1.77706544e+01 - 2.98801012e+01  9.21998686e+00  2.03813773e+00
    #  - 7.79515947e-01  1.21671753e-01 - 7.46861284e-03 - 1.15597134e-03
    #  7.25425869e-06 - 6.02529176e-07 - 9.44343617e-06  1.99232677e-03
    #  - 7.15731190e-02  7.67973484e-01  3.78578229e+00 - 7.41805086e+00
    #  1.80301097e+01 - 2.29724367e+01 - 1.69831066e+01  1.74565276e+01
    #  4.42852727e+00  6.83860508e-01 - 3.07035783e-01 - 1.19823778e-02
    #  2.79923816e-03  1.55758206e-05  3.23482184e-07 - 1.71997383e-07
    #  1.85001279e-05  2.08233450e-03  1.54923462e-01 - 1.26579023e+00
    #  - 6.62869682e+00 - 7.07380080e+00  3.95850138e+01]
    # 1% Magnitude Amogus Y
    # freqs = np.array([
    #     2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 34,
    # ])
    # Perfect solution: (masses needs to be adjusted)
    # [-3.11020809e+00 - 2.73352983e+00 - 1.31198012e+00 - 2.30371616e-01
    #  - 2.31981173e-01  2.55657198e-02  1.23926930e-02 - 2.77974217e-03
    #  - 6.65041926e-04 - 5.86858843e-05  2.47464807e-05 - 7.38167900e-06
    #  - 9.05467234e-05 - 2.16001618e-03 - 3.83823315e-01  1.09526682e+01
    #  4.39124935e+00 - 1.99400472e+00 - 1.50919919e+00  3.23034392e-01
    #  - 1.56875489e-01  4.55583620e-02 - 1.72783843e-02 - 3.99661258e-03
    #  - 1.17072621e-03 - 7.75808950e-05  3.85885563e-05 - 1.67652937e-06
    #  1.21139575e-05 - 1.06743030e-04 - 3.99471000e-03  6.52223190e-01
    #  - 1.47531180e+01]

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
    print(es.result[0])
    print(calc_freqs(N, es.result[0]))

