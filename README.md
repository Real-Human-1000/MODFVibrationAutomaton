Multi-Degree-of-Freedom Vibrational Automaton

The problem: animate a point (i.e. construct a trajectory) with no additional input beyond the initial conditions.
The theory roughly goes like this:
A trajectory (a path through space over time) can be described as a continuous function.
Continuous functions can be approximated (or equally represented, in the limit) as a Fourier Series (a combination of sine/cosine waves).
A harmonic oscillator (mass-spring system) moves as a sine/cosine wave of a certain natural frequency, depending on the component values and initial conditions.
A system with multiple masses and springs will have many different modes of vibration, which will express themselves as wave generators of different frequencies.
Ergo, can we choose the component values and initial conditions such that the right frequencies and phases are generated to construct a Fourier Series approximation of a desired trajectory for one of the masses?
Turns out, the answer is yes.

This project was originally written in MATLAB, but I moved it to Python in order to have better image manipulation tools and to make it more accessible to normies.

I tentatively store information in the MultiSystem class in the interest of organization, but to the extent that I actually use MultiSystem, it can be ignored.
The main script is render.py, which has options for creating a video like the one I put on YouTube demonstrating the component values and trajectory.
The desired system properties can be optimized for using findSystem.py, which uses a CMA-ES algorithm package to very quickly find system solutions.
My original MATLAB code basically just used random optimization and it was literally like 400 times slower because I was being dumb. Now it's better and I will probably use this package in other projects as a nonlinear nonconvex optimization solution because it seems to work really well.
I tried so hard to find a nice algebraic solution for this problem, but that turns it into this nasty nonlinear symbolic system that's even worse.

The basic theory of mass-spring systems can be found in any Vibrations or Dynamic Systems course, but the system can be represented as a matrix equation:

M * xdd + K * x = 0

Where the matrices M and K depend on the system. If M is diagonal, the system is said to be without dynamic coupling, and if K is diagonal, it is said to be without static coupling.
For this project, M is diagonal and K is not diagonal, but is symmetric and has a very regular structure on the off-diagonals.
The natural frequencies are the values w which satisfy the equation:

det|-w^2 * M + K| = 0

They can also be found as the eigenvalues of M^-1 * K.
This is useful because the eigenvectors of M^-1 * K are the modes of vibration. If they are represented as unit column vectors, they can be horizontally stacked to create a matrix P.
This matrix P is neat because it can decouple the system response, allowing us to transform the system into a collection of individual equations that are much easier to deal with:

x = P * y

P^T * M * P * ydd + P^T * K * P * y = 0

Md * y + Kd * y = 0

md_i * ydd_i + kd_i * y_i = 0

At this point, you can use P to decouple the desired trajectory into individual solutions and reconstruct the response to calculate the proper initial conditions.
This is trivial to do, which is good because it's been like months since I actually did it for the Amogus trajectory so I don't remember how.

Possible extensions to this project include using a bigger system (for more detailed trajectories), introducing more coupling for more natural trajectories (maybe we can make the Amogus with only like 5 masses instead of 14), introducing damping for a more complex transient response, and using nonlinear components (all you chief, I aint doing that)

If you, for some ungodly reason, decide to adopt this project, best of luck to you and I am open to answering questions only if you prove yourself to be worthy.
