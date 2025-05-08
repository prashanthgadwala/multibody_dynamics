import numpy as np
import matplotlib.pyplot as plt


def semi_implicit_euler(M, f, t0, r0, v0, h, tf):
    """
    Simulates the motion of a particle system using the explicit Euler scheme (Algorithm 1).
    Args:
        M (function): Mass matrix with signature M(r).
        f (function): Force vector with signature f(t, r, v).
        t0 (float): Initial time.
        r0 (np.array): Initial position vector.
        v0 (np.array): Initial velocity vector.
        h (float): Time step.
        tf (float): Final time of simulation.
    Returns:
        tuple: A tuple containing lists of time, position, and velocity vectors at each time step.
    """
    # Allocate memory for saving the solution
    N = int((tf - t0) / h)  # Number of time steps
    nt = N + 1              # Number of time nodes
    nr = len(r0)            # Number of position coordinates
    nv = len(v0)            # Number of velocity coordinates
    t = np.zeros(nt)        # Array of time instants
    r = np.zeros((nt, nr))  # Array of positions
    v = np.zeros((nt, nv))  # Array of velocities

    # Set initial conditions
    t[0] = t0
    r[0] = r0
    v[0] = v0

    # Begin the simulation for loop
    for k in range(N):
        # Update time
        t[k + 1] = t[k] + h
        # Update velocity using explicit Euler
        a_k = np.linalg.solve(M(r[k]), f(t[k], r[k], v[k]))
        v[k + 1] = v[k] + h * a_k
        # Update position using implicit Euler
        r[k + 1] = r[k] + h * v[k + 1]
    return t, r, v

if __name__ == '__main__':
    # Simulation of elastic pendulum from Section 3.2

    # Define physical parameters of the simulation
    m = 1.0     # Mass (kg)
    c = 100.0   # Spring stiffness (N/m)
    L = 1.0     # Undeformed length of the spring (m)
    g = 9.81    # Gravitational acceleration (m/s^2)

    # Define initial conditions
    t0 = 0                      # Initial time
    r0 = np.array([1.0, 0.0])  # Initial position
    v0 = np.array([0.0, 0.0])   # Initial velocity

    # Define mass matrix and force vector
    def M(r):
        return np.diag([m, m])
    
    def f(t, r, v):
        norm_r = np.linalg.norm(r)
        direction = r / norm_r
        return - c * (norm_r - L) * direction + np.array([0, - m * g])

    # Define the simulation parameters
    tf = 10                 # Final time (s)
    h = 1e-2                # Time step (s)

    # Simulate using both methods
    t, r, v = semi_implicit_euler(M, f, t0, r0, v0, h, tf)

    # Compute lengths
    lengths_expl = np.linalg.norm(r, axis=1)

    # Plot the trajectories on the same plot
    plt.figure(figsize=(10, 5))
    plt.plot(r[:, 0], r[:, 1], label='SI Euler', linestyle='-')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.gca().set_aspect('equal')
    plt.title('Trajectory of the Elastic Pendulum')
    plt.legend()
    plt.grid(True)

    # Plot the lengths over time on the same plot
    plt.figure(figsize=(10, 5))
    plt.plot(t, lengths_expl, label='SI Euler', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Length (m)')
    plt.title('Length of the Elastic Pendulum Over Time')
    plt.legend()
    plt.grid(True)

    plt.show()

    ###########
    # Animation
    import matplotlib.animation as animation
    
    fps=10
    N_frames = (tf - t0) * fps
    frac = int(np.ceil(len(t) / N_frames))
    r_frames = r[::frac]
    N_frames = r_frames.shape[0]

    def update(frame, r_frames, pend, ax):
        # Update the line of the explicit Euler
        pend.set_data([0, r_frames[frame, 0]], [0, r_frames[frame, 1]])
        return pend, 

    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(-2*L, 2*L)
    ax.set_ylim(-2*L, 2*L)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    ax.set_title('Elastic Pendulum Motion (Semi-implicit Euler)')
    ax.grid(True)

    # Initialize a line plot: starting location of the pendulum
    pend, = ax.plot([0, r_frames[0, 0]], [0, r_frames[0, 1]], marker='o', label='expl. Euler')
    
    ax.legend()

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=N_frames, interval=1000/fps, fargs=( r_frames, pend, ax),
        blit=True
    )

    # Display the animation
    plt.show()