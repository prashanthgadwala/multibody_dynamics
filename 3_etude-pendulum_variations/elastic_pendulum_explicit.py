import numpy as np
import matplotlib.pyplot as plt


def simulate_elastic_pendulum_explicit_euler(m, c, L, g, t0, r0, v0, h, N):
    """
    Simulates the motion of an elastic pendulum using the explicit Euler scheme (Algorithm 1).
    Args:
        m (float): Mass of the point mass.
        c (float): Spring stiffness.
        L (float): Undeformed length of the spring.
        g (float): Gravitational acceleration.
        t0 (float): Initial time.
        r0 (np.array): Initial position vector [x0, y0].
        v0 (np.array): Initial velocity vector [vx0, vy0].
        h (float): Time step.
        N (int): Total number of time steps.
    Returns:
        tuple: A tuple containing lists of time, position, and velocity vectors at each time step.
    """
    # Allocate memory for saving the solution
    nt = N + 1  # Number of time instants
    nr = 2      # Number of pos. coordinates
    nv = 2      # Number of vel. coordinates
    t = np.zeros(nt)        # Array of time instants
    r = np.zeros((nt, nr))  # Array of positions
    v = np.zeros((nt, nv))  # Array of velocities

    # Set initial conditions
    t[0] = t0
    r[0] = r0
    v[0] = v0

    # Define the gravitational force vector
    F_g = m * g * np.array([0, -1])

    # Begin the simulation for loop
    for k in range(N):
        # Update time
        t[k + 1] = t[k] + h
        # Calculate the spring force and the total force F(rk)
        force_magnitude = c * (np.linalg.norm(r[k]) - L)
        F_s = - force_magnitude * r[k] / np.linalg.norm(r[k])
        Fk = F_s + F_g
        # Update position using explicit Euler
        r[k + 1] = r[k] + h * v[k]
        # Update velocity using explicit Euler
        v[k + 1] = v[k] + h * (Fk/m)
    return t, r, v

if __name__ == '__main__':
    # Define physical parameters of the simulation
    m = 1.0     # Mass (kg)
    c = 10.0   # Spring stiffness (N/m)
    L = 1.0     # Undeformed length of the spring (m)
    g = 9.81    # Gravitational acceleration (m/s^2)

    # Define initial conditions
    t0 = 0                      # Initial time
    r0 = np.array([1.0, 0.0])  # Initial position
    v0 = np.array([0.0, 0.0])   # Initial velocity

    # Define the simulation parameters
    tf = 10                 # Final time (s)
    h = 1e-2                # Time step (s)
    N = int((tf - t0) / h)  # Number of time steps

    # Simulate using both methods
    t_expl, r_expl, v_expl = simulate_elastic_pendulum_explicit_euler(m, c, L, g, t0, r0, v0, h, N)
   
    # Compute lengths
    lengths_expl = np.linalg.norm(r_expl, axis=1)
    
    # Plot the trajectories 
    L_max = np.max(lengths_expl)
    plt.plot(r_expl[:, 0], r_expl[:, 1], label='Explicit Euler', linestyle='-')
    plt.gca().set_xlim(-L_max, L_max)
    plt.gca().set_ylim(-L_max, L_max)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.gca().set_aspect('equal')
    plt.title('Trajectory of the Elastic Pendulum')
    plt.legend()
    plt.grid(True)

    # Plot the lengths over time 
    plt.figure(figsize=(10, 5))
    plt.plot(t_expl, lengths_expl, label='Explicit Euler', linestyle='-')
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
    frac = int(np.ceil(N / N_frames))
    r_expl_frames = r_expl[::frac]
    N_frames = r_expl_frames.shape[0]

    def update(frame, r_expl_frames, pend_expl, ax):
        # Update the line of the explicit Euler
        pend_expl.set_data([0, r_expl_frames[frame, 0]], [0, r_expl_frames[frame, 1]])
        return pend_expl,

    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(-2*L, 2*L)
    ax.set_ylim(-2*L, 2*L)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    ax.set_title('Elastic Pendulum Motion (Explicit Euler)')
    ax.grid(True)

    # Initialize a line plot: starting location of the pendulum
    pend_expl, = ax.plot([0, r_expl_frames[0, 0]], [0, r_expl_frames[0, 1]], marker='o', label='expl. Euler')
    ax.legend()

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=N_frames, interval=1000/fps, fargs=( r_expl_frames, pend_expl, ax),
        blit=True
    )

    # Display the animation
    plt.show()