import numpy as np
import matplotlib.pyplot as plt

from simultaneous_relaxation import simultaneous_relaxation
from newton_method import newton_method
from scipy.optimize import root

def simulate_rigid_pendulum_implicit_euler(m, L, g, t0, r0, v0, h, N):
    """
    Simulates the motion of an rigid pendulum using the implicit Euler scheme (similar to Algorithm 2).
    Args:
        m (float): Mass of the point mass.
        L (float): Length of the pendulum.
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
    nla = 1     # Number of constraint forces (lambda)
    t = np.zeros(nt)        # Array of time instants
    r = np.zeros((nt, nr))  # Array of positions
    v = np.zeros((nt, nv))  # Array of velocities
    la = np.zeros((nt, nla))  # Array of velocities

    # Set initial conditions
    t[0] = t0
    r[0] = r0
    v[0] = v0
    la[0] = 0

    # Simulate over all time steps
    for k in range(N):
        # Update the time
        t[k+1] = t[k] + h
        # Use the prediction for the position and the velocity as start of the solver
        initial_guess = np.concatenate((r[k] + h * v[k], v[k], la[k]))
        # Define residual
        def R(x):
             # Extract rk+1, vk+1, lambdak+1 from the combined vector x
            rk1 = x[0:2]
            vk1 = x[2:4]  
            lak1 = x[4]    
            # Calculate the residual for the position update
            residual_r = rk1 - r[k] - h * vk1
            # Calculate the constraint force
            F_c = - lak1 * rk1 / np.linalg.norm(rk1)
            # Define the gravitational force vector
            F_g = m * g * np.array([0, -1])
            # Calculate the residual for the velocity update
            residual_v = m * vk1 - m * v[k] - h * (F_c + F_g)
            # Calculate the residual for the constraint
            residual_la = np.array([np.linalg.norm(rk1) - L])
            # Return the combined residual vector
            return np.concatenate((residual_r, residual_v, residual_la))

        # Solve for the next state using simultaneous relaxation
        # (Unfortunately, it is very difficult to find a parameter mu that leads to convergence. Feel free to try!)
        # x, converged = simultaneous_relaxation(R, initial_guess, tol=1e-5, mu=0.01, max_iter=int(1e4))
        
        # Solve for the next state using Newton's method
        # Implement dR here:
        # def dR(x):
        #     pass
        # x, converged = newton_method(R, dR, initial_guess)
        
        # Solve for the next state using more sophisticated method from scipy
        sol = root(R, initial_guess)
        x = sol.x
        converged = sol.success

        # If the solver converged, update the state
        if converged:
            r[k+1] = x[0:2]
            v[k+1] = x[2:4]
            la[k+1] = x[4]
        # If the solver did not converge, print an error message and return the partial results
        else:
            print(f"No convergence found in step {k}.")
            return t[:k+1], r[:k+1], v[:k+1], la[:k+1]
    # Return the complete results
    return t, r, v, la

if __name__ == '__main__':
    # Define physical parameters of the simulation
    m = 1.0     # Mass (kg)
    L = 1.0     # Length of the pendulum (m)
    g = 9.81    # Gravitational acceleration (m/s^2)

    # Define initial conditions
    t0 = 0                  # Initial time
    phi0 = np.deg2rad(90)   # Initial angle of the pendulum
    r0 = L * np.array([np.sin(phi0), -np.cos(phi0)]) # Initial position
    phi_dot0 = 0            # initial angular velocity
    v0 = phi_dot0 * L * np.array([np.cos(phi0), np.sin(phi0)]) # Initial velocity
    
    # Define the simulation parameters
    tf = 10                 # Final time (s)
    h = 1e-2                # Time step (s)
    N = int((tf - t0) / h)  # Number of time steps

    # Simulate using both methods
    t, r, v, la = simulate_rigid_pendulum_implicit_euler(m, L, g, t0, r0, v0, h, N)

    # Compute lengths
    lengths = np.linalg.norm(r, axis=1)

    # Plot the trajectories on the same plot
    plt.figure(figsize=(10, 5))
    plt.gca().set_xlim(-1.1*L, 1.1*L)
    plt.gca().set_ylim(-1.1*L, 1.1*L)
    plt.plot(r[:, 0], r[:, 1], label='Implicit Euler', linestyle='-')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Trajectory of the Pendulum')
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.grid(True)

    # Plot the lengths over time on the same plot
    plt.figure(figsize=(10, 5))
    plt.plot(t, lengths, label='Implicit Euler', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Length (m)')
    plt.title('Length of the Rigid Pendulum Over Time')
    plt.legend()
    plt.grid(True)

    plt.show()

    ###########
    # Animation
    import matplotlib.animation as animation
    
    fps=10
    N_frames = (tf - t0) * fps
    frac = int(np.ceil(N / N_frames))
    r_frames = r[::frac]
    N_frames = r_frames.shape[0]

    def update(frame, r_frames, pend, ax):
        # Update the line of the explicit Euler
        pend.set_data([0, r_frames[frame, 0]], [0, r_frames[frame, 1]])
        return pend,

    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(-1.2*L, 1.2*L)
    ax.set_ylim(-1.2*L, 1.2*L)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    ax.set_title('Elastic Pendulum Motion (Explicit Euler)')
    ax.grid(True)

    # Plot reference circle
    phi = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(phi), np.sin(phi), color='0.5', linestyle='--')
    
    # Initialize a line plot: starting location of the pendulum
    pend, = ax.plot([0, r_frames[0, 0]], [0, r_frames[0, 1]], marker='o', label='impl. Euler')
    ax.legend()

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=N_frames, interval=1000/fps, fargs=( r_frames, pend, ax),
        blit=True
    )

    # Display the animation
    plt.show()