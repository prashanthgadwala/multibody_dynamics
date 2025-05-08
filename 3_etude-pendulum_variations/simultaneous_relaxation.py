import numpy as np

def simultaneous_relaxation(R, x0, mu=0.01, tol=1e-6, max_iter=int(1e4)):
    """
    Applies the simultaneous relaxation method to find the root of a nonlinear equation system R(x) = 0.

    Args:
        R (function): A function that represents the residual of the nonlinear system. 
                      It takes x as input and returns a numpy array representing the residual vector.
        x0 (np.array): The initial guess for the solution.
        mu (float): The relaxation parameter.
        tol (float): The tolerance for convergence. 
        max_iter (int): The maximum number of iterations.

    Returns:
        np.array or None: The approximate solution if convergence is achieved within max_iter iterations, otherwise returns None.
        bool: Flag indicating convergence. True if converged, False if not.
    """
    # Set initial condition
    x = x0
    # Iterate through the number of max iterations
    for i in range(max_iter):
        # Evaluate residual
        Rx = R(x)
        # print(np.max(np.abs(Rx))) # For debugging
        # Check if the residual is within tolerance
        if np.max(np.abs(Rx)) < tol:  # max norm is used here
            converged = True
            return x, True
        
        # Update approximation using simultaneous relaxation method
        x = x - mu * Rx
    
    # If the solution is not found, return None and converged=False flag.
    return None, False

if __name__ == '__main__':
    # Example usage:
    # Define a sample residual function (example: a simple system of equations)
    def R(x):
        # Example system: 
        # x[0]^2 = 0
        return np.array([x[0]**2])

    # Set initial guess
    x0 = np.array([1.0])

    # Set parameters
    mu = 0.1
    epsilon = 1e-6
    Nmax = 10000

    # Run the simultaneous relaxation method
    solution, converged = simultaneous_relaxation(R, x0, mu=mu, tol=epsilon, max_iter=Nmax)
    
    # Print the solution
    if converged:
        print("Approximate solution:", solution)
    else:
        print("Not converged.")