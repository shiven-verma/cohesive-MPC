import numpy as np




class system:
    def __init__(self, A, b, x0=None):
        self.A = A
        self.b = b
        self.x0 = x0 

    def evolve(self, x):
        """
        Evolve the system state based on the current state x.
        This is a placeholder for the actual evolution logic.
        """
        # Example: x = self.A @ x + self.b
        return self.A @ x + self.b
    

class MPC:
    def __init__(self,N=10, C=1):
        self.N = N  # Prediction horizon
        self.C = C #control horizon


    def prediction_system(self,system,x):
        A_est = system.A
        b_est = system.b

        x_pred = np.zeros((self.N, x.shape[0]))
        x_pred[0] = x
        for i in range(1, self.N):
            x_pred[i] = A_est @ x_pred[i-1] + b_est
        return x_pred
    
    def control_input(self, system, x):
        """
        Calculate the control input based on the current state x.
        This is a placeholder for the actual control logic.
        """
        # Example: u = -K @ x, where K is a gain matrix
        K = np.eye(x.shape[0])
        u = -K @ x
        return u
    
    def cost_function(self, system, x):
        """
        Calculate the cost function based on the current state x.
        This is a placeholder for the actual cost function logic.
        """
        # Example: J = x.T @ Q @ x + u.T @ R @ u
        Q = np.eye(x.shape[0])
        R = np.eye(x.shape[0])
        u = self.control_input(system, x)
        J = x.T @ Q @ x + u.T @ R @ u
        return J
    

    

    
