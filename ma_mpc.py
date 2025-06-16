from scipy.optimize import minimize
import numpy as np
import time
import matplotlib.pyplot as plt


class MultiAgentSystem:
    def __init__(self, num_agents, initial_positions, initial_velocities):
        self.num_agents = num_agents
        self.initial_positions = np.array(initial_positions)
        self.initial_velocities = np.array(initial_velocities)
        self.info_logger(f'Multi-agent system initialized with {num_agents} agents')
        self.info_logger(f'Initial positions: {self.initial_positions}')
        self.info_logger(f'Initial velocities: {self.initial_velocities}')

    def agent_dynamics(self, X, u):
        """Dynamics for a single agent: double integrator"""
        position = X[0]
        velocity = X[1]
        acceleration = u
        
        states_derivative = np.array([velocity, acceleration])
        return states_derivative
    
    def multi_agent_dynamics(self, X, U):
        """Dynamics for all agents"""
        # X shape: (num_agents, 2) - position and velocity for each agent
        # U shape: (num_agents,) - control input for each agent
        X_dot = np.zeros_like(X)
        
        for i in range(self.num_agents):
            X_dot[i, :] = self.agent_dynamics(X[i, :], U[i])
        
        return X_dot
    
    def info_logger(self, info):
        print(f'[{time.time()}]: {info}')


class MultiAgentMPC:
    def __init__(self, num_agents, initial_positions, initial_velocities, laplacian_matrix):
        self.num_agents = num_agents
        self.num_states_per_agent = 2  # position and velocity
        self.dt = 0.01
        self.N = 10  # prediction horizon
        
        # Initialize multi-agent system
        self.system = MultiAgentSystem(num_agents, initial_positions, initial_velocities)
        
        # Initial state vector: [pos1, vel1, pos2, vel2, ..., posN, velN]
        self.X0 = np.zeros(num_agents * 2)
        for i in range(num_agents):
            self.X0[2*i] = initial_positions[i]
            self.X0[2*i + 1] = initial_velocities[i]
        
        # Network parameters
        self.K = laplacian_matrix  # Pinned Laplacian matrix
        self.B = np.zeros(num_agents)
        self.B[0] = 1  # B = [1, 0, 0, ...]
        self.xd = 1.0  # Setpoint
        self.Q = np.eye(num_agents)  # Weight matrix
        self.Qd = np.eye(num_agents)  # Weight matrix
        self.Qd[0,0] = 0.0

        
        print(f"Laplacian matrix K:\n{self.K}")
        print(f"Selection matrix B: {self.B}")
        print(f"Setpoint xd: {self.xd}")

    def create_pinned_laplacian(self, num_agents):
        """Create a simple pinned Laplacian matrix for a path graph with pinned first node"""

        L = np.array([[1, 0, 0, 0, 0, 0],
                      [-1, 2, -1, 0, 0, 0],
                      [0, -1, 2, -1, 0, 0],
                      [0, -1, 0, 1, 0, 0],
                      [0, 0, -1, 0, 1, 0],
                      [0, 0, 0, 0, -1, 1]])
       
        
        return L

    def step(self, X, U):
        """Single integration step for all agents"""
        # Reshape X to (num_agents, 2)
        X_reshaped = X.reshape(self.num_agents, 2)
        
        # Get derivatives
        X_dot = self.system.multi_agent_dynamics(X_reshaped, U)
        
        # Integrate
        X_next_reshaped = X_reshaped + self.dt * X_dot
        
        # Flatten back
        X_next = X_next_reshaped.flatten()
        
        return X_next

    def prediction_loop(self, X0, U):
        """Predict future states over horizon N"""
        # U shape: (N * num_agents,) - flattened control sequence
        # Reshape U to (N, num_agents)
        U_reshaped = U.reshape(self.N, self.num_agents)
        
        predictions = np.zeros((self.N, self.num_agents * 2))
        X = X0.copy()
        
        for i in range(self.N):
            X = self.step(X, U_reshaped[i, :])
            predictions[i, :] = X
        return predictions

    def cost_function(self, U, X0):
        """Multi-agent consensus cost function with position and velocity consensus terms"""
        predictions = self.prediction_loop(X0, U)
        
        total_cost = 0.0
        
        for i in range(self.N):
            # Extract positions and velocities at time step i
            positions = predictions[i, ::2]  # Every other element starting from 0 (positions)
            velocities = predictions[i, 1::2]  # Every other element starting from 1 (velocities)
            
            # Position consensus cost: (K*x - B*xd)^T * Q * (K*x - B*xd)
            position_consensus_error = self.K @ positions - self.B * self.xd

            position_cost = position_consensus_error.T @ self.Q @ position_consensus_error
            # print(f"Position consensus error at step {i}: {position_cost}")
            
            
            # Velocity consensus cost: (K*x_dot)^T * Q * (K*x_dot)
            velocity_consensus_error = self.K @ velocities
            velocity_cost = velocity_consensus_error.T @ self.Qd @ velocity_consensus_error
            # print(velocity_consensus_error)
            
            # Control cost (regularization)
            U_step = U[i*self.num_agents:(i+1)*self.num_agents]
            control_cost = 0.001 * np.sum(U_step**2)  # Increased weight for control cost
            
            total_cost += position_cost + velocity_cost + control_cost

        # print(total_cost)
        
        return total_cost

    def optimizer(self, X0, initial_guess):
        """Optimize control sequence"""
        result = minimize(self.cost_function, initial_guess, args=(X0,))
        return result.x

    def run_mpc(self, max_iterations=100):
        """Run MPC loop until convergence"""
        # Initialize control sequence
        initial_u = np.zeros(self.N * self.num_agents)
        opt_u = initial_u + 1  # Start with non-zero to enter loop
        
        X = self.X0.copy()
        X_data = []
        U_data = []
        iter_data = []
        k = 0
        
        convergence_threshold = 0.01
        
        while k < max_iterations and np.max(np.abs(opt_u[:self.num_agents])) > convergence_threshold:
            # Optimize control sequence
            opt_u = self.optimizer(X, initial_u)
            
            # Apply first control input
            U_current = opt_u[:self.num_agents]
            X_next = self.step(X, U_current)
            
            # Store data
            X_data.append(X.copy())
            U_data.append(U_current.copy())
            iter_data.append(k)
            
            # Update state
            X = X_next
            print(k)
            k += 1
            
            # Shift control sequence (warm start)
            initial_u = np.roll(opt_u, -self.num_agents)
            initial_u[-self.num_agents:] = 0  # Set last control inputs to zero
            # print(np.max(np.abs(opt_u[:self.num_agents])))
        
        print(f'Converged in {k} iterations')
        print(f'Final positions: {X[::2]}')
        print(f'Final velocities: {X[1::2]}')
        
        return np.array(iter_data), np.array(X_data), np.array(U_data)

    def plot_results(self, iter_data, X_data, U_data):
        """Plot simulation results"""
        time_steps = iter_data * self.dt
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot positions
        for i in range(self.num_agents):
            positions = X_data[:, 2*i]
            ax1.plot(time_steps, positions, label=f'Agent {i+1}', linewidth=2)
        
        ax1.axhline(y=self.xd, color='r', linestyle='--', label='Setpoint')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position')
        ax1.set_title('Multi-Agent Position vs Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot velocities
        for i in range(self.num_agents):
            velocities = X_data[:, 2*i + 1]
            ax2.plot(time_steps, velocities, label=f'Agent {i+1}', linewidth=2)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity')
        ax2.set_title('Multi-Agent Velocity vs Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot control inputs
        for i in range(self.num_agents):
            controls = U_data[:, i]
            ax3.plot(time_steps, controls, label=f'Agent {i+1}', linewidth=2)
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Control Input')
        ax3.set_title('Multi-Agent Control Input vs Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Configuration
    num_agents = 6
    initial_positions = [0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Distributed initial positions
    initial_velocities = [0, 0, 0, 0, 0, 0]  # All start at rest
    
    # Create pinned Laplacian matrix for 6 agents (path graph with pinned first node)
    mpc = MultiAgentMPC(num_agents, initial_positions, initial_velocities, None)
    laplacian = mpc.create_pinned_laplacian(num_agents)
    
    # Re-initialize with the Laplacian
    mpc = MultiAgentMPC(num_agents, initial_positions, initial_velocities, laplacian)
    
    # Run MPC
    iter_data, X_data, U_data = mpc.run_mpc(max_iterations=500)
    
    # Plot results
    mpc.plot_results(iter_data, X_data, U_data)
