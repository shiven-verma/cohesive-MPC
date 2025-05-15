import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import argparse
import time

# Optional pygame visualization
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

class Agent:
    def __init__(self, agent_id, initial_state, dt=0.1, horizon=10, is_leader=False):
        """
        Initialize an agent with double integrator dynamics
        state = [x, y, vx, vy]
        """
        self.id = agent_id
        self.state = np.array(initial_state, dtype=float)  # [x, y, vx, vy]
        self.dt = dt
        self.horizon = horizon
        self.is_leader = is_leader
        
        # Control limits
        self.u_max = 2.0  # Maximum acceleration
        
        # State history for plotting
        self.state_history = [self.state.copy()]
        
        # For ADMM
        self.planned_trajectory = None
        self.lagrange_multipliers = np.zeros((horizon, 4))
        self.rho = 1.0  # ADMM penalty parameter
        
        # Matrices for double integrator dynamics
        self.A = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        self.B = np.array([
            [0.5*dt**2, 0.0],
            [0.0, 0.5*dt**2],
            [dt, 0.0],
            [0.0, dt]
        ])
        
        # MPC weights
        self.Q = np.diag([10.0, 10.0, 1.0, 1.0])  # State cost
        self.R = np.diag([0.1, 0.1])  # Control cost
        
    def dynamics(self, state, control):
        """Apply double integrator dynamics"""
        return self.A @ state + self.B @ control
    
    def compute_control(self, neighbors_trajectories=None, goal_state=None):
        """Compute control using MPC"""
        if self.planned_trajectory is None:
            # Initialize planned trajectory
            self.planned_trajectory = np.zeros((self.horizon, 4))
            state = self.state
            for i in range(self.horizon):
                self.planned_trajectory[i] = state
                state = self.dynamics(state, np.zeros(2))
                
        # Define optimization problem
        def objective(u_flat):
            u = u_flat.reshape(self.horizon, 2)
            cost = 0
            state = self.state.copy()
            trajectory = np.zeros((self.horizon, 4))
            
            for i in range(self.horizon):
                trajectory[i] = state
                if self.is_leader and goal_state is not None:
                    # Leader optimizes to reach goal
                    cost += (state - goal_state).T @ self.Q @ (state - goal_state)
                state = self.dynamics(state, u[i])
                cost += u[i].T @ self.R @ u[i]
                
                # Add obstacle avoidance cost
                for obstacle in obstacles:
                    dist = np.sqrt((state[0] - obstacle[0])**2 + (state[1] - obstacle[1])**2) - obstacle[2]
                    cost += 100 * max(0, 1.0 - dist)**2
              # Add formation cost
            if not self.is_leader and neighbors_trajectories is not None:
                for neighbor_id, neighbor_traj in neighbors_trajectories.items():
                    if neighbor_traj is not None:  # Make sure the neighbor has a planned trajectory
                        desired_offset = formation_offsets[(self.id, neighbor_id)]
                        for i in range(self.horizon):
                            rel_pos = trajectory[i, :2] - neighbor_traj[i, :2] - desired_offset
                            cost += 50.0 * np.sum(rel_pos**2)
            
            # Add ADMM coupling terms
            if self.planned_trajectory is not None:
                for i in range(self.horizon):
                    diff = trajectory[i] - self.planned_trajectory[i]
                    cost += self.lagrange_multipliers[i].dot(diff) + (self.rho/2) * np.sum(diff**2)
            
            return cost
        
        # Initial control sequence is zero
        u_init = np.zeros(self.horizon * 2)
        
        # Control limits
        bounds = [(-self.u_max, self.u_max) for _ in range(self.horizon * 2)]
        
        # Optimize
        result = minimize(objective, u_init, method='SLSQP', bounds=bounds)
        u_optimal = result.x.reshape(self.horizon, 2)
        
        # Update planned trajectory
        state = self.state.copy()
        new_trajectory = np.zeros((self.horizon, 4))
        for i in range(self.horizon):
            state = self.dynamics(state, u_optimal[i])
            new_trajectory[i] = state
        
        # Update ADMM variables
        self.lagrange_multipliers += self.rho * (new_trajectory - self.planned_trajectory)
        self.planned_trajectory = new_trajectory
        
        # Return first control action
        return u_optimal[0]
    
    def update(self, control):
        """Update agent state using control input"""
        self.state = self.dynamics(self.state, control)
        self.state_history.append(self.state.copy())
        return self.state


def distance(pos1, pos2):
    """Calculate Euclidean distance between two positions"""
    # Convert to numpy arrays if they are lists
    pos1_array = np.array(pos1)
    pos2_array = np.array(pos2)
    return np.sqrt(np.sum((pos1_array - pos2_array)**2))


def calculate_formation_error(agents):
    """Calculate maximum deformation in the formation"""
    errors = []
    for i in range(len(agents)):
        for j in range(i+1, len(agents)):
            desired_dist = distance(
                initial_formation[i][:2], 
                initial_formation[j][:2]
            )
            actual_dist = distance(
                agents[i].state[:2], 
                agents[j].state[:2]
            )
            errors.append(abs(actual_dist - desired_dist))
    return max(errors)


def visualize_pygame(agents, obstacles, goal_position, screen_size=800, scale=100):
    """Visualize the simulation using pygame"""
    if not PYGAME_AVAILABLE:
        return None
        
    pygame.init()
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("Formation Control with MPC")
    clock = pygame.time.Clock()
    
    # Colors
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    BLACK = (0, 0, 0)
    YELLOW = (255, 255, 0)
    
    def world_to_screen(pos):
        """Convert world coordinates to screen coordinates"""
        x = int(pos[0] * scale + screen_size / 2)
        y = int(screen_size / 2 - pos[1] * scale)  # Flip y-axis
        return (x, y)
    
    return {
        'screen': screen,
        'clock': clock,
        'colors': (WHITE, RED, GREEN, BLUE, BLACK, YELLOW),
        'world_to_screen': world_to_screen
    }


def update_visualization(vis, agents, obstacles, goal_position):
    """Update pygame visualization"""
    if vis is None:
        return
        
    screen, clock, colors, world_to_screen = vis['screen'], vis['clock'], vis['colors'], vis['world_to_screen']
    WHITE, RED, GREEN, BLUE, BLACK, YELLOW = colors
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return False
    
    screen.fill(WHITE)
    
    # Draw obstacles
    for obs in obstacles:
        pygame.draw.circle(
            screen,
            BLACK,
            world_to_screen(obs[:2]),
            int(obs[2] * vis['screen'].get_width() / 16)
        )
    
    # Draw goal
    pygame.draw.circle(screen, YELLOW, world_to_screen(goal_position[:2]), 10)
    
    # Draw agents
    colors = [RED, GREEN, BLUE]
    for i, agent in enumerate(agents):
        pos = world_to_screen(agent.state[:2])
        pygame.draw.circle(screen, colors[i], pos, 8)
        
        # Draw velocity vectors
        vel_endpoint = (
            pos[0] + int(agent.state[2] * 20), 
            pos[1] - int(agent.state[3] * 20)
        )
        pygame.draw.line(screen, BLACK, pos, vel_endpoint, 2)
    
    # Update display
    pygame.display.flip()
    clock.tick(30)
    return True


def plot_results(agents, formation_errors, goal_position):
    """Plot simulation results"""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Trajectories
    ax = axs[0, 0]
    colors = ['r', 'g', 'b']
    for i, agent in enumerate(agents):
        traj = np.array(agent.state_history)
        ax.plot(traj[:, 0], traj[:, 1], f'{colors[i]}-', label=f'Agent {i+1}')
        ax.plot(traj[0, 0], traj[0, 1], f'{colors[i]}o', markersize=8)
        ax.plot(traj[-1, 0], traj[-1, 1], f'{colors[i]}s', markersize=8)
    
    # Draw obstacles
    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='gray', alpha=0.5)
        ax.add_patch(circle)
    
    # Draw goal
    ax.plot(goal_position[0], goal_position[1], 'y*', markersize=15)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Agent Trajectories')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')
    
    # Position vs time
    ax = axs[0, 1]
    time_steps = np.arange(len(agents[0].state_history)) * dt
    for i, agent in enumerate(agents):
        traj = np.array(agent.state_history)
        ax.plot(time_steps, traj[:, 0], f'{colors[i]}-', label=f'Agent {i+1} X')
        ax.plot(time_steps, traj[:, 1], f'{colors[i]}--', label=f'Agent {i+1} Y')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position')
    ax.set_title('Position vs Time')
    ax.grid(True)
    ax.legend()
    
    # Velocity vs time
    ax = axs[1, 0]
    for i, agent in enumerate(agents):
        traj = np.array(agent.state_history)
        ax.plot(time_steps, traj[:, 2], f'{colors[i]}-', label=f'Agent {i+1} Vx')
        ax.plot(time_steps, traj[:, 3], f'{colors[i]}--', label=f'Agent {i+1} Vy')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity')
    ax.set_title('Velocity vs Time')
    ax.grid(True)
    ax.legend()
    
    # Formation error
    ax = axs[1, 1]
    ax.plot(time_steps, formation_errors, 'k-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Max Formation Error')
    ax.set_title('Formation Error vs Time')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('formation_control_results.png')
    plt.show()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Decentralized MPC for Formation Control')
    parser.add_argument('--visualize', action='store_true', help='Enable pygame visualization')
    parser.add_argument('--timesteps', type=int, default=100, help='Number of simulation time steps')
    args = parser.parse_args()
    
    # Simulation parameters
    dt = 0.1
    horizon = 10
    
    # Initial formation (triangular)
    initial_formation = [
        [0.0, 0.0, 0.0, 0.0],  # Agent 1 (leader): [x, y, vx, vy]
        [1.0, 0.0, 0.0, 0.0],  # Agent 2
        [0.5, 0.866, 0.0, 0.0]  # Agent 3
    ]
    
    # Desired formation offsets from agent i to agent j
    formation_offsets = {
        (1, 0): np.array([1.0, 0.0]),      # Agent 2 offset from Agent 1
        (2, 0): np.array([0.5, 0.866]),    # Agent 3 offset from Agent 1
        (0, 1): np.array([-1.0, 0.0]),     # Agent 1 offset from Agent 2
        (2, 1): np.array([-0.5, 0.866]),   # Agent 3 offset from Agent 2
        (0, 2): np.array([-0.5, -0.866]),  # Agent 1 offset from Agent 3
        (1, 2): np.array([0.5, -0.866])    # Agent 2 offset from Agent 3
    }
    
    # Goal position
    goal_position = np.array([5.0, 5.0, 0.0, 0.0])  # [x, y, vx, vy]
    
    # Obstacles [x, y, radius]
    obstacles = [
        [2.5, 2.5, 1.0],
        [3.5, 1.5, 0.8]
    ]
    
    # Create agents
    agents = [
        Agent(0, initial_formation[0], dt, horizon, is_leader=True),
        Agent(1, initial_formation[1], dt, horizon),
        Agent(2, initial_formation[2], dt, horizon)
    ]
    
    # Initialize visualization
    vis = None
    if args.visualize and PYGAME_AVAILABLE:
        vis = visualize_pygame(agents, obstacles, goal_position)
    
    # Store formation errors
    formation_errors = [calculate_formation_error(agents)]
    
    # Simulation loop
    for t in range(args.timesteps):
        # Exchange trajectories
        trajectories = {i: agent.planned_trajectory for i, agent in enumerate(agents)}
        
        # Compute and apply control
        controls = []
        for i, agent in enumerate(agents):
            # Filter out own trajectory
            neighbor_trajectories = {j: traj for j, traj in trajectories.items() if j != i}
            
            # Only leader knows the goal
            goal = goal_position if agent.is_leader else None
            
            control = agent.compute_control(neighbor_trajectories, goal)
            controls.append(control)
        
        for agent, control in zip(agents, controls):
            agent.update(control)
        
        # Calculate formation error
        formation_errors.append(calculate_formation_error(agents))
        
        # Update visualization
        if vis:
            if not update_visualization(vis, agents, obstacles, goal_position):
                break
    
    # Close pygame if used
    if vis:
        pygame.quit()
    
    # Plot results
    plot_results(agents, formation_errors, goal_position)
    
    print("Simulation complete!")
    print(f"Final formation error: {formation_errors[-1]:.4f}")
    print(f"Distance to goal: {distance(agents[0].state[:2], goal_position[:2]):.4f}")