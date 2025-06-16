from scipy.optimize import minimize
import numpy as np
import time
import matplotlib.pyplot as plt




class Machine:
    def __init__(self,initial_states):
        self.x0 = initial_states[0]
        self.v0 = initial_states[1]
        self.info_logger(f'Initial states: {self.x0}, {self.v0}')

    def dynamics(self,X,u):
        vel = X[1]
        
        accelration = u


        states_derivative = np.array([vel,accelration])

        return states_derivative
    
    def info_logger(self,info):
        print(f'[{time.time()}]: {info}')



class MPC:
    def __init__(self,initial_states,dynamics):
        self.num_states = 2
        self.dt = 0.01
        self.N = 10
        self.C = 1
        self.dynamics = dynamics
        self.x0 = initial_states[0]
        self.v0 = initial_states[1]
        self.goal = 1


    def reference_trajectory(self):
        # Generate a reference trajectory for the MPC to follow
        
        Xref = np.ones((self.N, self.num_states))*self.goal

        return Xref
        #more complex reference trajectory can be generated here



    def optimizer(self,X0,initial_guess):
        optimized_result = minimize(self.cost_function , initial_guess, args=(X0,))  
        
        return optimized_result.x
        ...

    def prediction_loop(self,X,u):
        prediction = np.zeros((self.N, self.num_states))
        for i in range(self.N):
            X = self.step(X, u[i])
            prediction[i,:] = X
            
        return prediction


        ...

    def cost_function(self,u,X0):
        X = self.prediction_loop(X0,u)
        Xref = self.reference_trajectory()
        assert X.shape[0] == Xref.shape[0] == self.N, "X should have shape (N, num_states)"
        cost = 0.0
        for i in range(self.N):
            cost += np.sum((X[i,0] - Xref[i])**2) + np.sum(u[i]**2)
        print(cost,u[0])
        return cost
        # Cost function to minimize

        ...
    def step(self,X,u):
        X_der = self.dynamics(X,u)
        X_next = X + self.dt * X_der
        return X_next
    
    def run(self, u):
        X_data = np.zeros((self.N, self.num_states))
        X = np.array([self.x0, self.v0])
        for i in range(self.N):
            X = self.step(X, u[i])
            X_data[i, :] = X    
        return X_data
    # single run
    
    def run2(self):
        initial_u = np.zeros((self.N, 1))
        opt_u = initial_u+1
        X = np.array([self.x0, self.v0])
        X_data = []
        iter = []
        k=0
        while abs(opt_u[0])>1e-4:
            opt_u = self.optimizer(X,initial_u)
            X_next = self.step(X, opt_u[0])
            X = X_next
            # print(f'Iteration{k+1} and opt_u: {opt_u[0]}')
            k=k+1
            X_data.append(X)
            iter.append(k)
        print(f'Converged in {k} iterations with final control input: {opt_u[0]}')
        iter = np.array(iter)
        X_data = np.array(X_data)
        return [iter,X_data]
        # run the MPC loop until convergence or a stopping criterion is met


        ...

      

if __name__ == "__main__":

    initial_states = [0,0]
    machine = Machine(initial_states)
    mpc = MPC(initial_states, machine.dynamics)

    [iter,X_data] = mpc.run2()

    plt.plot(iter*mpc.dt, X_data[:, 0], label='Position')
    plt.plot(iter*mpc.dt, X_data[:, 1], label='Velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('States')    
    plt.title('MPC Prediction')
    plt.legend()
    plt.show()

    
        



        
