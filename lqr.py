import numpy as np
import matplotlib.pyplot as plt

# System setup
dt = 0.1
A = np.array([[1, dt],
              [0, 1]])
B = np.array([[0.5 * dt ** 2],
              [dt]])

n, m = A.shape[0], B.shape[1]
N = 100  # horizon

# Cost matrices
Q = np.eye(n)
R = 0.1 * np.eye(m)
Qf = 10 * np.eye(n)

# Linear cost terms (tracking final state to x_target)
x_target = np.array([10, 0])
q = -1 * Q @ x_target   # encourages x_k to be near x_target
qf = -1 * Qf @ x_target

# Preallocate arrays
P = [None] * (N + 1)
p = [None] * (N + 1)
K = [None] * N
d = [None] * N

# Terminal cost
P[N] = Qf
p[N] = qf

# Backward Riccati recursion
for k in reversed(range(N)):
    BT_PB = B.T @ P[k+1] @ B
    BT_PA = B.T @ P[k+1] @ A
    inv_term = np.linalg.inv(R + BT_PB)

    K[k] = inv_term @ BT_PA
    d[k] = inv_term @ B.T @ p[k+1]

    P[k] = Q +K[k].T@R@K[k] + (A-B@K[k]).T @ P[k+1] @ (A - B @ K[k])
    p[k] = q + (A-B@K[k]).T @ (p[k+1] - P[k+1] @ B @ d[k]) + K[k].T @(R@d[k])

# Simulate forward
x = np.zeros((n, N+1))
u = np.zeros((m, N))

x[:, 0] = np.array([0, 0])  # initial state

for k in range(N):
    u[:, k] = -K[k] @ x[:, k] - d[k]
    x[:, k+1] = A @ x[:, k] + B @ u[:, k]

# Plot
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(np.arange(N+1)*dt, x[0], label='Position')
plt.plot(np.arange(N+1)*dt, x[1], label='Velocity')
plt.axhline(10, color='gray', linestyle='--', label='Target Pos')
plt.title("State Trajectories")
plt.xlabel("Time [s]")
plt.legend()

plt.subplot(1,2,2)
plt.plot(np.arange(N)*dt, u[0])
plt.title("Control Input (Acceleration)")
plt.xlabel("Time [s]")
plt.tight_layout()
plt.show()
