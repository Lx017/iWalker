import numpy as np
from osqpmpc import OSQPMPC
from scipy.interpolate import CubicSpline

def global_hlip_initialization(COM_height, TSSP, TDSP):
    lambda_val = np.sqrt(9.81 / COM_height)  # Example calculation for lambda

    # Initialize the A matrix
    A_S2S = np.zeros((3, 3))
    A_S2S[0, 0] = 1
    A_S2S[0, 1] = np.cosh(TSSP * lambda_val) - 1
    A_S2S[0, 2] = TDSP * np.cosh(TSSP * lambda_val) + 1.0 / lambda_val * np.sinh(TSSP * lambda_val)
    A_S2S[1, 0] = 0
    A_S2S[1, 1] = np.cosh(TSSP * lambda_val)
    A_S2S[1, 2] = TDSP * np.cosh(TSSP * lambda_val) + 1.0 / lambda_val * np.sinh(TSSP * lambda_val)
    A_S2S[2, 0] = 0
    A_S2S[2, 1] = lambda_val * np.sinh(TSSP * lambda_val)
    A_S2S[2, 2] = np.cosh(TSSP * lambda_val) + lambda_val * TDSP * np.sinh(TSSP * lambda_val)

    # Initialize the B matrix
    B_S2S = np.zeros((3, 1))
    B_S2S[0, 0] = - np.cosh(TSSP * lambda_val) + 1
    B_S2S[1, 0] = B_S2S[0, 0] - 1
    B_S2S[2, 0] = - lambda_val * np.sinh(TSSP * lambda_val)

    # Calculate deadbeat gain
    Kdeadbeat = np.zeros(3)
    Kdeadbeat[0] = 1 / (-2 + 2 * np.cosh(TSSP * lambda_val) + TDSP * lambda_val * np.sinh(TSSP * lambda_val))
    Kdeadbeat[1] = 1
    Kdeadbeat[2] = (4 * TDSP * lambda_val + np.cosh(TSSP * lambda_val / 2) / np.sinh(TSSP * lambda_val / 2) * (3.0 + 2 * TDSP**2 * lambda_val**2 + 2 * TDSP * lambda_val * (np.cosh(TSSP * lambda_val / 2) / np.sinh(TSSP * lambda_val / 2))) + np.tanh(TSSP * lambda_val / 2)) / (2 * lambda_val * (2 + TDSP * lambda_val * 1 / (np.tanh(TSSP * lambda_val / 2))))

    return A_S2S, B_S2S, Kdeadbeat


height = 0.8
TSSP = 0.4
TDSP = 0.1 
vdes = 0.2
step_size = 0.3

Ax, Bx, _ = global_hlip_initialization(height, TSSP, TDSP)
print("Matrix A:\n", Ax)
print("Matrix B:\n", Bx)
Ay, By, _ = global_hlip_initialization(height, TSSP, TDSP)

A = np.block([[Ax, np.zeros((3,3))],[np.zeros((3,3)), Ay]])
B = np.block([[Bx, np.zeros((3,1))],[np.zeros((3,1)), By]])

N = 20

MPC = OSQPMPC(nx=6,nu=2,N=N,polish=True)

F = np.concatenate([A, B], axis=1)
C = np.array([[0, 0, 0, 0, 0, 0,1,0],
              [0, 0, 0, 0, 0, 0,0,1]])

MPC.init_dyn(F, C)
MPC.set_lower(np.array([-2,-2]))
MPC.set_upper(np.array([2,2]))

state_cost = np.array([0,0,0,0,0,0,1,1])
tar = np.array([10,3])
M = np.array([[1,0,0,0,0,0,0,0],
              [0,0,0,1,0,0,0,0]])
tar_cost = np.array([10,10])
MPC.set_cost(state_cost=state_cost, M=M, tar_cost=tar_cost)

waypoints = np.array([[-1,0],
             [-0.71,0.71],
             [0,1],
             [0.71,0.71],
             [1,0]])
spline = CubicSpline(waypoints[:,0], waypoints[:,1])

x = 0
last_x = 0
last_y = spline(last_x)
tar_dis = 0.05
sampled = []
while x < 1:
    if len(sampled) == N:
        break
    y = spline(x)
    dx = x - last_x
    dy = y - last_y
    dis = np.sqrt(dx**2 + dy**2)
    if dis > tar_dis:
        sampled.append([x,y])
        last_x = x
        last_y = y
    x+=0.01

Org_tar_traj = np.array(sampled)

MPC.set_tar_traj(Org_tar_traj)
x0 = np.array([0,0,0,0,0,0])
org_traj, cost = MPC.solve(x0,step_cost=True)
org_cost = cost.sum()
PERTURBATION = 0.1
import time
_time = time.time()
vec = np.zeros_like(Org_tar_traj)
for i in range(Org_tar_traj.shape[0]):
    for j in range(Org_tar_traj.shape[1]):
        new_tar_traj = Org_tar_traj.copy()
        new_tar_traj[i,j] += PERTURBATION
        MPC.set_tar_traj(new_tar_traj)
        traj, cost = MPC.solve(x0,step_cost=True)
        diff = cost.sum() - org_cost
        gradient = diff/PERTURBATION
        vec[i,j] = -gradient

print("Time taken:", time.time()-_time)
sample_N = 100

_time = time.time()
sample_Vec =np.zeros_like(vec)
for i in range(Org_tar_traj.shape[0]):
    for j in range(Org_tar_traj.shape[1]):
        best_loss = np.inf
        best_sample = None
        for s in range(-sample_N,sample_N):
            new_tar_traj = Org_tar_traj.copy()
            perturbation = s/sample_N*0.5
            new_tar_traj[i,j] += perturbation
            MPC.set_tar_traj(new_tar_traj)
            traj, cost = MPC.solve(x0,step_cost=True)
            cost = cost.sum()
            if cost < best_loss:
                best_loss = cost
                sample_Vec[i,j] = perturbation
print("Time taken:", time.time()-_time)


x_traj = org_traj[:,0]
y_traj = org_traj[:,3]
print("Trajectory:\n", traj, "\nstep costs:\n", cost, "\nTotal cost:", np.sum(cost))

import matplotlib.pyplot as plt

u = np.random.rand(N+1)
v = np.random.rand(N+1)
vec/=10
direct_error = np.array([x_traj,y_traj]).T[1:] - Org_tar_traj
direct_error*=10
sample_Vec*=10
plt.quiver(Org_tar_traj[:,0], Org_tar_traj[:,1], sample_Vec[:,0], sample_Vec[:,1], angles='xy', scale_units='xy', scale=1, color='orange',label="local sampling")
plt.quiver(Org_tar_traj[:,0], Org_tar_traj[:,1], direct_error[:,0], direct_error[:,1], angles='xy', scale_units='xy', scale=1, color='green',label="subtraction")
plt.quiver(Org_tar_traj[:,0], Org_tar_traj[:,1], vec[:,0], vec[:,1], angles='xy', scale_units='xy', scale=1, color='red',label="perturbation")
plt.scatter(x_traj, y_traj, color="blue", marker="o", label="optimized")
plt.scatter(Org_tar_traj[:,0], Org_tar_traj[:,1], color="black", marker="x", label="target")
plt.title("2D Points Plot")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.legend()
plt.show()