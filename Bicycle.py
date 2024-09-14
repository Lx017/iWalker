import numpy as np
from osqpmpc import OSQPMPC
from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt
import pypose as pp
import torch


dt = 0.02
v = 1

class Bicycle(pp.module.NLS):
    def __init__(self):
        super(Bicycle, self).__init__()
    
    def observation(self, state, input, t=None):
        return 0

    def state_transition(self, state, input, t=None):
        theta = state[0,2]
        x = state[0,0]
        y = state[0,1]

        new_state = torch.zeros((1,3))
        new_state[0,0] += x + v*torch.cos(theta)*dt
        new_state[0,1] += y + v*torch.sin(theta)*dt
        #new_state[3] = theta + v*np.tan(input[1])*dt
        new_state[0,2] += theta + input[0,0]*dt

        return new_state
device = torch.device("cpu")
n_batch, T = 1, 80
n_state, n_ctrl = 3, 1
n_sc = n_state + n_ctrl
Q_ = torch.eye(n_sc, device=device)*1e-1
Q_[0, 0] = 1
Q_[1, 1] = 1
Q = torch.tile(Q_, (n_batch, T, 1, 1))
p = torch.zeros(n_batch, T, n_sc)
time  = torch.arange(0, T, device=device) * dt
current_u = torch.zeros((1, T, 1), device=device)
x_init = torch.tensor([[-1, -1, 0]]).float()
stepper = pp.utils.ReduceToBason(steps=15, verbose=True)
cartPoleSolver = Bicycle().to(device)
MPC = pp.module.MPC(cartPoleSolver, Q, p, T, stepper=stepper).to(device)
# x, u, cost = MPC(dt, x_init, u_init=current_u)
# print("x = ", x)
# print("u = ", u)
    
# plt.scatter(x[0,:,0], x[0,:,1])
# plt.show()
step_interval = 0.043
def getF_batched(theta):
    # Create a batch of identity matrices
    batch_size = theta.shape[0]
    m = np.tile(np.eye(3), (batch_size, 1, 1))
    # Set the cos(theta) and sin(theta) values in the appropriate places
    m[:, 0, 2] = np.cos(theta)*step_interval
    m[:, 1, 2] = np.sin(theta)*step_interval
    m[:, 2, 2] = 1
    
    F = np.concatenate([m, np.zeros((batch_size, 3, 2))], axis=2)
    F[:, 0, 3] = -np.sin(theta)
    F[:, 1, 3] = np.cos(theta)
    F[:, 2, 4] = 1
    return F

N = 50

MPC = OSQPMPC(nx=3,nu=2,N=N,polish=True,verbose=True)
F = getF_batched(np.array([0.001]))[0]

test_xu = torch.tensor([0,0,0,0.],requires_grad=True)
F_t = torch.tensor(F).float()

C = np.array([[0, 0, 0, 1,0]])

MPC.init_dyn(F, C)
MPC.set_lower(np.array([-0.01]))
MPC.set_upper(np.array([0.01]))

step_cost = 0.01
state_cost = np.array([0,0,0,1000,10])
tar = np.array([0,0])
M = np.array([[1,0,0,0,0],
              [0,1,0,0,0]])
tar_cost = np.array([1,1])
MPC.set_cost(state_cost=state_cost, M=M, tar_cost=tar_cost)
if __name__ == "__main__":
    # waypoints = np.array([[0,0],
    #                       [1,0],
    #                       [1,1],
    #                       [0,1],])
    waypoints = np.array([[-1,-1],
                          [-0.75,-0.7],
                          [-0.5,-0.9],
                          [-0.1,-0.9],
                          [-0.1,-0.5],
                          [-0.3,-0.3],
                          [0,0],])
    #waypoints[:,0]+=1
    pts = pp.chspline(torch.tensor(waypoints).float(), 0.01)

    sampled = []
    lastX = 0
    lastY = 0
    idx = 0
    while len(sampled)<N:
        cur_x, cur_y = pts[idx]
        dis = np.sqrt((cur_x-lastX)**2 + (cur_y-lastY)**2)
        if dis>step_interval:
            sampled.append([cur_x,cur_y])
            lastX = cur_x
            lastY = cur_y
        else:
            idx+=1

    step_traj = np.array([np.arange(0,N)*step_interval,np.zeros(N)]).T
    org_traj = np.array(sampled)

    MPC.set_tar_traj(org_traj)
    thetas = np.arctan2(step_traj[1:,1]-step_traj[:-1,1],step_traj[1:,0]-step_traj[:-1,0])
    for i in range(1000):
        thetas = np.arctan2(step_traj[1:,1]-step_traj[:-1,1],step_traj[1:,0]-step_traj[:-1,0])
        x0 = np.array([-1,-1,1])
        F = getF_batched(thetas)
        MPC.update_F(np.arange(1,N),F)
        solved, cost = MPC.solve(x0,step_cost=True)
        step_traj = solved[:-1,:2]
        org_cost = cost.sum()
        print(org_cost)

        COM_x = solved[:,0]
        COM_y = solved[:,1]

    plt.scatter(COM_x, COM_y, color="blue", marker="o", label="solved COM")
    plt.scatter(org_traj[:,0], org_traj[:,1], color="red", marker="x", label="waypoints")
    plt.title("2D Points Plot")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.legend()
    plt.show()