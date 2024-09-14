import numpy as np
from osqpmpc import OSQPMPC
from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt
import pypose as pp
import torch



class Unicycle_MPC:
    def __init__(self, N):
        self.N = 100

        self.MPC = OSQPMPC(nx=3,nu=2,N=N,polish=True,verbose=False)
        F = self.getF_batched(np.array([0.5]))[0]
        C = np.array([[0, 0, 0, 1,0],
                      [0, 0, 0, 0,1]])

        self.MPC.init_dyn(F, C)
        self.MPC.set_lower(np.array([-0.001,-0.01]))
        self.MPC.set_upper(np.array([0.001,0.01]))

        tar = np.array([0,0])
        M = np.array([[1,0,0,0,0],
                    [0,1,0,0,0]])
        tar_cost = np.array([1,1])
    
    def solve(self, tar_traj, init_theta, turning_cost,ACC_COST,v, iter=200):
        POS_COST = 1
        state_cost = np.array([0,0,0,turning_cost*10,ACC_COST])
        M = np.array([[1,0,0,0,0],
                    [0,1,0,0,0]])
        tar_cost = np.array([POS_COST,POS_COST])
        self.MPC.set_cost(state_cost=state_cost, M=M, tar_cost=tar_cost)
        self.MPC.set_tar_traj(tar_traj)
        total_cost = 0
        for i in range(iter):
            if init_theta is not None:
                new_thetas = np.repeat(init_theta,self.N-1)
                init_theta = None
            else:
                new_thetas = np.arctan2(traj[1:,1]-traj[:-1,1],traj[1:,0]-traj[:-1,0])
            #thetas = 0.1*thetas + 0.9*new_thetas #meaningless smoothing
            #thetas = np.concatenate([[new_thetas[0]],new_thetas])
            thetas = np.concatenate([np.zeros(1),new_thetas])
            x0 = np.array([tar_traj[0,0],tar_traj[0,1],v])
            F = self.getF_batched(thetas)
            self.MPC.update_F(np.arange(0,self.N),F)
            #self.MPC.update_F(np.arange(10,self.N),F[10:])#not changing initial theta
            solved, cost = self.MPC.solve(x0,step_cost=True)
            traj = solved[:-1,:2]
            org_cost = cost.sum()
            total_cost += org_cost

            COM_x = solved[:,0]
            COM_y = solved[:,1]
        return COM_x, COM_y, total_cost

    def getF_batched(self, theta):
        # Create a batch of identity matrices
        batch_size = theta.shape[0]
        m = np.tile(np.eye(3), (batch_size, 1, 1))
        # Set the cos(theta) and sin(theta) values in the appropriate places
        m[:, 0, 2] = np.cos(theta)
        m[:, 1, 2] = np.sin(theta)
        m[:, 2, 2] = 1
        
        F = np.concatenate([m, np.zeros((batch_size, 3, 2))], axis=2)
        F[:, 0, 3] = -np.sin(theta)
        F[:, 1, 3] = np.cos(theta)
        F[:, 2, 4] = 1
        return F

N = 100
UMPC = Unicycle_MPC(N)
if __name__ == "__main__":
    waypoints = np.array([[-1,-1],
                          [-0.75,-0.7],
                          [-0.5,-0.9],
                          [-0.1,-0.9],
                          [-0.1,-0.5],
                          [-0.3,-0.3],
                          [0,0.3],])
    pts = pp.chspline(torch.tensor(waypoints).float(), 6/N)[:N]
    traj = np.array([np.arange(0,N)*1,np.arange(0,N)*1]).T
    org_traj = np.array(pts)

    COM_x, COM_y = UMPC.solve(org_traj,traj,50)
    plt.scatter(COM_x, COM_y, color=(0,0,1), marker="o", label="turning cost 50")
    COM_x, COM_y = UMPC.solve(org_traj,traj,2e2)
    plt.scatter(COM_x, COM_y, color=(0.2,0,1), marker="o", label="turning cost 200")
    COM_x, COM_y = UMPC.solve(org_traj,traj,5e2)
    plt.scatter(COM_x, COM_y, color=(0.4,0,1), marker="o", label="turning cost 500")
    COM_x, COM_y = UMPC.solve(org_traj,traj,1e3)
    plt.scatter(COM_x, COM_y, color=(0.6,0,1), marker="o", label="turning cost 1000")
    COM_x, COM_y = UMPC.solve(org_traj,traj,3e3)
    plt.scatter(COM_x, COM_y, color=(0.8,0,1), marker="o", label="turning cost 3000")
    COM_x, COM_y = UMPC.solve(org_traj,traj,1e4)
    plt.scatter(COM_x, COM_y, color=(1,0,1), marker="o", label="turning cost 10000")
    COM_x, COM_y = UMPC.solve(org_traj,traj,1e5)
    plt.scatter(COM_x, COM_y, color=(1,0,0.8), marker="o", label="turning cost 100000")
    plt.scatter(org_traj[:,0], org_traj[:,1], color="red", marker="x", label="target")
    plt.title("2D Points Plot")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.legend()
    plt.show()