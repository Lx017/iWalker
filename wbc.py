import numpy as np
import pypose as pp
import torch
import matplotlib.pyplot as plt

dt = 0.01

M=np.array([20])

A = torch.tensor([
    [1, dt],
    [0, 1],
])
B = torch.tensor([
    [0],
    [dt/M[0]],
]).float()

F = np.array([
    [1, dt, 0],
    [0, 1, dt],
    [0, 0, 0]
])

def state_transition(x):
    return F @ x

sys = pp.module.LTI(A=A,B=B,C=torch.zeros_like(A),D=torch.zeros_like(B))
T=500
Q=torch.diag(torch.tensor([1,1,0.001])).repeat(1,T,1,1)
p=torch.zeros(3).repeat(1,T,1)
lqr = pp.module.LQR(sys, Q=Q, p=p,T=T)
x_init = np.random.rand(3)
x_init[1] *= 0.1
x_init[2] = 0
x,u,cost = lqr(torch.tensor(x_init[:2]).float()[None],dt)
x_hists=[]
plt.plot(x[0,:,0])
for j in range(len(M)):
    x_hist = []
    x = x_init.copy()
    for i in range(T):
        P_error = 0-x[0]
        V_error = 0-x[1]
        f = 10*P_error + 5*V_error
        noise_scale = np.random.normal(1, 0.1)
        _M = M[j] * noise_scale
        x[2] = f/_M #acceleration
        x_next = state_transition(x)
        x_hist.append(x)
        x = x_next
    x_hist = np.array(x_hist)
    plt.plot(x_hist[:,0])
    x_hists.append(x_hist)

plt.show()