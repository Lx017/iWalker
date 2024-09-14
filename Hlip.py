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
TSSP = 0.24
TDSP = 0.0
vdes = 0.2
step_size = 0.3

Ax, Bx, _ = global_hlip_initialization(height, TSSP, TDSP)
print("Matrix A:\n", Ax)
print("Matrix B:\n", Bx)
Ay, By, _ = global_hlip_initialization(height, TSSP, TDSP)

A = np.block([[Ax, np.zeros((3,3))],[np.zeros((3,3)), Ay]])
B = np.block([[Bx, np.zeros((3,1))],[np.zeros((3,1)), By]])

N = 150

MPC = OSQPMPC(nx=6,nu=2,N=N,polish=True)

F = np.concatenate([A, B], axis=1)
C = np.array([[0, 0, 0, 0, 0, 0,1,0],
              [0, 0, 0, 0, 0, 0,0,1]])

MPC.init_dyn(F, C)
MPC.set_lower(np.array([-2,-2]))
MPC.set_upper(np.array([2,2]))

step_cost = 0.01
state_cost = np.array([0,0,0,0,0,0,step_cost,step_cost])
tar = np.array([10,3])
M = np.array([[1,-1,0,0,0,0,0,0],
              [0,0,0,1,-1,0,0,0]])
tar_cost = np.array([10,10])
MPC.set_cost(state_cost=state_cost, M=M, tar_cost=tar_cost)

if __name__ == "__main__":
    waypoints = np.array([[-1,0],
                [-0.71,0.71],
                [0,1],
                [0.71,0.71],
                [1,0]])
    #waypoints[:,0]+=1
    spline = CubicSpline(waypoints[:,0], waypoints[:,1])

    x = 0
    last_x = 0
    last_y = spline(last_x)
    tar_step_len = 0.03
    sampled = []
    traj = []
    step_width = 0.05
    while len(sampled)<N:
        y = spline(x)
        dx = x - last_x
        dy = y - last_y
        dis = np.sqrt(dx**2 + dy**2)
        org_loc = np.array([x,y])
        traj.append(org_loc)
        if dis > tar_step_len:
            step_idx = len(sampled)
            vec = np.array([dx,dy])
            vec = vec / np.linalg.norm(vec)
            perpendic = np.array([-vec[1],vec[0]]) * (1 if step_idx % 2 == 0 else -1)
            offseted = org_loc + perpendic*step_width
            sampled.append(offseted)
            last_x = x
            last_y = y
        x+=0.001

    step_traj = np.array(sampled)
    org_traj = np.array(traj)

    MPC.set_tar_traj(step_traj)
    x0 = np.array([step_traj[0,0],0,0,step_traj[0,1],0,0])
    solved, cost = MPC.solve(x0,step_cost=True)
    org_cost = cost.sum()


    COM_x = solved[:,0]
    COM_y = solved[:,3]
    step_x = solved[:,0] - solved[:,1]
    step_y = solved[:,3] - solved[:,4]

    import matplotlib.pyplot as plt

    u = np.random.rand(N+1)
    v = np.random.rand(N+1)
    direct_error = np.array([COM_x,COM_y]).T[1:] - step_traj
    direct_error*=1
    plt.scatter(org_traj[:,0], org_traj[:,1], color="green", marker="o", label="target traj")
    #plt.quiver(step_traj[:,0], step_traj[:,1], direct_error[:,0], direct_error[:,1], angles='xy', scale_units='xy', scale=1, color='green',label="subtraction")
    plt.scatter(COM_x, COM_y, color="blue", marker="o", label="solved COM")
    plt.scatter(step_traj[:,0], step_traj[:,1], color="black", marker="x", label="step target")
    plt.scatter(step_x, step_y, color="red", marker="o", label="solved step")
    plt.title("2D Points Plot")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.legend()
    plt.show()