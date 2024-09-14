import osqp
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
import time

np.set_printoptions(precision=4, suppress=True)

def get_csc_idx(csc_matrix):
    rows = csc_matrix.indices
    cols = csc_matrix.indptr

    row_indices = []
    col_indices = []

    # Iterate over columns
    for col in range(len(cols) - 1):
        start = cols[col]
        end = cols[col + 1]
        for index in range(start, end):
            row = rows[index]
            row_indices.append(row)
            col_indices.append(col)

    return np.array(row_indices), np.array(col_indices)


class OSQPMPC:
    def __init__(self, nx, nu, N, verbose=False, polish=False):
        self.prob = osqp.OSQP()
        self.verbose = verbose
        self.P, self.q = None, None
        self.l, self.u = None, None
        self.A = None
        self.nx, self.nu = nx, nu
        self.nc, self.nt = None, None
        self.dyn_updated = False
        self.cost_updated = False
        self.setuped = False
        self.dyn_map = None
        self.cnst_map = None
        self.N = N
        self.sol = None
        self.polish = polish

    def init_dyn(self, F, C):
        """
        Step 1: initialize the dynamics and constraints
        prepare the constraint matrix for the MPC and set the sparsity pattern
        F: 2D dynamics matrix [x,u] => [x']
        C: 2D constraint mapping matrix [x,u] => [c]
        """
        nx, nu, nc = self.nx, self.nu, C.shape[0]
        N = self.N
        assert F.ndim == 2, "F should be 2D"
        assert C.ndim == 2, "C should be 2D"
        assert C.shape[1] == nx + nu, "check C shape or nx, nu"
        assert F.shape[1] == nx + nu, "check F shape or nx, nu"
        assert F.shape[0] == nx, "F shape is not correct"
        F = sparse.csc_matrix(F)  # remove 0 elements
        C = sparse.csc_matrix(C)

        self.nc = nc
        self.A = sparse.csc_matrix((nx * (N + 1) + nc * N, (nx + nu) * N + nx))
        self.l = np.ones(nx * (N + 1) + nc * N) * -np.inf
        self.u = np.ones(nx * (N + 1) + nc * N) * np.inf
        self.N = N

        for i in range(nx):
            # set up the initial equity
            self.A[i, i] = 1

        for i in range(N):
            # set up the dynamics equity
            y_offset = (i + 1) * nx
            x_offset = i * (nx + nu)
            self.A[y_offset : y_offset + nx, x_offset : x_offset + nx + nu] = (
                F  # define sparsity
            )
            for j in range(nx):
                self.A[y_offset + j, x_offset + nx + nu + j] = -1

        for i in range(N):
            # set up constraint mapping
            y_offset = (N + 1) * nx + i * nc
            x_offset = i * (nx + nu)
            self.A[y_offset : y_offset + nc, x_offset : x_offset + nx + nu] = (
                C  # define sparsity
            )

        self.l[nx : nx * (N + 1)] = 0  # Ax_k+Bu_k - x_{k+1} = 0
        self.u[nx : nx * (N + 1)] = 0

        F_data_num = F.data.size  # num of non-zero elements in F
        C_data_num = C.data.size  # num of non-zero elements in C
        self.F_data_num = F_data_num
        self.C_data_num = C_data_num
        self.dyn_map = np.zeros(
            (N, F_data_num, 3), dtype=int
        )  # N * n * [src_row, src_col, dst_index]
        self.cnst_map = np.zeros(
            (N, C_data_num, 3), dtype=int
        )  # N * nc * [src_row, src_col, dst_index]
        A_row, A_col = get_csc_idx(self.A)

        dyn_block_record = [0] * N
        constraint_block_record = [0] * N

        for row, col, idx in zip(A_row, A_col, range(len(self.A.data))):
            # construct the map from src to dst
            if row < (N + 1) * nx:  # dynamic data
                block_idx = row // nx - 1
                if block_idx == -1:  # skip the initial equity
                    continue
                y_offset = (block_idx + 1) * nx
                x_offset = block_idx * (nx + nu)
                block_x = col - x_offset
                if block_x >= nx + nu:  # only record F data
                    continue
                record, map = dyn_block_record, self.dyn_map
                data_num = F_data_num
            else:  # constraint data
                block_idx = (row - (N + 1) * nx) // nc
                y_offset = (N + 1) * nx + block_idx * nc
                x_offset = block_idx * (nx + nu)
                block_x = col - x_offset
                record, map = constraint_block_record, self.cnst_map
                data_num = C_data_num

            block_y = row - y_offset
            block_dataIdx = record[block_idx]
            assert block_dataIdx < data_num, "block data index is out of range"
            map[block_idx, block_dataIdx, 0] = block_y
            map[block_idx, block_dataIdx, 1] = block_x
            map[block_idx, block_dataIdx, 2] = idx
            record[block_idx] += 1

        self.sol = np.zeros((N + 1) * (nx + nu))

        for i in range(N):  # check if every block is filled
            assert dyn_block_record[i] == F_data_num, "map construct might be wrong"
            assert (
                constraint_block_record[i] == C_data_num
            ), "map construct might be wrong"

    def update_F(self, idxs, Fs):
        """
        F = [A, B], in 3D batched form
        """
        assert Fs.ndim == 3, "Fs should be 3D"
        assert self.dyn_map is not None, "Dynamics is not prepared"
        assert idxs.ndim == 1, "idxs should be 1D"
        assert idxs.size == Fs.shape[0], "idxs and Fs should have the same size"
        assert Fs.shape[0] <= self.N, "Fs size should be less than N"
        try:
            sp_map = self.dyn_map[idxs].flatten()
            idxs = idxs
            N_updated = idxs.size
            src_idx = np.arange(N_updated).repeat(self.F_data_num)
            self.A.data[sp_map[2::3]] = Fs[src_idx, sp_map[0::3], sp_map[1::3]]
            self.dyn_updated = True
        except:
            raise RuntimeError("dynamic update failed, please check sparsity pattern")

    def set_cost(self, state_cost, M, tar_cost):
        """
        Step 2: initialize the cost matrices
        define the cost function, must be after def_dyn
        state_cost: state cost
        M: 2D target transform matrix mapping from [x,u] to [t]
        tar: target vector, not trajectory
        tar_cost: mapped target cost
        """
        assert M.ndim == 2, "M should be 2D"
        assert self.dyn_map is not None, "Dynamics is not prepared"
        assert len(state_cost) == self.nx + self.nu, "state cost is not correct"
        assert len(tar_cost) == len(M), "target cost is not correct"
        W = np.diag(np.array(tar_cost) ** 2)
        self.W2 = W
        self.M = M
        self.nt = M.shape[0]
        P_block = M.T @ W @ M + np.diag(state_cost)
        P_blocks = P_block[None].repeat(self.N + 1, 0)
        P_blocks[0, : self.nx, : self.nx] = 0  # ignore the initial state cost
        self.P_blocks = P_blocks
        P = sparse.block_diag(P_blocks, format="csc")
        P = P[: -self.nu, : -self.nu]  # we don't care about the last control input
        self.P = sparse.csc_matrix(P)
        self.cost_updated = True

    def set_tar_traj(self, tar_traj):
        """
        STEP 3:
        update the target for the MPC
        """
        assert self.M is not None, "cost matrix is not prepared"
        assert tar_traj.ndim == 2, "target trajectory should be 2D"
        assert tar_traj.shape[0] == self.N, "target size is not correct"
        tar_traj = np.concatenate([np.zeros((1, self.nt)), tar_traj], axis=0)
        self.tar_traj = tar_traj
        q = (-self.M.T[None] @ self.W2[None] @ tar_traj[:, :, None]).flatten()
        q = q[: -self.nu]
        self.q = q
        self.target_updated = True

    def udpate_C(self, idxs, Cs):
        """
        set the constraint for the MPC
        C: constraint mapping matrix [x,u] => [c]
        """
        assert Cs.ndim == 3, "Cs should be 3D"
        assert Cs.shape[2] == self.nx + self.nu, "C shape is not correct"
        assert Cs.shape[0] <= self.N, "Cs size should be less than N"
        assert idxs.ndim == 1, "idxs should be 1D"
        assert idxs.size == Cs.shape[0], "idxs and Cs should have the same size"
        try:
            sp_map = self.cnst_map[idxs].flatten()
            idxs = idxs
            N_updated = idxs.size
            src_idx = np.arange(N_updated).repeat(self.C_data_num)
            self.A.data[sp_map[2::3]] = Cs[src_idx, sp_map[0::3], sp_map[1::3]]
            self.dyn_updated = True
        except:
            raise RuntimeError(
                "constraint update failed, please check sparsity pattern"
            )

    def set_lower(self, l):
        assert self.nc is not None, "call init_dyn first"
        assert len(l) == self.nc, "lower bound is not correct"
        self.l[self.nx * (self.N + 1) :] = np.concatenate([l] * self.N)

    def set_upper(self, u):
        assert self.nc is not None, "call init_dyn first"
        assert len(u) == self.nc, "upper bound is not correct"
        self.u[self.nx * (self.N + 1) :] = np.concatenate([u] * self.N)

    def solve(self, x0, step_cost=False):
        kwargs = {}

        if not self.setuped:
            self.prob.setup(
                P=self.P,
                q=self.q,
                A=self.A,
                l=self.l,
                u=self.u,
                warm_start=True,
                verbose=self.verbose,
                polish=self.polish,
            )
            self.setuped = True
            self.cost_updated = False
            self.target_updated = False
            self.dyn_updated = False
        
        if self.cost_updated:
            kwargs["Px"] = sparse.triu(self.P).data
            self.cost_updated = False

        if self.target_updated:
            kwargs["q"] = self.q
            self.target_updated = False

        if self.dyn_updated:
            kwargs["Ax"] = self.A.data
            self.dyn_updated = False

        self.l[: self.nx] = x0
        self.u[: self.nx] = x0
        kwargs["l"] = self.l
        kwargs["u"] = self.u
        self.prob.update(**kwargs)

        res = self.prob.solve()
        self.sol[: -self.nu] = res.x
        traj = self.sol.reshape(-1, self.nx + self.nu)

        if step_cost:
            cost = self.get_step_cost(add_const=True)
        else:
            cost = res.info.obj_val

        return traj.copy(), cost

    def get_step_cost(self, add_const=True):
        """
        get the cost of each step of last solution
        """
        n_xu = self.nx + self.nu
        steps = self.sol.reshape(-1, n_xu)
        q = np.concatenate([self.q, np.zeros(self.nu)]).reshape(self.N + 1, n_xu)
        # add zero input to the last step

        step_costs = np.zeros(self.N + 1)

        quad_costs = 0.5 * np.einsum("ij,ijk,ik->i", steps, self.P_blocks, steps)
        linear_costs = np.einsum("ij,ij->i", steps, q)

        step_costs = quad_costs + linear_costs  # the sum should match obj_val

        if add_const:  # add the constant term to the cost
            W2t = self.W2[None] @ self.tar_traj[:, :, None]
            const = self.tar_traj[:, None, :] @ W2t * 0.5
            step_costs += const.flatten()

        return step_costs


if __name__ == "__main__":
    dt = 0.1
    N = 100

    Ad = np.array( # a simple 2D dynamics
        [
            [1.0, dt, 0, 0],
            [0.0, 1.0, 0, 0],
            [0, 0, 1.0, dt],
            [0, 0, 0.0, 1.0],
        ]
    )
    Bd = np.array([[0, 0], [dt, 0], [0, 0], [0, dt]])

    state_cost = [0, 0, 0, 0, 1, 1]  # state cost of [x,u]

    M = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ]
    )  # M is the target mapping matrix mapping from [x,u] to [t]
    tar_cost = [2, 2]  # target cost of mapped state
    target = [4, 4]  # mapped target

    C = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
    # C is the constraint matrix mapping from [x,u] to [c]


    # Initialize the MPC after defining the dynamics and constraints
    MPC = OSQPMPC(nx=4, nu=2, N=N, verbose=True)

    # STEP 1: initialize and set the dynamics and constraints mapping
    F = np.concatenate([Ad, Bd], axis=1)
    MPC.init_dyn(F, C)

    # STEP 2: initialize and set the cost matrices
    MPC.set_cost(state_cost, M, tar_cost)

    # STEP 3: update the target trajectory
    Org_tar_traj = np.array(target)[None].repeat(N, 0)
    MPC.set_tar_traj(Org_tar_traj)

    # Finally, solve the MPC
    x_init = np.array([1, 0, 1, 0])
    traj, s_cost = MPC.solve(x_init, step_cost=False)

    # Visualize
    pos_x = traj[:, 0]
    pos_y = traj[:, 2]
    print("terminal position:", pos_x[-1], pos_y[-1])
    plt.scatter(pos_x, pos_y, color="blue", marker="o", label="Points")
    plt.title("2D Points Plot")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.legend()
    plt.show()

    # below are examples how to update the dynamics and constraints

    # OPTIONAL: batch update F example
    F = F[None]
    update_N = 10
    F[0, 1, 4] = 0  # remove acc integration
    F[0, 3, 5] = 0
    Fs = np.concatenate([F] * update_N, axis=0)
    idxs = np.arange(update_N) + 1
    _time = time.time()
    MPC.update_F(idxs, Fs)
    print("update F time:", time.time() - _time)

    # OPTIONAL: batch update C example
    update_N = 100
    Cs = np.concatenate([C[None]] * update_N, axis=0) * 0
    idxs = np.arange(update_N)
    _time = time.time()
    MPC.udpate_C(idxs, Cs)
    print("update C time:", time.time() - _time)

    # OPTIONAL: set target trajectory
    MPC.set_tar_traj(((np.arange(N) + 1) / (N))[:, None] * np.array([[5, 10]]))

    # again, solve the MPC
    x_init = np.array([1, 0, 1, 0])
    _time = time.time()
    traj, s_cost = MPC.solve(x_init, step_cost=True)
    print("step cost:", s_cost, "sum:", np.sum(s_cost), "time:", time.time() - _time)

    # Visualize again
    pos_x = traj[:, 0]
    pos_y = traj[:, 2]
    print("terminal position:", pos_x[-1], pos_y[-1])
    plt.scatter(pos_x, pos_y, color="blue", marker="o", label="Points")
    plt.title("2D Points Plot")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.legend()
    plt.show()
