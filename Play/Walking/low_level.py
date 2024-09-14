#!usr/bin/env python
__author__ = "Westwood Robotics Corporation"
__email__ = "info@westwoodrobotics.io"
__copyright__ = "Copyright 2023 Westwood Robotics Corporation"
__date__ = "November 1, 2023"
__project__ = "BRUCE"
__version__ = "0.0.4"
__status__ = "Product"

"""
QP-Based Weighted Whole-Body Control
"""

import osqp
import Startups.memory_manager as MM
import Library.ROBOT_MODEL.BRUCE_kinematics as kin
from scipy import linalg
from scipy import sparse
from termcolor import colored
from Play.initialize import *
from Settings.BRUCE_macros import *
from Play.Walking.walking_macros import *
from Settings.Bruce import *
import matplotlib.pyplot as plt
from multiprocessing import shared_memory

def shmArr(name, shape, dtype=np.float32, reset=False):
    try:
        shm = shared_memory.SharedMemory(name=name, create=False)
    except:
        typeLen = 1
        if dtype==np.uint8:
            typeLen = 1
        elif dtype==np.float32:
            typeLen = 4
        elif dtype==np.float64:
            typeLen = 8
        elif dtype==np.int32:
            typeLen = 4
        shm = shared_memory.SharedMemory(name=name, create=True, size=np.prod(shape)*typeLen)
    arr = np.frombuffer(shm.buf, dtype=dtype).reshape(shape)
    if reset:
        arr[:] = 0
    return arr


contactForceSMArr = shmArr("forceSM", (4,3), np.float32, reset=True)
# BRUCE SETUP
bot = RDS.BRUCE()


def check_simulation_or_estimation_thread(robot):
    error = True
    while error:
        thread_data = MM.THREAD_STATE.get()
        if thread_data["simulation"][0] == 1.0 or thread_data["estimation"][0] == 1.0:
            error = False

            # move shoulder twice if good
            num = 2
            for _ in range(num):
                set_joint_positions(
                    robot,
                    30,
                    0.01,
                    arm_move=True,
                    arm_goal_positions=np.array([-0.7, 1.1, 2.0, 0.7, -1.1, -2.0]),
                )
                set_joint_positions(
                    robot,
                    30,
                    0.01,
                    arm_move=True,
                    arm_goal_positions=np.array([-0.7, 1.3, 2.0, 0.7, -1.3, -2.0]),
                )
        time.sleep(0.1)









def main_loop():
    # Check if estimation is running
    check_simulation_or_estimation_thread(bot)

    # CONTROL FREQUENCY
    loop_freq = 500  # run at 500 Hz
    loop_duration = 1.0 / loop_freq

    # FEEDBACK GAINS
    # swing foot
    if bot.getModuleState(MODULE_IFSIM):
        Kp_p = np.array([100.0, 100.0, 100.0]) * 10.0
        Kd_p = np.array([10.0, 10.0, 10.0]) * 10.0
    else:
        Kp_p = np.array([100.0, 100.0, 100.0]) * 1.0
        Kd_p = np.array([10.0, 10.0, 10.0]) * 1.0
    Kp_R = np.array([0.0, 50.0, 300.0]) * 1.0
    Kd_R = np.array([0.0, 10.0, 50.0]) * 1.0

    # [ang mom, lin mom, body rot, sw pos r, sw rot r, sw pos l, sw rot l]
    # balancing (b), right stance (r), left stance (l), single stance (s)

    Kp_all_b = np.array(
        [
            0.0,
            0.0,
            0.0,
            50,
            50,
            150,
            1000,
            1000,
            200,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )
    Kp_all_r = np.array(
        [
            0.0,
            0.0,
            0.0,
            0,
            0,
            150,
            1000,
            1000,
            200,
            0,
            0,
            0,
            0,
            0,
            0,
            Kp_p[0],
            Kp_p[1],
            Kp_p[2],
            Kp_R[0],
            Kp_R[1],
            Kp_R[2],
        ]
    )
    Kp_all_l = np.array(
        [
            0.0,
            0.0,
            0.0,
            0,
            0,
            150,
            1000,
            1000,
            200,
            Kp_p[0],
            Kp_p[1],
            Kp_p[2],
            Kp_R[0],
            Kp_R[1],
            Kp_R[2],
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )

    Kd_all_b = np.array(
        [10, 10, 1.0, 10, 10, 15, 100, 100, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    Kd_all_r = np.array(
        [
            10,
            10,
            1.0,
            5,
            5,
            15,
            100,
            100,
            20,
            0,
            0,
            0,
            0,
            0,
            0,
            Kd_p[0],
            Kd_p[1],
            Kd_p[2],
            Kd_R[0],
            Kd_R[1],
            Kd_R[2],
        ]
    )
    Kd_all_l = np.array(
        [
            10,
            10,
            1.0,
            5,
            5,
            15,
            100,
            100,
            20,
            Kd_p[0],
            Kd_p[1],
            Kd_p[2],
            Kd_R[0],
            Kd_R[1],
            Kd_R[2],
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )

    Kp_all = np.vstack((Kp_all_b, Kp_all_r, Kp_all_l))
    Kd_all = np.vstack((Kd_all_b, Kd_all_r, Kd_all_l))

    # COST WEIGHTS
    # angular momentum
    wk = 1.0

    # linear momentum xy
    wl_b = 50.0
    wl_s = 1.0

    # linear momentum z
    wz = 100.0

    # body orientation
    wR_b = 10.0
    wR_s = 10.0

    # stance foot contact
    wc_b = np.array([1e3, 1e3])
    wc_r = np.array([1e3, 1e-4])
    wc_l = np.array([1e-4, 1e3])

    # swing foot position
    wsp_b = np.array([1e-4, 1e-4])
    wsp_r = np.array([1e-4, 10.0])
    wsp_l = np.array([10.0, 1e-4])

    # swing foot orientation
    wsR_b = np.array([1e-4, 1e-4])
    wsR_r = np.array([1e-4, 1.0])
    wsR_l = np.array([1.0, 1e-4])

    # [ang mom, lin mom z, body rot, contact r, sw pos r, sw rot r, contact l, sw pos l, sw rot l]
    W_all_b = np.array(
        [wk, wl_b, wR_b, wc_b[0], wsp_b[0], wsR_b[0], wc_b[1], wsp_b[1], wsR_b[1]]
    )
    W_all_r = np.array(
        [wk, wl_s, wR_s, wc_r[0], wsp_r[0], wsR_r[0], wc_r[1], wsp_r[1], wsR_r[1]]
    )
    W_all_l = np.array(
        [wk, wl_s, wR_s, wc_l[0], wsp_l[0], wsR_l[0], wc_l[1], wsp_l[1], wsR_l[1]]
    )
    W_all = np.vstack((W_all_b, W_all_r, W_all_l))

    # regularization terms
    W_F = np.eye(12) * 1e-3  # contact force
    W_ddq = linalg.block_diag(1e-4 * np.eye(6), 1e-4 * np.eye(10))  # joint acceleration

    # QP SETUP
    # Decision Variables
    # x = [ddq, F] -     16 + 12 = 28
    # ddq          -   6 + 2 * 5 = 16 (6 dofs for floating base and 5 dofs for each leg)
    # F            - 2 * (3 + 3) = 12 (3 dofs for each toe/heel)

    # Constraints
    # Con1 - dynamics constraint: H_b*ddq - J_b"*F = -CG_b (6 cons)
    Aeq = np.zeros((6, 28))
    beq = np.zeros(6)

    # Con2 - contact force constraint:
    # -2*mu*fz_max <= -fx-mu*fz <= 0
    # -2*mu*fz_max <=  fx-mu*fz <= 0
    # -2*mu*fz_max <= -fy-mu*fz <= 0
    # -2*mu*fz_max <=  fy-mu*fz <= 0
    #                        fz <= fz_max (2 * 2 * 5 = 20 cons)
    mu = 0.5  # friction coefficient
    fz_max = 100.0
    fz_min = 1.0
    fz_max_all = np.array([[fz_max, fz_max], [fz_max, 0.0], [0.0, fz_max]])
    fz_min_all = np.array([[fz_min, fz_min], [fz_min, 0.0], [0.0, fz_min]])
    Aineq = np.zeros((20, 28))
    bineq_ls = np.array(
        [-2 * mu * fz_max, -2 * mu * fz_max, -2 * mu * fz_max, -2 * mu * fz_max, 0.0]
    )
    bineq_us = np.array([0.0, 0.0, 0.0, 0.0, fz_max])
    bineq_l = np.kron(np.ones(4), bineq_ls)
    bineq_u = np.kron(np.ones(4), bineq_us)
    Aineq[0:20, 16:28] = np.kron(
        np.eye(4),
        np.array(
            [
                [-1.0, 0.0, -mu],
                [1.0, 0.0, -mu],
                [0.0, -1.0, -mu],
                [0.0, 1.0, -mu],
                [0.0, 0.0, 1.0],
            ]
        ),
    )

    # Overall
    A0 = np.vstack((Aeq, Aineq))
    l0 = np.hstack((beq, bineq_l))
    u0 = np.hstack((beq, bineq_u))

    P0 = np.zeros((28, 28))
    q0 = np.zeros(28)

    P0[0:16, 0:16] = 1e-12 * np.ones((16, 16)) + W_ddq
    P0[16:28, 16:28] = W_F



    def constraint_update(
        l0, u0, Hb, CGb, Jrtb, Jrhb, Jltb, Jlhb, fz_max_r, fz_min_r, fz_max_l, fz_min_l
    ):
        A = np.copy(A0)
        l = np.copy(l0)
        u = np.copy(u0)

        # Constraint 1 - Dynamics
        A[0:6, 0:16] = Hb

        l[0:6] = -CGb
        u[0:6] = l[0:6]

        A[0:6, 16:19] = -Jrtb.T
        A[0:6, 19:22] = -Jrhb.T
        A[0:6, 22:25] = -Jltb.T
        A[0:6, 25:28] = -Jlhb.T

        # Constraint 2 - Contact Force
        u[10], l[10] = fz_max_r, fz_min_r
        u[15], l[15] = fz_max_r, fz_min_r
        u[20], l[20] = fz_max_l, fz_min_l
        u[25], l[25] = fz_max_l, fz_min_l

        return A, l, u

    def update_BaseOrientationCost():
        estRot = bot[EST]["bodyRot"]
        estOmg = bot[EST]["bodyOmg"]
        isThisRotCost = np.diag(Kp_all[Bs, 6:9]) @ MF.logvee(estRot.T @  bot[PLAN]["bodyRot"])
        isThisOmgCost = np.diag(Kd_all[Bs, 6:9]) @ (bot[PLAN]["bodyOmg"] - estOmg)
        ac2 = isThisOmgCost +isThisRotCost
        return W_all[Bs, 2] * np.eye(3), -W_all[Bs, 2] * ac2
    
    # def update_AngularMomentumCost(Jac0, dJac0dq, w0, Kd0, kt):
    #     ac0 = -Kd0 @ kt
    #     JacTW0 = Jac0.T * w0
    #     return JacTW0 @ Jac0, JacTW0 @ (dJac0dq - ac0)
    
    def update_AngularMomentumCost():
        ac0 = -np.diag(Kd_all[Bs, 0:3]) @ bot[EST]["angMomemtum"]
        JacTW0 = bot[EST]["AG_matrix"][0:3, :].T * W_all[Bs, 0]
        return JacTW0 @ bot[EST]["AG_matrix"][0:3, :], JacTW0 @ (bot[EST]["dAGdq_vector"][0:3] - ac0)

    def update_LinearMomentumCost():
        COMPos=bot[EST]["COMPos"]
        COMTarPos=bot[PLAN]["COMPos"]
        COMVel=bot[EST]["COMVel"]
        COMTarVel=bot[PLAN]["COMVel"]

        ac1 = (np.diag(Kp_all[Bs, 3:6]) @ (COMTarPos - COMPos) + np.diag(Kd_all[Bs, 3:6]) @ (COMTarVel - COMVel)) * MASS_TOT
        JacTW1 = bot[EST]["AG_matrix"][3:6, :].T @ np.diag([W_all[Bs, 1], W_all[Bs, 1], wz])
        return JacTW1 @ bot[EST]["AG_matrix"][3:6, :], JacTW1 @ (bot[EST]["dAGdq_vector"][3:6] - ac1)

    def update_RightStanceContactCost(Jac3, dJac3dq, w3):
        JacTW3 = Jac3.T * w3
        return JacTW3 @ Jac3, JacTW3 @ dJac3dq

    def update_RightSwingPositionCost(Jac4, dJac4dq, w4, Kp4, Kd4, pr, prd, vr, vrd):
        ac4 = Kp4 @ (prd - pr) + Kd4 @ (vrd - vr)
        JacTW4 = Jac4.T * w4
        return JacTW4 @ Jac4, JacTW4 @ (dJac4dq - ac4)

    def update_RightSwingOrientationCost(Jac5, dJac5dq, w5, Kp5, Kd5, Rr, Rrd, wr, wrd):
        ac5 = Kp5 @ MF.logvee(Rr.T @ Rrd) + Kd5 @ (wrd - wr)
        JacTW5 = Jac5.T * w5
        return JacTW5 @ Jac5, JacTW5 @ (dJac5dq - ac5[1:3])

    def update_LeftStanceContactCost(Jac6, dJac6dq, w6):
        JacTW6 = Jac6.T * w6
        return JacTW6 @ Jac6, JacTW6 @ dJac6dq

    def update_LeftSwingPositionCost(Jac7, dJac7dq, w7, Kp7, Kd7, pl, pld, vl, vld):
        ac7 = Kp7 @ (pld - pl) + Kd7 @ (vld - vl)
        JacTW7 = Jac7.T * w7
        return JacTW7 @ Jac7, JacTW7 @ (dJac7dq - ac7)

    def update_LeftSwingOrientationCost(Jac8, dJac8dq, w8, Kp8, Kd8, Rl, Rld, wl, wld):
        ac8 = Kp8 @ MF.logvee(Rl.T @ Rld) + Kd8 @ (wld - wl)
        JacTW8 = Jac8.T * w8
        return JacTW8 @ Jac8, JacTW8 @ (dJac8dq - ac8[1:3])

    # START CONTROL
    # confirm = input("Start whole body control? (y/n) ")
    # if confirm != "y":
    #     exit()

    plan_data = {
        "robot_state": np.zeros(1),
        "bodyRot": MF.Rz(bot[EST]["body_yaw_ang"][0]),
        "bodyOmg": np.zeros(3),
        "COMPos": np.copy(bot[EST]["COMPos"]),  # sair
        "COMVel": np.zeros(3),
    }

    for key in plan_data.keys():
        bot[PLAN][key]=plan_data[key]

    qp_setup = False  # is qp set up?

    tau_sol = np.zeros(10)
    ddq_sol = np.zeros(16)

    q_des = np.zeros(10)
    dq_des = np.zeros(10)
    ddq_des = np.zeros(10)
    tau_des = np.zeros(10)

    q_knee_max = -0.1

    danger_duration = [
        0.3,
        0.3,
    ]  # stop robot if in dangerous zone over 0.2 seconds, e.g., large tilt angle/lose foot contact
    danger_start_time = [0.0, 0.0]
    t0 = bot.get_time()
    thread_run = False
    while True:
        loop_start_time = bot.get_time()
        bot.setModuleState(MODULE_LOW, 1)

        # elapsed time
        elapsed_time = loop_start_time - t0

        # robot state update
        res2 = bot.update_leg_status()

        Bs = int(bot[PLAN]["robot_state"][0])

        PP = np.copy(P0)
        qp_q = np.copy(q0)

        _Q, _p = update_AngularMomentumCost()

        PP[0:16, 0:16] += _Q
        qp_q[0:16] += _p

        _Q, _p = update_LinearMomentumCost()

        PP[0:16, 0:16] += _Q
        qp_q[0:16] += _p

        _Q, _p = update_BaseOrientationCost()

        PP[0:3, 0:3] += _Q
        qp_q[0:3] += _p

        _Q, _p = update_RightStanceContactCost(
            np.vstack((bot[EST]["R_Jac_wToeVel"], bot[EST]["R_Jac_wHeelVel"])),
            np.hstack((bot[EST]["R_dJ_toeVel*dq"], bot[EST]["R_dJ_heelVel*dq"])),
            W_all[Bs, 3],
        )

        PP[0:16, 0:16] += _Q
        qp_q[0:16] += _p

        _Q, _p = update_LeftStanceContactCost(
            np.vstack((bot[EST]["L_Jac_wToeVel"], bot[EST]["L_Jac_wHeelVel"])),
            np.hstack((bot[EST]["L_dJ_toeVel*dq"], bot[EST]["L_dJ_heelVel*dq"])),
            W_all[Bs, 6],
        )

        PP[0:16, 0:16] += _Q
        qp_q[0:16] += _p




        _Q, _p = update_RightSwingPositionCost(
            bot[EST]["R_Jac_wAnkleVel"],
            bot[EST]["R_dJ_AnkleVel*dq"],
            W_all[Bs, 4],
            np.diag(Kp_all[Bs, 9:12]),
            np.diag(Kd_all[Bs, 9:12]),
            bot[EST]["R_anklePos"],
            bot[PLAN]["R_footPos"],
            bot[EST]["R_ankleVel"],
            bot[PLAN]["R_footVel"],
        )

        PP[0:16, 0:16] += _Q
        qp_q[0:16] += _p

        _Q, _p = update_RightSwingOrientationCost(
            bot[EST]["R_foot_Jw"][1:3, :],
            bot[EST]["R_foot_dJwdq"][1:3],
            W_all[Bs, 5],
            np.diag(Kp_all[Bs, 12:15]),
            np.diag(Kd_all[Bs, 12:15]),
            bot[EST]["R_footRot"],
            bot[PLAN]["R_footRot"],
            bot[EST]["R_footOmg"],
            bot[PLAN]["R_footOmg"],
        )

        PP[0:16, 0:16] += _Q
        qp_q[0:16] += _p

        _Q, _p = update_LeftSwingPositionCost(
            bot[EST]["L_Jac_wAnkleVel"],
            bot[EST]["L_dJ_AnkleVel*dq"],
            W_all[Bs, 7],
            np.diag(Kp_all[Bs, 15:18]),
            np.diag(Kd_all[Bs, 15:18]),
            bot[EST]["L_anklePos"],
            bot[PLAN]["L_footPos"],
            bot[EST]["L_ankleVel"],
            bot[PLAN]["L_footVel"],
        )

        PP[0:16, 0:16] += _Q
        qp_q[0:16] += _p

        _Q, _p = update_LeftSwingOrientationCost(
            bot[EST]["L_foot_Jw"][1:3, :],
            bot[EST]["L_foot_dJwdq"][1:3],
            W_all[Bs, 8],
            np.diag(Kp_all[Bs, 18:21]),
            np.diag(Kd_all[Bs, 18:21]),
            bot[EST]["L_footRot"],
            bot[PLAN]["L_footRot"],
            bot[EST]["L_footOmg"],
            bot[PLAN]["L_footOmg"],
        )

        PP[0:16, 0:16] += _Q
        qp_q[0:16] += _p



        AA, qp_l, qp_u = constraint_update(
            l0,
            u0,
            bot[EST]["H_matrix"][0:6, :],
            bot[EST]["CG_vector"][0:6],
            bot[EST]["R_Jac_wToeVel"][:, 0:6],
            bot[EST]["R_Jac_wHeelVel"][:, 0:6],
            bot[EST]["L_Jac_wToeVel"][:, 0:6],
            bot[EST]["L_Jac_wHeelVel"][:, 0:6],
            fz_max_all[Bs, 0],
            fz_min_all[Bs, 0],
            fz_max_all[Bs, 1],
            fz_min_all[Bs, 1],
        )

        qp_P = sparse.csc_matrix(PP)
        qp_A = sparse.csc_matrix(AA)

        if not qp_setup:
            # QP initialize
            prob = osqp.OSQP()
            prob.setup(
                P=sparse.triu(qp_P, format="csc"),
                q=qp_q,
                A=qp_A,
                l=qp_l,
                u=qp_u,
                verbose=False,
                warm_start=True,
                eps_abs=1e-3,
                eps_rel=1e-3,
                max_iter=1000,
                check_termination=1,
                adaptive_rho_interval=50,
                scaling=10,
            )
            qp_setup = True
            '''
            Minimize:
            0.5 * x^T * P * x + q^T * x

            Subject to:
            A * x <= u
            A * x >= l
            '''
        
        # QP solving
        prob.update(Px=sparse.triu(qp_P).data, q=qp_q, 
                    Ax=qp_A.data, 
                    l=qp_l, 
                    u=qp_u
                    )
        sol = prob.solve()

        if sol.info.status != "solved":
            # QP infeasible
            print(
                colored(
                    "OSQP did not solve the problem!!! "
                    + sol.info.status
                    + "!!! Te = "
                    + str(elapsed_time)[0:5]
                    + " s",
                    "red",
                )
            )
            sol.x *= 0.0
            exit()
        # QP solved
        ddq_sol = sol.x[0:16]#acceleration
        markerlist = []
        Frt = sol.x[16:19]
        Frh = sol.x[19:22]
        Flt = sol.x[22:25]
        Flh = sol.x[25:28]
        contactForceSMArr[0] = Frt
        contactForceSMArr[1] = Frh
        contactForceSMArr[2] = Flt
        contactForceSMArr[3] = Flh

        tau_sol = bot[EST]["H_matrix"][6:16, :] @ ddq_sol + bot[EST]["CG_vector"][6:16]
        tau_sol -= bot[EST]["R_Jac_wToeVel"][:, 6:16].T @ Frt
        tau_sol -= bot[EST]["R_Jac_wHeelVel"][:, 6:16].T @ Frh
        tau_sol -= bot[EST]["L_Jac_wToeVel"][:, 6:16].T @ Flt
        tau_sol -= bot[EST]["L_Jac_wHeelVel"][:, 6:16].T @ Flh


        for idx, joint_id in enumerate(LEG_JOINT_LIST):
            tau_des[idx] = MF.exp_filter(tau_des[idx], tau_sol[idx], 0.0)
            ddq_des[idx] = MF.exp_filter(ddq_des[idx], ddq_sol[idx + 6], 0.0)

            # Decayed joint value integration
            dq_des[idx] = (
                MF.exp_filter(dq_des[idx], 0.0, 0.0) + ddq_des[idx] * loop_duration
            )
            q_des[idx] = (
                MF.exp_filter(q_des[idx], bot.joint[joint_id]["q"], 0.0)
                + dq_des[idx] * loop_duration
            )

        # IK compensation
        if Bs == 1:
            worldBodyRot = bot[EST]["bodyRot"].T
            L_FootBodyPos = worldBodyRot @ (bot[PLAN]["L_footPos"] - bot[EST]["bodyPos"])
            L_bodyFootRot = worldBodyRot @ bot[PLAN]["L_footRot"]
            L_footVel = (
                bot[PLAN]["L_footVel"]
                - bot[EST]["bodyVel"]
                - bot[EST]["bodyRot"] @ MF.hat(bot[EST]["bodyOmg"]) @ L_FootBodyPos
            )
            ql1, ql2, ql3, ql4, ql5 = kin.legIK_ankle(
                L_FootBodyPos, L_bodyFootRot[:, 0], -1
            )
            dql = np.linalg.solve(
                np.vstack(
                    (
                        bot[EST]["L_Jac_wAnkleVel"][0:3, 11:15],
                        bot[EST]["L_foot_Jw"][2, 11:15],
                    )
                ),
                np.hstack((L_footVel, 0.0)),
            )
            q_des[5], q_des[6], q_des[7], q_des[8], q_des[9] = (
                ql1,
                ql2,
                ql3,
                ql4,
                ql5,
            )
            dq_des[5], dq_des[6], dq_des[7], dq_des[8], dq_des[9] = (
                dql[0],
                dql[1],
                dql[2],
                dql[3],
                0.0,
            )
        elif Bs == 2:
            worldBodyRot = bot[EST]["bodyRot"].T
            R_FootBodyPos = worldBodyRot @ (bot[PLAN]["R_footPos"] - bot[EST]["bodyPos"])
            R_bodyFootRot = worldBodyRot @ bot[PLAN]["R_footRot"]
            R_footVel = (
                bot[PLAN]["R_footVel"]
                - bot[EST]["bodyVel"]
                - bot[EST]["bodyRot"] @ MF.hat(bot[EST]["bodyOmg"]) @ R_FootBodyPos
            )
            qr1, qr2, qr3, qr4, qr5 = kin.legIK_ankle(
                R_FootBodyPos, R_bodyFootRot[:, 0], +1
            )
            dqr = np.linalg.solve(
                np.vstack(
                    (
                        bot[EST]["R_Jac_wAnkleVel"][0:3, 6:10],
                        bot[EST]["R_foot_Jw"][2, 6:10],
                    )
                ),
                np.hstack((R_footVel, 0.0)),
            )
            q_des[0], q_des[1], q_des[2], q_des[3], q_des[4] = (
                qr1,
                qr2,
                qr3,
                qr4,
                qr5,
            )
            dq_des[0], dq_des[1], dq_des[2], dq_des[3], dq_des[4] = (
                dqr[0],
                dqr[1],
                dqr[2],
                dqr[3],
                0.0,
            )

        if q_des[3] > q_knee_max:
            q_des[3] = q_knee_max

        if q_des[8] > q_knee_max:
            q_des[8] = q_knee_max

        for idx, joint_id in enumerate(LEG_JOINT_LIST):
            bot.joint[joint_id]["tau_goal"] = tau_des[idx]
            bot.joint[joint_id]["dq_goal"] = dq_des[idx]
            bot.joint[joint_id]["q_goal"] = q_des[idx]

        if elapsed_time > 0.5:
            # safety check, i.e., large tilt angle or lose contact -> robot is falling
            if np.arccos(bot[EST]["bodyRot"][2, 2]) > PI_4:
                if elapsed_time - danger_start_time[0] > danger_duration[0]:
                    # print(colored("Robot Large Tilt Angle! Terminate Now!", "red"))
                    # bot.damping_robot()
                    # bot.stop_threading()
                    pass
            else:
                danger_start_time[0] = elapsed_time

            if int(bot[PLAN]["robot_state"][0]) == 1 and not np.any(bot[EST]["foot_contacts"][0:2]):
                danger_duration[1] = 2#0.5
                if elapsed_time - danger_start_time[1] > danger_duration[1]:
                    print(
                        colored("Robot Losing Right Contact! Terminate Now!", "red")
                    )
                    # bot.damping_robot()
                    # bot.stop_threading()
                    pass
            elif int(bot[PLAN]["robot_state"][0]) == 2 and not np.any(bot[EST]["foot_contacts"][2:4]):
                danger_duration[1] = 2#0.5
                if elapsed_time - danger_start_time[1] > danger_duration[1]:
                    print(
                        colored("Robot Losing Left Contact! Terminate Now!", "red")
                    )
                    # bot.damping_robot()
                    # bot.stop_threading()
                    pass
            else:
                danger_start_time[1] = elapsed_time

            if elapsed_time > 1:
                if not thread_run:
                    MM.THREAD_STATE.set(
                        {"low_level": np.array([1.0])}, opt="only"
                    )  # thread is running
                    thread_run = True

                # check threading error
                if bot.thread_error():
                    bot.stop_threading()

            # send command
            if bot.getModuleState(MODULE_IFSIM):
                bot.set_command_leg_torques()
            else:
                bot.set_command_leg_values()
            bot.Thread_SMArr[0] = 1.0

        # check time to ensure that the whole-body controller stays at a consistent running loop.
        loop_end_time = loop_start_time + loop_duration
        present_time = bot.get_time()
        if present_time > loop_end_time:
            delay_time = 1000 * (present_time - loop_end_time)
            if delay_time > 1.0:
                print(
                    colored(
                        "Delayed "
                        + str(delay_time)[0:5]
                        + " ms at Te = "
                        + str(elapsed_time)[0:5]
                        + " s",
                        "yellow",
                    )
                )
        else:
            while bot.get_time() < loop_end_time:
                pass
            # while bot.get_time() < loop_end_time:
            #     print(bot.get_time(), loop_end_time)
            #     pass


def lowMain():
    main_loop()
    # try:
    #     main_loop()
    # except (NameError, KeyboardInterrupt) as error:
    #     MM.THREAD_STATE.set(
    #         {"low_level": np.array([0.0])}, opt="only"
    #     )  # thread is stopped
    # except Exception as error:
    #     print(error)
    #     MM.THREAD_STATE.set(
    #         {"low_level": np.array([2.0])}, opt="only"
    #     )  # thread in error
