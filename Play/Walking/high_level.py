#!usr/bin/env python
__author__    = "Westwood Robotics Corporation"
__email__     = "info@westwoodrobotics.io"
__copyright__ = "Copyright 2023 Westwood Robotics Corporation"
__date__      = "November 1, 2023"
__project__   = "BRUCE"
__version__   = "0.0.4"
__status__    = "Product"

"""
DCM-Based Footstep Planner
"""

import time
import osqp
import numpy as np
import Util.math_function as MF
from Settings.Bruce import *
import Startups.memory_manager as MM
from scipy import linalg
from scipy import sparse
from termcolor import colored
from Play.Walking.walking_macros import *
from snp import SharedNp

def cost_update(q0c, W_L, W_b, w_s,
                leg, Rd, cop, eTs, dy, N,
                Lox, Woy, box, boy1, boy2):
    q = np.copy(q0c)

    q[0:2] = -W_L @ (Rd @ np.array([Lox[0], Woy[0] + dy * leg]) + cop)
    q[2:4] = -W_b @ Rd @ np.array([box[0], boy1[0] - boy2[0] * leg])

    q[-1]  = -w_s * eTs

    for idx in range(N - 1):
        idx1 = idx + 1

        id1 = 4 * idx
        id2 = id1 + 2
        id3 = id1 + 4
        id4 = id1 + 6
        id5 = id1 + 8

        Avar = W_L @ Rd @ np.array([Lox[idx1], Woy[idx1] - dy * leg * (-1.) ** idx])
        q[id1:id2] += Avar
        q[id3:id4] -= Avar
        q[id4:id5]  = -W_b @ Rd @ np.array([box[idx1], boy1[idx1] + boy2[idx1] * leg * (-1.) ** idx])

    return q


def constraint_update(A0c, l0c, u0c,
                      leg, Rd, tau, cop, b_tau,
                      omg, eTs, N,
                      L_min_1, L_min_2, L_max_1, L_max_2):
    A = np.copy(A0c)
    l = np.copy(l0c)
    u = np.copy(u0c)

    RdT = np.copy(Rd.T)
    cop = np.copy(cop)

    # Equality Constraints
    # eq1 - DCM dynamics constraint - 2 * N cons
    A[0:2, -1] = b_tau * np.exp(-omg * tau)
    l[0:2] = -cop
    u[0:2] = l[0:2]

    I2 = np.eye(2)
    for idx in range(N - 1):
        A[2*idx+2:2*idx+4, 4*idx:4*idx+8] = np.kron(np.array([1., eTs, -1., -1.]), I2)

    # Inequality Constraints
    # ineq1 - kinematic reachability: L_min <= L_k <= L_max - 2 * N cons
    i1 = 2 * N
    i2 = i1 + 2
    A[i1:i2, 0:2] = RdT
    l[i1:i2] = RdT @ cop + L_min_1 + L_min_2 * leg
    u[i1:i2] = RdT @ cop + L_max_1 + L_max_2 * leg
    for idx in range(N - 1):
        id1 = 2 * idx + i2
        id2 = id1 + 2

        id3 = 4 * idx
        id4 = id3 + 2
        id5 = id4 + 2
        id6 = id5 + 2
        A[id1:id2, id3:id4] = -RdT
        A[id1:id2, id5:id6] =  RdT

        l[id1:id2] = L_min_1 - L_min_2 * leg * (-1.) ** idx
        u[id1:id2] = L_max_1 - L_max_2 * leg * (-1.) ** idx

    return A, l, u


def get_swing_traj(te, tar_t_sw, footTarPos, footTarVel, footTarAcc,
                   Ts_sol, p1_sol, pz0, pzf, p1_nom, pzm):
    # Calculate powers of the current time tp for polynomial evaluation.
    tp2 = tar_t_sw * tar_t_sw
    tp3 = tar_t_sw * tp2
    tp4 = tar_t_sw * tp3
    tp5 = tar_t_sw * tp4
    # Create matrices for position, velocity, and acceleration at current time tp.
    t_mat = np.array([[1.,  tar_t_sw,    tp2,    tp3,      tp4,      tp5],
                      [0.,  1, 2 * tar_t_sw, 3 * tp2,  4 * tp3,  5 * tp4],
                      [0.,  0,      2,  6 * tar_t_sw, 12 * tp2, 20 * tp3]])

    # x-coordinate polynomial coefficients calculation.
    Tx1 = Ts_sol - Txf  # Adjust time for final x-coordinates.
    # Powers of adjusted time for x.
    Tx2 = Tx1 * Tx1
    Tx3 = Tx1 * Tx2
    Tx4 = Tx1 * Tx3
    Tx5 = Tx1 * Tx4
    # Matrix for x-coordinate at final time.
    Tx_mat = np.array([[1., Tx1,     Tx2,      Tx3,      Tx4,      Tx5],
                       [0.,   1, 2 * Tx1,  3 * Tx2,  4 * Tx3,  5 * Tx4],
                       [0.,   0,       2,  6 * Tx1, 12 * Tx2, 20 * Tx3]])
    # Combine current and final time matrices for x.
    Txf_mat = np.vstack((t_mat, Tx_mat))

    # Calculate x-coordinate coefficients based on time condition.
    if te < Txn:
        cxo = np.linalg.solve(Txf_mat, np.array([footTarPos[0], footTarVel[0], footTarAcc[0], p1_nom[0], 0., 0.]))
    else:
        cxo = np.linalg.solve(Txf_mat, np.array([footTarPos[0], footTarVel[0], footTarAcc[0], p1_sol[0], 0., 0.]))

    # y-coordinate polynomial coefficients calculation similar to x-coordinate.
    Ty1 = Ts_sol - Tyf
    # Powers and matrix for y-coordinate.
    Ty2 = Ty1 * Ty1
    Ty3 = Ty1 * Ty2
    Ty4 = Ty1 * Ty3
    Ty5 = Ty1 * Ty4
    Ty_mat = np.array([[1., Ty1,     Ty2,      Ty3,      Ty4,      Ty5],
                       [0.,   1, 2 * Ty1,  3 * Ty2,  4 * Ty3,  5 * Ty4],
                       [0.,   0,       2,  6 * Ty1, 12 * Ty2, 20 * Ty3]])
    Tyf_mat = np.vstack((t_mat, Ty_mat))

    # Calculate y-coordinate coefficients based on time condition.
    if te < Tyn:
        cyo = np.linalg.solve(Tyf_mat, np.array([footTarPos[1], footTarVel[1], footTarAcc[1], p1_nom[1], 0., 0.]))
    else:
        cyo = np.linalg.solve(Tyf_mat, np.array([footTarPos[1], footTarVel[1], footTarAcc[1], p1_sol[1], 0., 0.]))

    # z-coordinate polynomial coefficients calculation.
    # Initialize coefficient array for two parts of z trajectory.
    czo = np.zeros((2, 6))
    # Powers for apex time of z-coordinate.
    Tzm2 = Tzm * Tzm
    Tzm3 = Tzm * Tzm2
    Tzm4 = Tzm * Tzm3
    Tzm5 = Tzm * Tzm4
    # Matrix for z-coordinate at apex time.
    Tzm_mat = np.array([[1.,   0,       0,        0,         0,         0],
                        [0.,   1,       0,        0,         0,         0],
                        [0.,   0,       2,        0,         0,         0],
                        [1., Tzm,    Tzm2,     Tzm3,      Tzm4,      Tzm5],
                        [0.,   1, 2 * Tzm, 3 * Tzm2,  4 * Tzm3,  5 * Tzm4],
                        [0.,   0,       2,  6 * Tzm, 12 * Tzm2, 20 * Tzm3]])

    # Solve for the first part of z-coordinate coefficients up to apex.
    czo[0, :] = np.linalg.solve(Tzm_mat, np.array([pz0, 0., 0., pzm, 0., 0.]))

    # Prepare for the second part of the z trajectory, from apex to final position.
    Tz1 = Ts_sol - Tzf
    # Powers of adjusted time for z.
    Tz2 = Tz1 * Tz1
    Tz3 = Tz1 * Tz2
    Tz4 = Tz1 * Tz3
    Tz5 = Tz1 * Tz4
    # Matrix for z-coordinate from apex to final position.
    Tz_mat = np.array([[1., Tz1,     Tz2,      Tz3,      Tz4,      Tz5],
                       [0.,   1, 2 * Tz1,  3 * Tz2,  4 * Tz3,  5 * Tz4],
                       [0.,   0,       2,  6 * Tz1, 12 * Tz2, 20 * Tz3]])

    # Depending on the current time `tp`, solve for the second part of z-coordinate coefficients.
    if tar_t_sw < Tzm:
        Tzf_mat = np.vstack((Tzm_mat[3:6, :], Tz_mat))
        czo[1, :] = np.linalg.solve(Tzf_mat, np.array([pzm, 0., 0., pzf, 0., 0.]))
    else:
        Tzf_mat = np.vstack((t_mat, Tz_mat))
        czo[1, :] = np.linalg.solve(Tzf_mat, np.array([footTarPos[2], footTarVel[2], footTarAcc[2], pzf, 0., 0.]))

    # Return the calculated polynomial coefficients for each coordinate.
    return cxo, cyo, czo

def get_swing_ref(cxo, cyo, czo,
                  p0, pzf, p1, T, t,):
    # Check if the current time is within the step duration
    if t < T:
        # Calculate the time powers used for polynomial evaluation
        t2  = t * t
        t3  = t * t2
        t4  = t * t3
        t5  = t * t4
        # Polynomial terms for position, velocity, and acceleration
        t_p = np.array([1., t,    t2,     t3,      t4,      t5])
        t_v = np.array([0., 1, 2 * t, 3 * t2,  4 * t3,  5 * t4])
        t_a = np.array([0., 0,     2,  6 * t, 12 * t2, 20 * t3])

        # Initialize arrays to store position, velocity, and acceleration
        pt, vt, at = np.zeros(3), np.zeros(3), np.zeros(3)

        # X-coordinate trajectory planning
        if t <= Txi:
            pt[0] = p0[0]  # Initial x position if before start moving time
        elif Txi < t < T - Txf:
            # Calculate x position, velocity, and acceleration during movement
            pt[0] = cxo @ t_p
            vt[0] = cxo @ t_v
            at[0] = cxo @ t_a
        else:
            pt[0] = p1[0]  # Final x position after movement time

        # Y-coordinate trajectory planning
        if t <= Tyi:
            pt[1] = p0[1]  # Initial y position if before start moving time
        elif Tyi < t < T - Tyf:
            # Calculate y position, velocity, and acceleration during movement
            pt[1] = cyo @ t_p
            vt[1] = cyo @ t_v
            at[1] = cyo @ t_a
        else:
            pt[1] = p1[1]  # Final y position after movement time

        # Z-coordinate trajectory planning
        if t <= Tzm:
            # Calculate z position, velocity, and acceleration before swing apex
            pt[2] = czo[0, :] @ t_p
            vt[2] = czo[0, :] @ t_v
            at[2] = czo[0, :] @ t_a
        elif Tzm < t < T - Tzf:
            # Calculate z position, velocity, and acceleration after swing apex
            pt[2] = czo[1, :] @ t_p
            vt[2] = czo[1, :] @ t_v
            at[2] = czo[1, :] @ t_a
        else:
            # Final z position, velocity, and acceleration after movement time
            at[2] = 0.
            vt[2] = 0.
            pt[2] = pzf
    else:
        # Post-step adjustments for final positioning
        vzf = -0.1  # Final z velocity
        at  = np.array([0., 0., 0.0])  # Final acceleration is zero
        vt  = np.array([0., 0., vzf])  # Set final z velocity
        pt  = np.array([p1[0], p1[1], pzf + vzf * (t - T)])  # Adjust final z position based on overstep time

    # Return the calculated position, velocity, and acceleration
    return pt, vt, at



def get_DCM_offset(pg_xy, vg_xy, pa_xy, omg):
    xi    = pg_xy + vg_xy / omg
    cop   = pa_xy
    b_tau = xi - cop

    return cop, b_tau




printRec={}

def printPlanData(key, freq, plan_data):
    if key not in printRec:
        printRec[key]=0
    printRec[key]+=1
    if printRec[key]>freq:
        printRec[key]=0
        print(key,plan_data)


smoothDict={
    "bodyRot":0.001,
    "COMPos":0.001,
}
def setPlanData(bot:BRUCE, plan_data):
    for key in plan_data:
        if key in smoothDict:
            org=bot.getPln(key)
            new=org+(plan_data[key]-org)*smoothDict[key]
            bot.setPlan(key,new)
        else:
            bot.setPlan(key,plan_data[key])

def main_loop():
    SN = SharedNp("shm1.json")
    # BRUCE SETUP
    bot = BRUCE()

    # CONTROL FREQUENCY
    loop_freq   = 1000  # run at 1000 Hz
    update_freq =  500  # DCM replanning frequency

    loop_duration   = 1. / loop_freq
    update_duration = 1. / update_freq

    # NOMINAL GAIT PATTERN
    nfreq = np.sqrt(9.81 / hz)  # LIPM natural frequency
    wTs = nfreq * Ts_desired
    eTs = np.exp(wTs)

    # COST WEIGHTS
    W_L = np.diag([  1.,   1.])
    W_b = np.diag([100., 100.])
    w_s = 0.1

    # QP SETUP
    # Decision Variables
    # x = [p1, b1, p2, b2, ... , pN, bN, es] - 2 * 2 * N + 1 = 4N + 1
    # pk - 2 (foothold location)
    # bk - 2 (initial DCM offset)
    # es - 1 (stance phase duration)
    N = 3           # number of steps
    Nv = 4 * N + 1  # number of decision variables

    # kinematic reachability [m]
    L_max_l = np.array([ lx, -lyi])
    L_min_l = np.array([-lx, -lyo])
    L_max_r = np.array([ lx,  lyo])
    L_min_r = np.array([-lx,  lyi])

    # Costs
    P0 = linalg.block_diag(W_L, W_b, np.zeros((Nv - 5, Nv - 5)), w_s)
    for idx in range(N - 1):
        id1 = 4 * idx
        id2 = id1 + 6
        P0[id1:id2, id1:id2] += np.kron(np.array([[ 1., 0., -1.],
                                                  [ 0., 0.,  0.],
                                                  [-1., 0.,  1.]]), W_L)
        id3 = id1 + 6
        id4 = id1 + 8
        P0[id3:id4, id3:id4] = W_b

    q0     = np.zeros(Nv)
    q0[-1] = -w_s * eTs

    # Equality Constraints
    # eq1 - DCM dynamics constraint: xi_k+1 = p_k + b_k * exp(w * Ts) - 2 * N cons
    Aeq           =  np.zeros((2 * N, Nv))
    Aeq[0:2, 0:2] = -np.eye(2)
    Aeq[0:2, 2:4] = -np.eye(2)
    Aeq[0:2, -1]  =  np.array([1., 1.])
    for idx in range(N - 1):
        Aeq[2*idx+2:2*idx+4, 4*idx:4*idx+8] = np.kron(np.array([1., eTs, -1., -1.]), np.eye(2))

    beq = np.zeros(2 * N)

    # Inequality Constraints
    # ineq1 - kinematic reachability: L_min <= L_k <= L_max - 2 * N cons
    L_max_1 = (L_max_r + L_max_l) / 2.
    L_max_2 = (L_max_r - L_max_l) / 2.
    L_min_1 = (L_min_r + L_min_l) / 2.
    L_min_2 = (L_min_r - L_min_l) / 2.

    Aineq1 = np.zeros((2 * N, Nv))
    Aineq1[0:2, 0:2] = np.ones((2, 2))

    for idx in range(N - 1):
        Aineq1[2*idx+2:2*idx+4, 4*idx:4*idx+6] = np.kron(np.array([-1., 0., 1.]), np.ones((2, 2)))

    bineq1 = np.zeros(2 * N)

    # ineq2 - phase duration limit: exp(w * Ts_min) <= es <= exp(w * Ts_max) - 1 con
    Aineq2 = np.zeros(Nv)
    Aineq2[-1] = 1.

    bineq2_l = np.exp(nfreq * Ts_min)#constant
    bineq2_u = np.exp(nfreq * Ts_max)#constant

    # Overall
    A0 = np.vstack((Aeq, Aineq1, Aineq2))
    l0 = np.hstack((beq, bineq1, bineq2_l))
    u0 = np.hstack((beq, bineq1, bineq2_u))

    qp_P = sparse.csc_matrix(P0)
    qp_A = sparse.csc_matrix(A0)

    # OSQP setup
    prob = osqp.OSQP()
    prob.setup(P=sparse.triu(qp_P, format="csc"), q=q0, A=qp_A, l=l0, u=u0, verbose=False, warm_start=True,
               eps_abs=1e-3, eps_rel=1e-3, max_iter=1000, check_termination=1, adaptive_rho_interval=50, scaling=10)

    # START CONTROL
    # confirm = input("Start DCM Walking? (y/n) ")
    # if confirm != "y":
    #     exit()

    input_data = {"walk": np.zeros(1),
                  "com_xy_velocity": np.zeros(2),
                  "yaw_rate": np.zeros(1)}
    MM.USER_COMMAND.set(input_data)

    plan_data = {"robot_state": np.zeros(1),
                 "bodyRot": np.copy(bot.R_yaw())}
    # robot state - balance 0
    #             - right stance 1
    #             - left  stance 2

    leg_st = -1  # right stance +1
                 # left  stance -1

    T0 = bot.get_time()
    thread_run = False
    
    
    # vxd  = np.zeros(N)
    # vyd  = np.zeros(N)
    Lox  = np.zeros(N)
    Woy  = np.zeros(N)
    box  = np.zeros(N)
    boy1 = np.zeros(N)
    boy2 = np.zeros(N)

    pxyd  = np.copy(bot[EST]["COMPos"][0:2])
    yawd  = bot[EST]["body_yaw_ang"][0]
    footTarRotRate = 0.

    step_num = 0
    pitch_angle = 0.1

    while True:
        # user input update
        bot.update_input_status()


        # STOP WALKING AND KEEP BALANCING
        # check if command stop walking and actual CoM velocities are small
        if bot.walk == 0. and np.sqrt(bot[EST]["COMVel"][0] ** 2 + bot[EST]["COMVel"][1] ** 2) <= 0.20 and MF.norm(bot[EST]["R_footPos"] - bot[EST]["L_footPos"]) >= 0.05:
            # stop walking and move CoM to center
            plan_data["robot_state"] = np.zeros(1)
            comPos_0 = np.copy(bot[EST]["COMPos"])
            feet_center = 0.5 * (bot[EST]["R_footPos"] + bot[EST]["L_footPos"])
            x_pos_dir = 0.5 * (bot[EST]["R_toePos"] + bot[EST]["L_toePos"]) - feet_center  # forward direction
            x_neg_dir = 0.5 * (bot[EST]["R_heelPos"] + bot[EST]["L_heelPos"]) - feet_center  # backward direction
            comPos_1    = feet_center + ka * x_pos_dir
            comPos_1[2] += hz

            pitch_angle = 0.1
            bodyRot_0 = np.copy(bot[EST]["bodyRot"])
            phi_0  = MF.logvee(bodyRot_0)
            bodyRot_1 = MF.Rz(bot[EST]["body_yaw_ang"][0]) @ MF.Ry(pitch_angle)
            phi_1  = MF.logvee(bodyRot_1)

            t1 = 0.05
            te = 0.
            t0 = bot.get_time()
            while te <= t1:
                te = bot.get_time() - t0
                plan_data["COMPos"]    = comPos_0 + (comPos_1 - comPos_0) / t1 * te
                printPlanData("D",10,plan_data["COMPos"])
                plan_data["COMVel"]    = np.zeros(3)
                plan_data["bodyRot"] = MF.hatexp(phi_0 + (phi_1 - phi_0) / t1 * te)

                setPlanData(bot,plan_data)
                time.sleep(0.001)

            # keep balancing until command walking
            comPos_0 = np.copy(comPos_1)
            bodyRot_0 = np.copy(bodyRot_1)
            while bot.walk == 0.:
                Te = bot.get_time() - T0

                if Te > 1:
                    if not thread_run:
                        MM.THREAD_STATE.set({"high_level": np.array([1.0])}, opt="only")  # thread is running
                        thread_run = True

                    # check threading error
                    if bot.thread_error():
                        bot.stop_threading()

                # user input update
                bot.update_input_status()

                # manipulating CoM
                # lateral
                y_pos_dir = bot[EST]["L_footPos"] - feet_center   # left  direction
                y_neg_dir = bot[EST]["R_footPos"] - feet_center   # right direction
                if bot.cmd_comPos_change[1] >= 0.:
                    dp_xy = y_pos_dir[0:2] * +bot.cmd_comPos_change[1]
                else:
                    dp_xy = y_neg_dir[0:2] * -bot.cmd_comPos_change[1]

                # longitudinal
                if bot.cmd_comPos_change[0] >= 0.:
                    dp_xy += x_pos_dir[0:2] * +bot.cmd_comPos_change[0]
                else:
                    dp_xy += x_neg_dir[0:2] * -bot.cmd_comPos_change[0]

                # vertical
                dp_z = bot.cmd_comPos_change[2]

                plan_data["COMPos"] = comPos_0 + np.array([dp_xy[0],
                                                               dp_xy[1],
                                                               dp_z])

                # manipulating body orientation
                plan_data["bodyRot"] = bodyRot_0 @ bot.cmd_R_change
                printPlanData("C",400,plan_data["COMPos"])
                setPlanData(bot,plan_data)
                time.sleep(0.001)

            # shift CoM for walking a
            # gain
            comPos_1 = np.copy(comPos_0)
            if leg_st == +1.:
                p_st_xy = bot[EST]["R_footPos"][0:2]
            elif leg_st == -1.:
                p_st_xy = bot[EST]["L_footPos"][0:2]
            comPos_1[0:2] += (p_st_xy - comPos_1[0:2]) * 0.70 + ka * x_pos_dir[0:2]
            t1 = 0.8
            te = 0.
            t0 = bot.get_time()
            while te <= t1:
                te = bot.get_time() - t0
                plan_data["COMPos"] = comPos_0 + (comPos_1 - comPos_0) / t1 * te
                printPlanData("B",200,plan_data["COMPos"])
                plan_data["COMVel"] = np.zeros(3)

                smoothDict["COMPos"]=1
                setPlanData(bot,plan_data)
                time.sleep(0.001)

            # initialize walking parameters
            # each step can have different velocities
            # vxd  = np.zeros(N)
            # vyd  = np.zeros(N)
            Lox  = np.zeros(N)
            Woy  = np.zeros(N)
            box  = np.zeros(N)
            boy1 = np.zeros(N)
            boy2 = np.zeros(N)

            pxyd  = np.copy(bot[EST]["COMPos"][0:2])
            yawd  = bot[EST]["body_yaw_ang"][0]
            footTarRotRate = 0.

            step_num = 0

        # initialize current cycle
        leg_st *= -1  # desired stance leg
        Ts_sol  = Ts_desired  # optimal Ts [s]
        te      = 0.  # phase elapsed time [s]

        step_num += 1

        if step_num > 1:
            plan_data["robot_state"] = np.array([1.5 - 0.5 * leg_st])

        bot.cmd_vxy_wg*=0
        # nominal pattern
        sf = 0.5  # smoothing factor
        # vxd[0:N-1] = vxd[1:N]
        # vxd[N-1]   = MF.exp_filter(vxd[N-1], bot.cmd_vxy_wg[0], sf)

        # vyd[0:N-1] = vyd[1:N]
        # vyd[N-1]   = MF.exp_filter(vyd[N-1], bot.cmd_vxy_wg[1], sf)

        # print(vxd,vyd)

        #pxyd += bot.R_yaw()[0:2, 0:2] @ np.array([vxd[0], vyd[0]]) * Ts
        pxyd += bot.R_yaw()[0:2, 0:2] @ np.array([0, 0]) * Ts_desired
        #print(pxyd)

        footTarRotRate = MF.exp_filter(footTarRotRate, bot.cmd_yaw_rate, sf)
        yawd += footTarRotRate * Ts_desired
        Rd    = MF.Rz(yawd) @ MF.Ry(pitch_angle) @ bot.cmd_R_change

        dyt = bot.cmd_dy
        #print(dyt)

        if leg_st == +1.:
            footPos_0 = np.copy(bot[EST]["L_anklePos"])
            footTarYaw = MF.Rz(yawd + bot.cmd_yaw_l_change + yaw_f_offset)
            SN["ifLeft"][:] = 1
        elif leg_st == -1.:
            footPos_0 = np.copy(bot[EST]["R_anklePos"])
            footTarYaw = MF.Rz(yawd + bot.cmd_yaw_r_change - yaw_f_offset)
            SN["ifLeft"][:] = 0

        p1_sol = np.copy(footPos_0)
        p1_nom = np.copy(p2_sol) if step_num > 1 else np.copy(footPos_0)  # current nominal position is the previous solution

        tar_t_sw, footTarPos, footTarVel, footTarAcc = te, footPos_0, np.zeros(3), np.zeros(3)
        tu = 0.  # swing update timer

        t0 = bot.get_time()








        while True:  # continue the current cycle
            loop_start_time = bot.get_time()

            Te = loop_start_time - T0
            te = loop_start_time - t0
            tr = Ts_sol - te

            # check threading error
            if bot.thread_error():
                bot.stop_threading()

            # robot state update
            Rt = bot.R_yaw()[0:2, 0:2]

            # stop current cycle
            contact_con = (te >= Ts_min and (bot[EST]["foot_contacts"][int(leg_st+1)] or bot[EST]["foot_contacts"][int(leg_st+2)])) or (te >= Ts_max)
            if contact_con:  # stop current cycle if touchdown
                break

            if leg_st == +1.:
                p_st_m = bot[EST]["R_footPos"]
                p_st_a = bot[EST]["R_anklePos"]
            elif leg_st == -1.:
                p_st_m = bot[EST]["L_footPos"]
                p_st_a = bot[EST]["L_anklePos"]


            pz0 = footPos_0[2]
            # pzm = p_st_a[2] + zm
            pzm = pz0 + z_max
            if bot.getModuleState(MODULE_IFSIM):
                pzf = p_st_a[2] + z_final
            else:
                # pzf = p_st_a[2] + zf
                pzf = pz0 + z_final

            omgt = nfreq
            eTst = np.exp(omgt * Ts_desired)

            def get_DCM_nom(vxd, vyd, dy, Ts, eTs):
                Lox = vxd * Ts  # nominal step difference [m]
                Woy = vyd * Ts  # nominal pelvis movement [m]

                # nominal initial DCM offset
                box  = Lox / (eTs - 1.)
                boy1 = Woy / (eTs - 1.)
                boy2 =  dy / (eTs + 1.)

                return Lox, Woy, box, boy1, boy2
            
            for i in range(N):
                # vxd=0
                # vyd=0
                #Lox[i], Woy[i], box[i], boy1[i], boy2[i] = get_DCM_nom(vxd[i], vyd[i], dyt, Ts, eTs)
                Lox[i], Woy[i], box[i], boy1[i], boy2[i] = get_DCM_nom(0, 0, dyt, Ts_desired, eTs)

                #print("X",SN["offsetX"],"Y",SN["offsetY"])
                bx_offset = SN["boffset"][0]
                box[i]  += bx_offset
                # boy1[i] += by_offset
                # boy2[i] += by_offset

            p_tau, b_tau = get_DCM_offset(bot[EST]["COMPos"][0:2], bot[EST]["COMVel"][0:2], p_st_a[0:2], omgt)

            if tr >= T_buff and te >= tu:  # constantly planning (at update_freq Hz) if tr larger than T_buff
                # qp update
                qp_q = cost_update(q0, W_L, W_b, w_s,
                                   leg_st, Rt, p_tau, eTst, dyt, N,
                                   Lox, Woy, box, boy1, boy2)
                AA, qp_l, qp_u = constraint_update(A0, l0, u0, leg_st, Rt, te, p_tau, b_tau,
                                                   omgt, eTst, N,
                                                   L_min_1, L_min_2, L_max_1, L_max_2)
                qp_A = sparse.csc_matrix(AA)

                # qp solve
                prob.update(Ax=qp_A.data, q=qp_q, l=qp_l, u=qp_u)
                sol = prob.solve()

                if sol.info.status != "solved":
                    # qp infeasible
                    print(colored("OSQP did not solve the problem!!!", "red"))
                else:
                    # qp solved
                    tu += update_duration

                    p1_sol = sol.x[0:2]                # current step location
                    p2_sol = sol.x[4:6]                # next    step location

                    #print(p1_sol)
                    p1_sol[1] += -SN["boffset"][1]
                    #p1_sol = 0.8* p1_sol + 0.2 * SN["footLoc"][:2]
                    SN["footLoc"][:2] = p1_sol
                    #p1_sol[0] -= SN["boffset"][0] / 30

                    Ts_sol = np.log(sol.x[-1]) / omgt  # current step timing

                    if Ts_sol < Ts_min:
                        Ts_sol = Ts_min

                cx, cy, cz = get_swing_traj(te, tar_t_sw, footTarPos, footTarVel, footTarAcc,
                                            Ts_sol, p1_sol, pz0, pzf, p1_nom, pzm,)

            # compute references for swing foot
            footTarPos, footTarVel, footTarAcc = get_swing_ref(cx, cy, cz,
                                                         footPos_0, pzf, p1_sol, Ts_sol, te)


            tar_t_sw = te

            # set planner command
            if step_num > 1:
                plan_data["bodyRot"] = Rd
                plan_data["bodyOmg"]   = np.array([0., 0., footTarRotRate])

                plan_data["COMPos"] = np.array([pxyd[0], pxyd[1], p_st_m[2] + hz + bot.cmd_comPos_change[2]])
                #plan_data["COMVel"] = bot.R_yaw() @ np.array([vxd[0], vyd[0], 0.])
                plan_data["COMVel"] = bot.R_yaw() @ np.array([0,0, 0.])

                if leg_st == +1.:
                    plan_data["L_footPos"]   = footTarPos
                    plan_data["L_footVel"]   = footTarVel
                    plan_data["L_footRot"]   = footTarYaw
                    plan_data["L_footOmg"]   = np.array([0., 0., footTarRotRate])
                elif leg_st == -1.:
                    plan_data["R_footPos"]   = footTarPos
                    plan_data["R_footVel"]   = footTarVel
                    plan_data["R_footRot"]   = footTarYaw
                    plan_data["R_footOmg"]   = np.array([0., 0., footTarRotRate])

                setPlanData(bot,plan_data)

            # loop time
            loop_target_time = loop_start_time + loop_duration
            present_time = bot.get_time()
            if present_time > loop_target_time:
                delay_time = 1000. * (present_time - loop_target_time)
                if delay_time > 1.:
                    print(colored("Delayed " + str(delay_time)[0:5] + " ms at Te = " + str(Te)[0:5] + " s", "yellow"))
            else:
                while bot.get_time() < loop_target_time:
                    pass


def highMain():
    main_loop()
    # try:
    #     main_loop()
    # except (NameError, KeyboardInterrupt) as error:
    #     MM.THREAD_STATE.set({"high_level": np.array([0.0])}, opt="only")  # thread is stopped
    #     print(error)
    # except Exception as error:
    #     print(error)
    #     MM.THREAD_STATE.set({"high_level": np.array([2.0])}, opt="only")  # thread in error
