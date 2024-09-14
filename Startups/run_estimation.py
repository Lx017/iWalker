#!usr/bin/env python
__author__    = "Westwood Robotics Corporation"
__email__     = "info@westwoodrobotics.io"
__copyright__ = "Copyright 2023 Westwood Robotics Corporation"
__date__      = "November 1, 2023"
__project__   = "BRUCE"
__version__   = "0.0.4"
__status__    = "Product"

"""
Script usage:
1. estimate robot states, e.g., body orientation, angular velocity, position, velocity, foot position, velocity, etc.
2. calculate robot model, e.g., kinematics (Jacobian and its derivative) and dynamics (equations of motion)
"""

import time
from Settings.Bruce import *
import Startups.memory_manager as MM
import Library.ROBOT_MODEL.BRUCE_dynamics as dyn
import Library.ROBOT_MODEL.BRUCE_kinematics as kin
from termcolor import colored
from Settings.BRUCE_macros import *
from Play.Walking.walking_macros import *
from Library.BRUCE_SENSE import Manager as sense_manager
import matplotlib.pyplot as plt
import scipy
import json
from pypose.module import EKF
from lxutil import RotMatZ

from snp import SharedNp
SN = SharedNp("shm1.json")

ROT_OFFSET = np.eye(3)
ROT_CORRECT = np.eye(3)
ROOT_OFFSET = np.array([0.,0.,0.])
YAW_OFFSET = 0
ROOT_MEA = np.zeros(3)

def IMU_check(sm):
    import statistics

    print("Checking IMU ...")
    IMU_error = True
    data = []
    data_std = []

    # Setup the plot
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot(data_std)
    plt.ylabel("Standard Deviation")
    plt.xlabel("Measurement Number")
    plt.title("IMU Standard Deviation Over Time")
    lastreset=time.time()

    def update_plot():
        line.set_ydata(data_std)
        line.set_xdata(range(len(data_std)))
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.1)

    while IMU_error:
        if sm.read_data():
            data.append(sm.accel[0])

        if len(data) > 300:
            # Calculate standard deviation
            current_std = statistics.stdev(data)
            data_std.append(current_std)
            data.pop(0)
            update_plot()
            print(current_std)
            data=[]

            if current_std > 0.4:

                if time.time()-lastreset>1:
                    print(colored("IMU Error! Resetting ...", "red"))
                    sm.send_data(pico_mode="reset")
                    time.sleep(0.5)
                    sm.send_data(pico_mode="nominal")
                    print("Checking IMU Again ...")
                    lastreset=time.time()
                # Add your error handling and reset logic here
            else:
                print("IMU OK!")
                IMU_error = False



def contact_check(sm):
    print("Checking Foot Contact ...")
    count = 0
    while count < 100:
        # get sensor data
        if sm.read_data():
            if np.any(sm.foot_contact[0:2]) or np.any(sm.foot_contact[2:4]):
                count += 1
    print("Foot On Ground!")


# FREQUENCY SETTING
freq  = 500  # run at 500 Hz
dt    = 1. / freq
dt2   = dt * dt
dt2_2 = dt2 / 2.

# BODY POSITION AND VELOCITY
# constant
ns = 3 * 5  # number of states
nn = 3 * 4  # number of process noises
I3 = np.eye(3)
Is = np.eye(ns)

# model
A = np.copy(Is)
A[0:3, 3:6] = I3 * dt

B = np.zeros((ns, 3))
B[0:3, 0:3] = I3 * dt2_2
B[3:6, 0:3] = I3 * dt

Gam = np.zeros((ns, nn))
Gam[6:ns, 3:nn] = np.eye(ns - 6) * dt

# covariance
Qa  = 1e-1**2 * np.diag([1., 1., 1.])  # accelerometer noise
Qba = 1e-4**2 * np.diag([1., 1., 1.])  # accelerometer bias noise
Qc0 = 1.e5**2 * np.diag([1., 1., 1.])  # new foot contact noise
Qc1 = 1e-3**2 * np.diag([1., 1., 1.])  # foot contact noise
Q   = scipy.linalg.block_diag(Qa, Qba, Qc1, Qc1)

Vp  = 1e-3**2 * np.diag([1., 1., 1.])  # foot position FK noise
Vv  = 1e-2**2 * np.diag([1., 1., 1.])  # foot velocity FK noise

def test_input_like(array: np.ndarray):
    array=np.array(array)
    """ Create a numpy array of the specified shape and dtype """
    return np.array(np.random.rand(*array.shape)).astype(array.dtype)

estRunCount=0

def run(bodyRot0, IMUOmg0,
        bodyAcc0, x0,
        P0,z1,
        IMUOmg1, IMUAcc, contacts_count, imu_static, g, kR):
    global estRunCount

    # SETTING
    # constant
    contacts_num = int(np.sum(contacts_count > 0))  # number of foot contacts

    # covariance reset for new foot contacts
    for i in range(2):  # now we only consider one contact for each foot
        if contacts_count[i] == 1:
            id1 = 3 * i + 9
            id2 = id1 + 3
            id3 = id2 + 3

            P0[id1:id2,    0:ns] = np.zeros((3, ns))
            P0[   0:ns, id1:id2] = np.zeros((ns, 3))
            P0[id1:id2, id1:id2] = Qc0

            x0[id1:id2] = z1[id2:id3]

    # ORIENTATION ESTIMATE
    # predict
    bodyRot1 = np.copy(bodyRot0) @ MF.hatexp(IMUOmg0 * dt)

    # update
    if imu_static:  # only when IMU is static
        gm   = bodyRot1 @ np.copy(IMUAcc)
        gmu  = gm / MF.norm(gm)
        dF = np.arccos(gmu[2] * np.sign(g))
        if np.abs(dF) > 1e-10:
            nv = MF.hat(gmu) @ np.array([0., 0., np.sign(g)]) / np.sin(dF)  # rotation axis
            bodyRot1 = MF.hatexp(kR * dF * nv) @ bodyRot1
            #print("update")

    yaw = np.arctan2(bodyRot1[1, 0], bodyRot1[0, 0])
    RT  = bodyRot1.T
    wRT = MF.hat(IMUOmg1) @ RT

    # POSITION AND VELOCITY ESTIMATE
    # predict
    Ak = np.copy(A)
    Ak[0:3, 6:9] = -bodyRot0 * dt2_2
    Ak[3:6, 6:9] = -bodyRot0 * dt

    Gamk = np.copy(Gam)
    Gamk[0:6, 0:3] = Ak[0:6, 6:9]

    P0 = Ak @ P0 @ Ak.T + Gamk @ Q @ Gamk.T

    x1 = np.copy(x0)
    x1[0:3] = x0[0:3] + x0[3:6] * dt + bodyAcc0 * dt2_2
    x1[3:6] = x0[3:6] + bodyAcc0 * dt

    estRunCount+=1
    # update
    if contacts_num > 0:
        nm = 6 * contacts_num  # current number of measurements
        Ck = np.zeros((nm, ns))
        Vk = np.zeros((nm, nm))
        yk = np.zeros(nm)      # innovation
        j  = 0
        for i in range(2):
            if contacts_count[i] > 0:
                id1 = 3 * j
                id2 = id1 + 3
                id3 = id1 + 3 * contacts_num
                id4 = id3 + 3

                id5 = 3 * i
                id6 = id5 + 3
                id7 = id6 + 3
                id8 = id7 + 3
                id9 = id8 + 3

                Ck[id1:id2,     0:3] = -RT
                Ck[id1:id2, id8:id9] =  RT
                Ck[id3:id4,     0:3] =  wRT
                Ck[id3:id4,     3:6] = -RT
                Ck[id3:id4, id8:id9] = -wRT

                Vk[id1:id2, id1:id2] = Vp
                Vk[id3:id4, id3:id4] = Vv

                ci_p = x1[id8:id9] - x1[0:3]#where contact ankle is used
                yk[id1:id2] = z1[id5:id6] -  RT @ ci_p#anke
                yk[id3:id4] = z1[id7:id8] + wRT @ ci_p + RT @ x1[3:6]

                j += 1

        Sk     = Ck @ P0 @ Ck.T + Vk
        Kk     = P0 @ Ck.T @ np.linalg.pinv(Sk)
        #IKC    = Is - Kk @ Ck
        #P1     = IKC @ P1 @ IKC.T + Kk @ Vk @ Kk.T
        P1     = (Is - Kk @ Ck) @ P0

        x1    += Kk @ yk

    #bodyPos1   = x1[0:3]
    # bodyVel1   = x1[3:6]
    # IMUAccBias  = x1[6:9]#not sure
    # bodyAcc1  = bodyRot1 @ (IMUAcc - IMUAccBias) - gv  # body acceleration excluding gravity
    # bodyVel_b = RT @ bodyVel1
    #RContact_AnklePos1 = x1[9:12]
    #LContact_AnklePos1 = x1[12:15]


    return bodyRot1, yaw, \
           x1, P1

def main_loop(sm):
    global ROOT_OFFSET
    # BRUCE Setup
    bot = BRUCE()

    # Parameters
    loop_freq               = 500  # run at 500 Hz
    cooling_update_freq     = 1    # update cooling speed at 1 Hz
    loop_duration           = 1. / loop_freq
    cooling_update_duration = 1. / cooling_update_freq

    # IMU Measurements
    gravity_accel = 9.81                           # gravity  acceleration
    accel = np.array([0., 0., gravity_accel])      # filtered accelerometer reading
    omega = np.array([0., 0., 0.])                 # filtered gyroscope     reading
    accel_new = np.array([0., 0., gravity_accel])  # new      accelerometer reading
    omega_new = np.array([0., 0., 0.])             # new      gyroscope     reading

    # Timing Mechanism (to check if IMU is static for orientation estimation)
    IMU_start_time = 0.     # start timer
    IMU_diff_max   = 0.5    # maximum acceptable difference between accelerometer reading and gravity
    IMU_period     = 1.0    # IMU considered static after some period of consistent reading, e.g., 1.0 seconds
    IMU_static     = False  # IMU considered static or not

    # Foot Contacts
    foot_contacts       = np.zeros(4)    # 0/1 indicate in air/contact (for right/left toe/heel)
    foot_contacts_count = np.zeros(4)    # indicate how long the foot is in contact

    # Initial Guess
    raw_bodyPos  = np.array([-hx, 0., 0.38])    # body position         - in world frame
    raw_bodyVel  = np.array([0., 0., 0.])       # body velocity         - in world frame
    raw_bodyAcc  = np.array([0., 0., 0.])       # body acceleration     - in world frame
    raw_bodyRot  = np.eye(3)                    # body orientation      - in world frame
    bodyVel_b  = raw_bodyRot.T @ raw_bodyVel                # body velocity         - in  body frame
    bodyOmg_b  = np.array([0., 0., 0.])       # body angular velocity - in  body frame
    IMUAccBias = np.array([0., 0., 0.])       # accelerometer bias    - in   IMU frame

    R_toePos = np.array([+at, -0.05, 0.00])  # right toe   position  - in world frame
    R_heelPos = np.array([-ah, -0.05, 0.00])  # right heel  position  - in world frame
    R_AnklePos = np.array([0.0, -0.05, 0.02])  # right ankle position  - in world frame
    R_footPos = np.array([0.0, -0.05, 0.00])  # right foot  position  - in world frame

    c_wt_r = np.array([+at, -0.05, 0.00])  # right toe   position if in contact
    c_wh_r = np.array([-ah, -0.05, 0.00])  # right heel  position if in contact
    RContact_AnklePos = np.array([0.0, -0.05, 0.02])  # right ankle position if in contact
    RContact_footPos = np.array([0.0, -0.05, 0.00])  # right foot  position if in contact

    L_toePos = np.array([+at,  0.05, 0.00])  # left  toe   position  - in world frame
    L_heelPos = np.array([-ah,  0.05, 0.00])  # left  heel  position  - in world frame
    L_AnklePos = np.array([0.0,  0.05, 0.02])  # left  ankle position  - in world frame
    L_footPos = np.array([0.0,  0.05, 0.00])  # left  foot  position  - in world frame

    c_wt_l = np.array([+at,  0.05, 0.00])  # left  toe   position if in contact
    c_wh_l = np.array([-ah,  0.05, 0.00])  # left  heel  position if in contact
    LContact_AnklePos = np.array([0.0,  0.05, 0.02])  # left  ankle position if in contact
    LContact_footPos = np.array([0.0,  0.05, 0.00])  # left  foot  position if in contact

    P0     = np.eye(15) * 1e-2  # Kalman filter state covariance matrix

    # Shared Memory Data
    estimation_data = {"bodyRot": np.zeros((3, 3))}

    # Start Estimation
    print("====== The State Estimation Thread is running at", loop_freq, "Hz... ======")

    t0 = bot.get_time()
    last_cooling_update_time = t0
    thread_run = False
    test_Estimation = True
    step_idx = 0
    while True:
        loop_start_time = bot.get_time()
        elapsed_time    = loop_start_time - t0

        if elapsed_time > 1:
            if not thread_run:
                MM.THREAD_STATE.set({"estimation": np.array([1.0])}, opt="only")  # thread is running
                thread_run = True

            # check threading error
            if bot.thread_error():
                bot.stop_threading()

        # get sensor data
        if sm.read_data():
            # IMU
            accel_new = sm.accel
            omega_new = sm.omega
            for idx in range(3):
                # accel[idx] = MF.exp_filter(accel[idx], accel_new[idx], 0.50)
                # omega[idx] = MF.exp_filter(omega[idx], omega_new[idx], 0.50)
                accel[idx] = accel_new[idx]#there is already a filter in the pico
                omega[idx] = omega_new[idx]

            # IMU static check
            IMU_static = False
            if int(bot[PLAN]["robot_state"][0]) == 0:
                if np.abs(MF.norm(accel_new) - gravity_accel) > IMU_diff_max:
                    IMU_start_time = bot.get_time()
                IMU_static = True if bot.get_time() - IMU_start_time > IMU_period else False

        # foot contacts for estimation
        new_contact= False
        if int(bot[PLAN]["robot_state"][0]) == 1:
            if foot_contacts[0] == 0 or foot_contacts[1] == 0:
                new_contact = True
            foot_contacts[0:2] = np.ones(2)
            foot_contacts[2:4] = np.zeros(2)
        elif int(bot[PLAN]["robot_state"][0]) == 2:
            foot_contacts[0:2] = np.zeros(2)
            if foot_contacts[2] == 0 or foot_contacts[3] == 0:
                new_contact = True
            foot_contacts[2:4] = np.ones(2)
        elif int(bot[PLAN]["robot_state"][0]) == 0:
            foot_contacts[0:4] = np.ones(4)
        if new_contact:
            print("New Foot Contact",step_idx)
            ROOT_OFFSET[0] = raw_bodyPos[0]
            ROOT_OFFSET[1] = raw_bodyPos[1]
            step_idx+=1
        for idx in range(4):
            foot_contacts_count[idx] = foot_contacts_count[idx] + 1 if foot_contacts[idx] else 0

        # send cooling speed info to pico
        if loop_start_time - last_cooling_update_time > cooling_update_duration:
            # user_data = MM.USER_COMMAND.get()
            # sm.send_data(cooling_speed=user_data["cooling_speed"][0])
            last_cooling_update_time = loop_start_time
            BEARTemp=bot.getModuleState(MODULE_COOLING)
            if BEARTemp>50:
                sm.send_data(cooling_speed=2)

            if BEARTemp<45:
                sm.send_data(cooling_speed=0)

        # get BEAR info from shared memory
        
            
        # get leg joint states
        q  = bot.State_SMArr[LEG_STATE_POS]#checked
        dq = bot.State_SMArr[LEG_STATE_VEL]#checked
        R_jntPos1, R_jntPos2, R_jntPos3, R_jntPos4, R_jntPos5 = q[0], q[1], q[2], q[3], q[4]
        L_jntPos1, L_jntPos2, L_jntPos3, L_jntPos4, L_jntPos5 = q[5], q[6], q[7], q[8], q[9]
        R_jntVel1, R_jntVel2, R_jntVel3, R_jntVel4, R_jntVel5 = dq[0], dq[1], dq[2], dq[3], dq[4]
        L_jntVel1, L_jntVel2, L_jntVel3, L_jntVel4, L_jntVel5 = dq[5], dq[6], dq[7], dq[8], dq[9]

                # compute leg forward kinematics
        R_ToeBodyPos    , R_ToeBodyVel  , R_Jac_bodyToeVel  , R_dJdt_bodyToeVel, \
        R_HeelBodyPos   , R_HeelBodyVel , R_Jac_bodyHeelVel , R_dJdt_bodyHeelVel, \
        R_AnkleBodyPos  , R_AnkleBodyVel, R_Jac_bodyAnkleVel, R_dJdt_bodyAnkleVel, \
        R_FootBodyPos   , R_FootBodyVel , R_bodyFootRot     , R_Jac_bodyFoot   , R_dJdt_bodyFoot, \
        L_ToeBodyPos    , L_ToeBodyVel  , L_Jac_bodyToeVel  , L_dJdt_bodyToeVel, \
        L_HeelBodyPos   , L_HeelBodyVel , L_Jac_bodyHeelVel , L_dJdt_bodyHeelVel, \
        L_AnkleBodyPos  , L_AnkleBodyVel, L_Jac_bodyAnkleVel, L_dJdt_bodyAnkleVel, \
        L_FootBodyPos   , L_FootBodyVel , L_bodyFootRot     , L_Jac_bodyFoot   , L_dJdt_bodyFoot = kin.legFK(R_jntPos1, R_jntPos2, R_jntPos3, R_jntPos4, R_jntPos5,
                                                                L_jntPos1, L_jntPos2, L_jntPos3, L_jntPos4, L_jntPos5,
                                                                R_jntVel1, R_jntVel2, R_jntVel3, R_jntVel4, R_jntVel5,
                                                                L_jntVel1, L_jntVel2, L_jntVel3, L_jntVel4, L_jntVel5)

        # state estimation
        kR = 0.002 if int(bot[PLAN]["robot_state"][0]) == 0 else 0.000

        ROT_OFFSET = RotMatZ(SN["yaw"].item())

        x0=np.hstack((raw_bodyPos, 
                      raw_bodyVel,
                      IMUAccBias,
                      RContact_AnklePos,
                      LContact_AnklePos))
        
        z1 = np.hstack((R_AnkleBodyPos, L_AnkleBodyPos,
                        R_AnkleBodyVel, L_AnkleBodyVel,
                        R_AnklePos, L_AnklePos))
        
        raw_bodyRot, yaw_angle, \
        x1, P0 = run(raw_bodyRot, omega,
                        raw_bodyAcc, x0,
                        P0,z1,
                        omega, accel, foot_contacts_count[::2],
                        IMU_static, gravity_accel, kR)
        
        bodyOmg_b=omega

        raw_bodyPos = x1[0:3]
        raw_bodyVel = x1[3:6]
        corrected_bodyVel = ROT_OFFSET @ raw_bodyVel
        IMUAccBias = x1[6:9]
        RContact_AnklePos = (x1[9:12])
        LContact_AnklePos = (x1[12:15])
        corrected_bodyRot = ROT_OFFSET @ raw_bodyRot
        yaw_angle = np.arctan2(corrected_bodyRot[1, 0], corrected_bodyRot[0, 0])
        corrected_bodyPos = raw_bodyPos - ROOT_OFFSET
        SN["IMUBodyPos"][:] = corrected_bodyPos
        # SN["IMUBodyVel"] = bodyVel
        # SN["IMUBodyAcc"] = bodyAcc
        SN["IMUBodyRot"][:] = raw_bodyRot
        bot[EST]["bodyPos"] = corrected_bodyPos
        bot[EST]["bodyVel"] = corrected_bodyVel
        #bot[EST]["bodyAcc"] = raw_bodyAcc #no use
        bot[EST]["bodyRot"] = corrected_bodyRot

        raw_bodyAcc  = raw_bodyRot @ (accel - IMUAccBias) - np.array([0., 0., gravity_accel])  # body acceleration excluding gravity
        bodyVel_b = raw_bodyRot.T @ raw_bodyVel            

        L_Vel=np.array([L_jntVel1,L_jntVel2,L_jntVel3,L_jntVel4,L_jntVel5])
        R_Vel=np.array([R_jntVel1,R_jntVel2,R_jntVel3,R_jntVel4,R_jntVel5])
        # compute robot forward kinematics
        R_toePos    , R_toeVel  , R_Jac_wToeVel  , R_dJdq_toeVel, \
        R_heelPos   , R_heelVel , R_Jac_wHeelVel , R_dJdq_heelVel, \
        R_AnklePos  , R_AnkleVel, R_Jac_wAnkleVel, R_dJdq_AnkleVel, \
        R_footPos   , R_footVel ,  \
        R_footRot   , R_footOmg, R_Jac_footOmg, dJwdq_ff_r, \
        L_toePos    , L_toeVel  , L_Jac_wToeVel  , L_dJdq_toeVel, \
        L_heelPos   , L_heelVel , L_Jac_wHeelVel , L_dJdq_heelVel, \
        L_AnklePos  , L_AnkleVel, L_Jac_wAnkleVel, L_dJdq_AnkleVel, \
        L_footPos   , L_footVel ,  \
        L_footRot   , L_footOmg, L_Jac_footOmg, dJwdq_ff_l =kin.robotFK(corrected_bodyRot     , corrected_bodyPos             , bodyOmg_b         , bodyVel_b,
                                                                                R_ToeBodyPos  , R_Jac_bodyToeVel    , R_dJdt_bodyToeVel ,
                                                                                R_HeelBodyPos , R_Jac_bodyHeelVel   , R_dJdt_bodyHeelVel,
                                                                                R_AnkleBodyPos, R_Jac_bodyAnkleVel  , R_dJdt_bodyAnkleVel, R_bodyFootRot, R_Jac_bodyFoot, R_dJdt_bodyFoot,
                                                                                L_ToeBodyPos  , L_Jac_bodyToeVel    , L_dJdt_bodyToeVel,
                                                                                L_HeelBodyPos , L_Jac_bodyHeelVel   , L_dJdt_bodyHeelVel,
                                                                                L_AnkleBodyPos, L_Jac_bodyAnkleVel  , L_dJdt_bodyAnkleVel, L_bodyFootRot, L_Jac_bodyFoot, L_dJdt_bodyFoot,
                                                                                R_Vel         , L_Vel)


        # calculate robot dynamics
        H, CG, AG, dAGdq, comPos, comVel, angMom = dyn.robotID(corrected_bodyRot, corrected_bodyPos, bodyOmg_b, bodyVel_b,
                                                         R_jntPos1, R_jntPos2, R_jntPos3, R_jntPos4, R_jntPos5,
                                                         L_jntPos1, L_jntPos2, L_jntPos3, L_jntPos4, L_jntPos5,
                                                         R_jntVel1, R_jntVel2, R_jntVel3, R_jntVel4, R_jntVel5,
                                                         L_jntVel1, L_jntVel2, L_jntVel3, L_jntVel4, L_jntVel5)

        # save data
        # Set each value using bot.setEst(key, value)
        bot.setEst("time_stamp", np.array([elapsed_time]))

        bot[EST]["bodyOmg"] = bodyOmg_b
        bot.setEst("body_yaw_ang", np.array([yaw_angle]))
        bot[EST]["COMPos"] = comPos
        bot[EST]["COMVel"] = comVel
        bot[EST]["angMomemtum"] = angMom
        bot[EST]["H_matrix"] = H
        bot[EST]["CG_vector"] = CG
        bot[EST]["AG_matrix"] = AG
        bot[EST]["dAGdq_vector"] = dAGdq
        bot[EST]["foot_contacts"] = sm.foot_contact

        bot[EST]["R_footRot"] = R_footRot
        bot[EST]["R_footOmg"] = R_footOmg
        bot[EST]["R_foot_Jw"] = R_Jac_footOmg
        bot[EST]["R_foot_dJwdq"] = dJwdq_ff_r
        bot[EST]["R_footPos"] = R_footPos
        bot[EST]["R_footVel"] = R_footVel
        bot[EST]["R_toePos"] = R_toePos
        bot[EST]["R_toeVel"] = R_toeVel
        bot[EST]["R_Jac_wToeVel"] = R_Jac_wToeVel
        bot[EST]["R_dJ_toeVel*dq"] = R_dJdq_toeVel
        bot[EST]["R_heelPos"] = R_heelPos
        bot[EST]["R_heelVel"] = R_heelVel
        bot[EST]["R_Jac_wHeelVel"] = R_Jac_wHeelVel
        bot[EST]["R_dJ_heelVel*dq"] = R_dJdq_heelVel
        bot[EST]["R_anklePos"] = R_AnklePos
        bot[EST]["R_ankleVel"] = R_AnkleVel
        bot[EST]["R_Jac_wAnkleVel"] = R_Jac_wAnkleVel
        bot[EST]["R_dJ_AnkleVel*dq"] = R_dJdq_AnkleVel

        bot[EST]["L_footRot"] = L_footRot
        bot[EST]["L_footOmg"] = L_footOmg
        bot[EST]["L_foot_Jw"] = L_Jac_footOmg
        bot[EST]["L_foot_dJwdq"] = dJwdq_ff_l
        bot[EST]["L_footPos"] = L_footPos
        bot[EST]["L_footVel"] = L_footVel
        bot[EST]["L_toePos"] = L_toePos
        bot[EST]["L_toeVel"] = L_toeVel
        bot[EST]["L_Jac_wToeVel"] = L_Jac_wToeVel
        bot[EST]["L_dJ_toeVel*dq"] = L_dJdq_toeVel
        bot[EST]["L_heelPos"] = L_heelPos
        bot[EST]["L_heelVel"] = L_heelVel
        bot[EST]["L_Jac_wHeelVel"] = L_Jac_wHeelVel
        bot[EST]["L_dJ_heelVel*dq"] = L_dJdq_heelVel
        bot[EST]["L_anklePos"] = L_AnklePos
        bot[EST]["L_ankleVel"] = L_AnkleVel
        bot[EST]["L_Jac_wAnkleVel"] = L_Jac_wAnkleVel
        bot[EST]["L_dJ_AnkleVel*dq"] = L_dJdq_AnkleVel

        #print(bodyPos, bodyVel, bodyAcc, bodyRot, bodyOmg_b, yaw_angle)

        # check time to ensure that the state estimator stays at a consistent running loop.
        loop_end_time = loop_start_time + loop_duration
        present_time  = bot.get_time()
        if present_time > loop_end_time:
            delay_time = 1000 * (present_time - loop_end_time)
            if delay_time > 1.:
                print(colored("Delayed " + str(delay_time)[0:5] + " ms at Te = " + str(elapsed_time)[0:5] + " s", "yellow"))
        else:
            while bot.get_time() < loop_end_time:
                pass


def estMain():
    # PICO Setup
    sm = sense_manager.SenseManager(port=PICO_port, baudrate=PICO_baudrate)
    sm.send_data(pico_mode="nominal")  # run PICO

    # IMU_check(sm)
    # contact_check(sm)
    main_loop(sm)
    # try:
    #     main_loop(sm)
    # except (NameError, KeyboardInterrupt) as error:
    #     MM.THREAD_STATE.set({"estimation": np.array([0.0])}, opt="only")  # thread is stopped
    # except Exception as error:
    #     print(error)
    #     MM.THREAD_STATE.set({"estimation": np.array([2.0])}, opt="only")  # thread in error
    # finally:
    #     sm.send_data(pico_mode="idle")  # set PICO to idle mode
