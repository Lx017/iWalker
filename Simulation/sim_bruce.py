#!usr/bin/env python
__author__    = "Westwood Robotics Corporation"
__email__     = "info@westwoodrobotics.io"
__copyright__ = "Copyright 2023 Westwood Robotics Corporation"
__date__      = "November 1, 2023"
__project__   = "BRUCE"
__version__   = "0.0.4"
__status__    = "Product"

"""
Script for communication with Gazebo
"""

import time
from Settings.Bruce import *
import Startups.memory_manager as MM
import Library.ROBOT_MODEL.BRUCE_kinematics as kin
import Library.ROBOT_MODEL.BRUCE_dynamics as dyn
from Settings.BRUCE_macros import *
from Library.BRUCE_GYM.GAZEBO_INTERFACE import Manager as gazint
from multiprocessing import shared_memory

bot=BRUCE()

bot.setModuleState(MODULE_IFSIM,1)

def robotFK(R, p, w, bv,
            prt, Jrt, dJrt,
            prh, Jrh, dJrh,
            pra, Jra, dJra, Rr, Jwr, dJwr,
            plt, Jlt, dJlt,
            plh, Jlh, dJlh,
            pla, Jla, dJla, Rl, Jwl, dJwl,
            dqr, dql):
    R = np.copy(R)

    # foot position in world frame
    prt = np.copy(prt)
    prh = np.copy(prh)
    pra = np.copy(pra)
    plt = np.copy(plt)
    plh = np.copy(plh)
    pla = np.copy(pla)

    crt = p + R @ prt
    crh = p + R @ prh
    cra = p + R @ pra
    clt = p + R @ plt
    clh = p + R @ plh
    cla = p + R @ pla

    crm = (crt + crh) / 2.
    clm = (clt + clh) / 2.

    # foot orientation in world frame
    Rr = np.copy(Rr)
    Rl = np.copy(Rl)

    wRr = R @ Rr
    wRl = R @ Rl

    # robot Jacobians in world frame
    wJrt = np.zeros((3, 16))
    wJrh = np.zeros((3, 16))
    wJra = np.zeros((3, 16))
    wJlt = np.zeros((3, 16))
    wJlh = np.zeros((3, 16))
    wJla = np.zeros((3, 16))

    Jrt = np.copy(Jrt)
    Jrh = np.copy(Jrh)
    Jra = np.copy(Jra)
    Jlt = np.copy(Jlt)
    Jlh = np.copy(Jlh)
    Jla = np.copy(Jla)

    dqAll = np.hstack((w, bv, dqr, dql))
    
    wJrt[ 0:3, 0:3] = -R @ MF.hat(prt)
    wJrt[ 0:3, 3:6] =  R
    wJrt[0:3, 6:11] =  R @ Jrt

    dcrt = wJrt @ dqAll

    wJrh[ 0:3, 0:3] = -R @ MF.hat(prh)
    wJrh[ 0:3, 3:6] =  R
    wJrh[0:3, 6:11] =  R @ Jrh

    wJra[0:3, 0:3]  = -R @ MF.hat(pra)
    wJra[0:3, 3:6]  =  R
    wJra[0:3, 6:11] =  R @ Jra

    wJlt[0:3,   0:3] = -R @ MF.hat(plt)
    wJlt[0:3,   3:6] =  R
    wJlt[0:3, 11:16] =  R @ Jlt

    wJlh[0:3,   0:3] = -R @ MF.hat(plh)
    wJlh[0:3,   3:6] =  R
    wJlh[0:3, 11:16] =  R @ Jlh

    wJla[0:3,   0:3] = -R @ MF.hat(pla)
    wJla[0:3,   3:6] =  R
    wJla[0:3, 11:16] =  R @ Jla

    # robot rotational Jacobians in foot frame
    wJwr = np.zeros((3, 16))
    wJwl = np.zeros((3, 16))

    Jwr = np.copy(Jwr)
    Jwl = np.copy(Jwl)

    RrT = Rr.T
    wJwr[0:3,  0:3] = RrT
    wJwr[0:3, 6:11] = RrT @ Jwr

    RlT = Rl.T
    wJwl[0:3,   0:3] = RlT
    wJwl[0:3, 11:16] = RlT @ Jwl


    # foot velocity in world frame
    dcrh = wJrh @ dqAll
    dcra = wJra @ dqAll
    dclt = wJlt @ dqAll
    dclh = wJlh @ dqAll
    dcla = wJla @ dqAll

    dcrm = (dcrt + dcrh) / 2.
    dclm = (dclt + dclh) / 2.

    # foot angular rate in foot frame
    wr = wJwr @ dqAll
    wl = wJwl @ dqAll

    # dJdq
    what = MF.hat(w)
    bv   = np.copy(bv)

    whatbv   = what @ bv
    what2    = 2 * what

    wdJrtdq = R @ (whatbv + (what2 @ Jrt + dJrt) @ dqr)
    wdJrhdq = R @ (whatbv + (what2 @ Jrh + dJrh) @ dqr)
    wdJradq = R @ (whatbv + (what2 @ Jra + dJra) @ dqr)

    wdJltdq = R @ (whatbv + (what2 @ Jlt + dJlt) @ dql)
    wdJlhdq = R @ (whatbv + (what2 @ Jlh + dJlh) @ dql)
    wdJladq = R @ (whatbv + (what2 @ Jla + dJla) @ dql)

    wdJwrdq = RrT @ (what @ Jwr + dJwr) @ dqr
    wdJwldq = RlT @ (what @ Jwl + dJwl) @ dql

    return crt, dcrt, wJrt, wdJrtdq, \
           crh, dcrh, wJrh, wdJrhdq, \
           cra, dcra, wJra, wdJradq, \
           crm, dcrm, wRr, wr, wJwr, wdJwrdq, \
           clt, dclt, wJlt, wdJltdq, \
           clh, dclh, wJlh, wdJlhdq, \
           cla, dcla, wJla, wdJladq, \
           clm, dclm, wRl, wl, wJwl, wdJwldq

class GazeboSimulator:
    def __init__(self):
        # robot info
        self.num_legs = 2
        self.num_joints_per_leg = 5
        self.num_arms = 2
        self.num_joints_per_arms = 3
        self.num_joints = self.num_legs * self.num_joints_per_leg + self.num_arms * self.num_joints_per_arms
        self.num_contact_sensors = 4
        
        self.leg_p_gains = [265, 150,  80,  80,    30]
        self.leg_i_gains = [  0,   0,   0,   0,     0]
        self.leg_d_gains = [ 1., 2.3, 0.8, 0.8, 0.003]

        self.arm_p_gains = [ 1.6,  1.6,  1.6]
        self.arm_i_gains = [   0,    0,    0]
        self.arm_d_gains = [0.03, 0.03, 0.03]

        self.p_gains = self.leg_p_gains * 2 + self.arm_p_gains * 2  # the joint order matches the robot"s sdf file
        self.i_gains = self.leg_i_gains * 2 + self.arm_i_gains * 2
        self.d_gains = self.leg_d_gains * 2 + self.arm_d_gains * 2
        
        # simulator info
        self.simulator = None
        self.simulation_frequency = 1000  # Hz
        self.simulation_modes = {"torque": 0, "position": 2}
        self.simulation_mode = self.simulation_modes["position"]
        
    def initialize_simulator(self):
        self.simulator = gazint.GazeboInterface(robot_name="bruce", num_joints=self.num_joints, num_contact_sensors=self.num_contact_sensors)
        self.simulator.set_step_size(1. / self.simulation_frequency)
        self.simulator.set_operating_mode(self.simulation_mode)
        self.simulator.set_all_position_pid_gains(self.p_gains, self.i_gains, self.d_gains)

        # arm pose
        ar1, ar2, ar3 = -0.7,  1.3,  2.0
        al1, al2, al3 =  0.7, -1.3, -2.0

        # leg pose
        bpr = np.array([0.04, -0.07, -0.42])  # right foot position  in body frame
        bpl = np.array([0.04, +0.07, -0.42])  # left  foot position  in body frame
        bxr = np.array([1., 0., 0.])          # right foot direction in body frame
        bxl = np.array([1., 0., 0.])          # left  foot direction in body frame
        lr1, lr2, lr3, lr4, lr5 = kin.legIK_foot(bpr, bxr, +1.)
        ll1, ll2, ll3, ll4, ll5 = kin.legIK_foot(bpl, bxl, -1.)
        initial_pose = [lr1+PI_2, lr2-PI_2, lr3, lr4, lr5,
                        ll1+PI_2, ll2-PI_2, ll3, ll4, ll5,
                        ar1, ar2, ar3,
                        al1, al2, al3]
        self.simulator.reset_simulation(initial_pose=initial_pose)
        
        print("Gazebo Initialization Completed!")
        
    def write_position(self, leg_positions, arm_positions):
        """
        Send goal positions to the simulator.
        """
        goal_position = [leg_positions[0]+PI_2, leg_positions[1]-PI_2, leg_positions[2], leg_positions[3], leg_positions[4],
                         leg_positions[5]+PI_2, leg_positions[6]-PI_2, leg_positions[7], leg_positions[8], leg_positions[9],
                         arm_positions[0], arm_positions[1], arm_positions[2],
                         arm_positions[3], arm_positions[4], arm_positions[5]]
        if self.simulation_mode != self.simulation_modes["position"]:
            self.simulation_mode = self.simulation_modes["position"]
            self.simulator.set_operating_mode(self.simulation_mode)
        self.simulator.set_command_position(goal_position)

    def write_torque(self, leg_torques, arm_torques):
        """
        Send goal torques to the simulator.
        """
        goal_torque = [leg_torques[0], leg_torques[1], leg_torques[2], leg_torques[3], leg_torques[4],
                       leg_torques[5], leg_torques[6], leg_torques[7], leg_torques[8], leg_torques[9],
                       arm_torques[0], arm_torques[1], arm_torques[2],
                       arm_torques[3], arm_torques[4], arm_torques[5]]
        if self.simulation_mode != self.simulation_modes["torque"]:
            self.simulation_mode = self.simulation_modes["torque"]
            self.simulator.set_operating_mode(self.simulation_mode)
        self.simulator.set_command_torque(goal_torque)

    def get_arm_goal_torques(self, arm_positions, arm_velocities):
        """
        Calculate arm goal torques.
        """
        arm_goal_torque = np.zeros(6)
        for i in range(6):
            arm_goal_torque[i] = self.arm_p_gains[i % self.num_joints_per_arms] * (arm_positions[i] - self.q_arm[i]) + self.arm_d_gains[i % self.num_joints_per_arms] * (arm_velocities[i] - self.dq_arm[i])
        return arm_goal_torque
        
    def update_sensor_info(self,bot:BRUCE):
        """
        Get sensor info and write it to shared memory.
        """
        # get joint states
        q = self.simulator.get_current_position()
        dq = self.simulator.get_current_velocity()
        
        self.q_leg  = np.array([q[0]-PI_2, q[1]+PI_2, q[2], q[3], q[4],
                                q[5]-PI_2, q[6]+PI_2, q[7], q[8], q[9]])
        self.q_arm  = q[10:16]
        self.dq_leg = dq[0:10]
        self.dq_arm = dq[10:16]

        arm_data = {"joint_positions":  self.q_arm,
                    "joint_velocities": self.dq_arm}
        
        
        # get imu states
        self.rot_mat = self.simulator.get_body_rot_mat()
        self.accel   = self.simulator.get_imu_acceleration()
        self.omega   = self.rot_mat.T @ self.simulator.get_imu_angular_rate()
        self.foot_contacts = self.simulator.get_foot_contacts()
        
        sense_data = {"imu_acceleration": self.accel,
                      "imu_ang_rate":     self.omega,
                      "foot_contacts":    self.foot_contacts}
        sensorArray = np.concatenate((self.accel, self.omega, self.foot_contacts))

        bot.setState(LEG_STATE_POS,self.q_leg)
        bot.setState(LEG_STATE_VEL,self.dq_leg)
        bot.setState(ARM_STATE_POS,self.q_arm)
        bot.setState(ARM_STATE_VEL,self.dq_arm)
        bot.setState(SENSOR_STATE,sensorArray)
    
    def calculate_robot_model(self,bot:BRUCE):
        """
        Calculate kinematics & dynamics and write it to shared memory.
        """
        R_jntPos1, R_jntPos2, R_jntPos3, R_jntPos4, R_jntPos5 = self.q_leg[0], self.q_leg[1], self.q_leg[2], self.q_leg[3], self.q_leg[4]
        R_jntVel1, R_jntVel2, R_jntVel3, R_jntVel4, R_jntVel5 = self.dq_leg[0], self.dq_leg[1], self.dq_leg[2], self.dq_leg[3], self.dq_leg[4]
        L_jntPos1, L_jntPos2, L_jntPos3, L_jntPos4, L_jntPos5 = self.q_leg[5], self.q_leg[6], self.q_leg[7], self.q_leg[8], self.q_leg[9]
        L_jntVel1, L_jntVel2, L_jntVel3, L_jntVel4, L_jntVel5 = self.dq_leg[5], self.dq_leg[6], self.dq_leg[7], self.dq_leg[8], self.dq_leg[9]

        bodyRot = self.rot_mat
        bodyOmg_b = self.omega
        bodyPos = self.simulator.get_bodyPos()
        bodyVel = self.simulator.get_bodyVel()
        bodyAcc = bodyRot @ self.accel
        bodyVel_b = bodyRot.T @ bodyVel
        yaw_angle = np.arctan2(bodyRot[1, 0], bodyRot[0, 0])

        # compute leg forward kinematics
        
        R_ToeBodyPos    , R_ToeBodyVel  , R_Jac_bodyToeVel  , R_dJdt_bodyToeVel, \
        R_HeelBodyPos   , R_HeelBodyVel , R_Jac_bodyHeelVel , R_dJdt_bodyHeelVel, \
        R_AnkleBodyPos  , R_AnkleBodyVel, R_Jac_bodyAnkleVel, R_dJdt_bodyAnkleVel, \
        R_FootBodyPos   , R_FootBodyVel , R_bodyFootRot     , R_Jac_bodyFoot   , R_dJdt_bodyFoot, \
        L_ToeBodyPos    , L_ToeBodyVel  , L_Jac_bodyToeVel  , L_dJdt_bodyToeVel, \
        L_HeelBodyPos   , L_HeelBodyVel , L_Jac_bodyHeelVel , L_dJdt_bodyHeelVel, \
        L_AnkleBodyPos  , L_AnkleBodyVel, L_Jac_bodyAnkleVel, L_dJdt_bodyAnkleVel, \
        L_FootBodyPos   , L_FootBodyVel , L_bodyFootRot     , L_Jac_bodyFoot   , L_dJdt_bodyFoot = kin.legFK(R_jntPos1 , R_jntPos2    , R_jntPos3    , R_jntPos4    , R_jntPos5,
                                                                                                            L_jntPos1  , L_jntPos2    , L_jntPos3    , L_jntPos4    , L_jntPos5,
                                                                                                            R_jntVel1 , R_jntVel2   , R_jntVel3   , R_jntVel4   , R_jntVel5,
                                                                                                            L_jntVel1 , L_jntVel2   , L_jntVel3   , L_jntVel4   , L_jntVel5)

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
        L_footRot   , L_footOmg, L_Jac_footOmg, dJwdq_ff_l =robotFK(bodyRot       , bodyPos             , bodyOmg_b         , bodyVel_b,
                                                                                R_ToeBodyPos  , R_Jac_bodyToeVel    , R_dJdt_bodyToeVel ,
                                                                                R_HeelBodyPos , R_Jac_bodyHeelVel   , R_dJdt_bodyHeelVel,
                                                                                R_AnkleBodyPos, R_Jac_bodyAnkleVel  , R_dJdt_bodyAnkleVel, R_bodyFootRot, R_Jac_bodyFoot, R_dJdt_bodyFoot,
                                                                                L_ToeBodyPos  , L_Jac_bodyToeVel    , L_dJdt_bodyToeVel,
                                                                                L_HeelBodyPos , L_Jac_bodyHeelVel   , L_dJdt_bodyHeelVel,
                                                                                L_AnkleBodyPos, L_Jac_bodyAnkleVel  , L_dJdt_bodyAnkleVel, L_bodyFootRot, L_Jac_bodyFoot, L_dJdt_bodyFoot,
                                                                                R_Vel         , L_Vel)

        # calculate robot dynamics
        H, CG, AG, dAGdq, comPos, comVel, angMom = dyn.robotID(bodyRot, bodyPos, bodyOmg_b, bodyVel_b,
                                                         R_jntPos1, R_jntPos2, R_jntPos3, R_jntPos4, R_jntPos5,
                                                         L_jntPos1, L_jntPos2, L_jntPos3, L_jntPos4, L_jntPos5,
                                                         R_jntVel1, R_jntVel2, R_jntVel3, R_jntVel4, R_jntVel5,
                                                         L_jntVel1, L_jntVel2, L_jntVel3, L_jntVel4, L_jntVel5)

        # save as estimation data
        bot[EST]["time_stamp"] = np.array([self.simulator.get_current_time()])
        bot[EST]["bodyPos"] = bodyPos
        bot[EST]["bodyVel"] = bodyVel
        bot[EST]["bodyAcc"] = bodyAcc
        bot[EST]["bodyRot"] = bodyRot
        bot[EST]["bodyOmg"] = bodyOmg_b
        bot[EST]["body_yaw_ang"] = np.array([yaw_angle])
        bot[EST]["COMPos"] = comPos
        bot[EST]["COMVel"] = comVel
        bot[EST]["angMomemtum"] = angMom
        bot[EST]["H_matrix"] = H
        bot[EST]["CG_vector"] = CG
        bot[EST]["AG_matrix"] = AG
        bot[EST]["dAGdq_vector"] = dAGdq
        bot[EST]["foot_contacts"] = self.foot_contacts
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



def main_loop():
    # When restart this thread, reset the shared memory (so the robot is in idle)
    bot = BRUCE()
    legposSMArr=bot.All_PosCmd_SMArr
    legtorqSMArr=bot.Leg_Torq_SMArr
    threadSMArr=bot.Thread_SMArr
    MM.init()
    MM.connect()
    threadSMArr[:]=0

    # BRUCE SETUP

    gs = GazeboSimulator()
    gs.initialize_simulator()

    MM.THREAD_STATE.set({"simulation": np.array([1.0])}, opt="only")  # thread is running

    while True:
        if bot.thread_error():
            bot.stop_threading()

        gs.update_sensor_info(bot)
        gs.calculate_robot_model(bot)


        ar1, ar2, ar3 = -0.7,  1.3,  2.0
        al1, al2, al3 =  0.7, -1.3, -2.0
        bpr = np.array([0.04, -0.07, -0.42])  # right foot position  in body frame
        bpl = np.array([0.04, +0.07, -0.42])  # left  foot position  in body frame
        bxr = np.array([1., 0., 0.])          # right foot direction in body frame
        bxl = np.array([1., 0., 0.])          # left  foot direction in body frame
        lr1, lr2, lr3, lr4, lr5 = kin.legIK_foot(bpr, bxr, +1.)
        ll1, ll2, ll3, ll4, ll5 = kin.legIK_foot(bpl, bxl, -1.)
        armTarPos = [ar1, ar2, ar3, al1, al2, al3]
        legTarPos = [lr1+PI_2, lr2-PI_2, lr3, lr4, lr5,
                        ll1+PI_2, ll2-PI_2, ll3, ll4, ll5]

        
        if legposSMArr[1] == 1.0:
            gs.write_position(legposSMArr[2:12], legposSMArr[14:20])
        elif legtorqSMArr[2]!=0:#if there is any data in the torque array
            if threadSMArr[0]==1:
                #gs.write_position(legTarPos, armTarPos)
                gs.write_torque(legtorqSMArr[2:12], gs.get_arm_goal_torques(armTarPos, np.zeros(6)))
        
        gs.simulator.step_simulation()
        time.sleep(0.000)  # delay if needed


def simMain():
    main_loop()