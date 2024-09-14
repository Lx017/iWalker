#!usr/bin/env python
__author__ = "Westwood Robotics Corporation"
__email__ = "info@westwoodrobotics.io"
__copyright__ = "Copyright 2023 Westwood Robotics Corporation"
__date__ = "November 1, 2023"
__project__ = "BRUCE"
__version__ = "0.0.4"
__status__ = "Product"

"""
Script that holds useful robot info
"""

import time
import collections
import Startups.memory_manager as MM
from termcolor import colored
from Settings.BRUCE_macros import *
from Play.Walking.walking_macros import *
from multiprocessing import shared_memory as sm

# states
LEG_STATE_POS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
LEG_STATE_VEL = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
ARM_STATE_POS = np.array([20, 21, 22, 23, 24, 25])
ARM_STATE_VEL = np.array([26, 27, 28, 29, 30, 31])
#SENSOR_STATE = np.array([32, 33, 34, 35, 36, 37, 38, 39, 40, 41])
LEG_STATE_TOR = np.array([42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
# states

LEG_TAR_POS = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
LEG_TAR_VEL = np.array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
LEG_TAR_TOR = np.array([22, 23, 24, 25, 26, 27, 28, 29, 30, 31])

MODULE_IFSIM = 0
MODULE_LOW = 1
MODULE_HIGH = 2
MODULE_TOP = 3
MODULE_EST = 4
MODULE_BEAR = 5
MODULE_COOLING = 6


def getSMArray(name, size):
    try:
        SM = sm.SharedMemory(name=name)
    except:
        print("create New")
        SM = sm.SharedMemory(name=name, create=True, size=size * 8)
    return np.ndarray(shape=(size,), dtype=np.float64, buffer=SM.buf), SM



BODY_DICT = {
    "time_stamp": [0, 1],
    "bodyPos": [1, 3],
    "bodyVel": [4, 3],
    "bodyAcc": [7, 3],
    "bodyRot": [10, 9],
    "body_euler_ang": [19, 3],
    "body_yaw_ang": [22, 1],
    "bodyOmg": [23, 3],
    "COMPos": [26, 3],
    "COMVel": [29, 3],
    "angMomemtum": [32, 3],
    "H_matrix": [35, 16 * 16],
    "CG_vector": [291, 16],
    "AG_matrix": [307, 6 * 16],
    "dAGdq_vector": [403, 6],
    "foot_contacts": [409, 4],
}

EST_DICT = {
    "time_stamp": [0, 1, (1, 1)],
    "bodyPos": [1, 3, (3,)],
    "bodyVel": [4, 3, (3,)],
    "bodyAcc": [7, 3, (3,)],
    "bodyRot": [10, 9, (3, 3)],
    "bodyOmg": [19, 3, (3,)],
    "body_yaw_ang": [22, 1, (1,)],
    "COMPos": [23, 3, (3,)],
    "COMVel": [26, 3, (3,)],
    "angMomemtum": [29, 3, (3,)],
    "H_matrix": [32, 256, (16, 16)],
    "CG_vector": [288, 16, (16,)],
    "AG_matrix": [304, 96, (6, 16)],
    "dAGdq_vector": [400, 6, (6,)],
    "foot_contacts": [406, 4, (4,)],
    "R_footRot": [410, 9, (3, 3)],
    "R_footOmg": [419, 3, (3,)],
    "R_foot_Jw": [422, 48, (3, 16)],
    "R_foot_dJwdq": [470, 3, (3,)],
    "R_footPos": [473, 3, (3,)],
    "R_footVel": [476, 3, (3,)],
    "R_toePos": [479, 3, (3,)],
    "R_toeVel": [482, 3, (3,)],
    "R_Jac_wToeVel": [485, 48, (3, 16)],
    "R_dJ_toeVel*dq": [533, 3, (3,)],
    "R_heelPos": [536, 3, (3,)],
    "R_heelVel": [539, 3, (3,)],
    "R_Jac_wHeelVel": [542, 48, (3, 16)],
    "R_dJ_heelVel*dq": [590, 3, (3,)],
    "R_anklePos": [593, 3, (3,)],
    "R_ankleVel": [596, 3, (3,)],
    "R_Jac_wAnkleVel": [599, 48, (3, 16)],
    "R_dJ_AnkleVel*dq": [647, 3, (3,)],
    "L_footRot": [650, 9, (3, 3)],
    "L_footOmg": [659, 3, (3,)],
    "L_foot_Jw": [662, 48, (3, 16)],
    "L_foot_dJwdq": [710, 3, (3,)],
    "L_footPos": [713, 3, (3,)],
    "L_footVel": [716, 3, (3,)],
    "L_toePos": [719, 3, (3,)],
    "L_toeVel": [722, 3, (3,)],
    "L_Jac_wToeVel": [725, 48, (3, 16)],
    "L_dJ_toeVel*dq": [773, 3, (3,)],
    "L_heelPos": [776, 3, (3,)],
    "L_heelVel": [779, 3, (3,)],
    "L_Jac_wHeelVel": [782, 48, (3, 16)],
    "L_dJ_heelVel*dq": [830, 3, (3,)],
    "L_anklePos": [833, 3, (3,)],
    "L_ankleVel": [836, 3, (3,)],
    "L_Jac_wAnkleVel": [839, 48, (3, 16)],
    "L_dJ_AnkleVel*dq": [887, 3, (3,)],
}

PLAN_DICT = {
    "time_stamp": [0, 1, (1,)],
    "robot_state": [1, 1, (1,)],
    "bodyPos": [2, 3, (3,)],
    "bodyVel": [5, 3, (3,)],
    "bodyRot": [8, 9, (3, 3)],
    "bodyOmg": [17, 3, (3,)],
    "COMPos": [20, 3, (3,)],
    "COMVel": [23, 3, (3,)],
    "R_footPos": [26, 3, (3,)],
    "R_footVel": [29, 3, (3,)],
    "R_footRot": [32, 9, (3, 3)],
    "R_footOmg": [41, 3, (3,)],
    "L_footPos": [44, 3, (3,)],
    "L_footVel": [47, 3, (3,)],
    "L_footRot": [50, 9, (3, 3)],
    "L_footOmg": [59, 3, (3,)],
}

EST = "est"
PLAN = "plan"

class dictGetter:
    def __init__(self, dict,arr):
        self.dict = dict
        self.arr = arr

    def __getitem__(self, key):
        len = self.dict[key][1]
        shape = self.dict[key][2]
        startIdx = self.dict[key][0]
        return self.arr[startIdx : startIdx + len].reshape(shape)
    
    def __setitem__(self, key, value):
        if key in self.dict:
            len = self.dict[key][1]
            startIdx = self.dict[key][0]
            self.arr[startIdx : startIdx + len] = value.flatten()

class BRUCE:
    All_PosCmd_SMArr, avoidLeak1 = getSMArray("pos", 50)
    Leg_Torq_SMArr, avoidLeak2 = getSMArray("legTorq", 50)
    Body_SMArr, avoidLeak3 = getSMArray("body", 413)
    Thread_SMArr, avoidLeak4 = getSMArray("thread", 5)
    State_SMArr, avoidLeak5 = getSMArray("state", 70)
    Est_SMArr, avoidLeak6 = getSMArray("est", 900)
    Value_SMArr, avoidLeak7 = getSMArray("value", 50)
    Plan_SMArr, avoidLeak8 = getSMArray("plan", 70)
    module_states, _ = getSMArray("module_states", 10)

    estDictgetter = dictGetter(EST_DICT, Est_SMArr)
    planDictGetter = dictGetter(PLAN_DICT, Plan_SMArr)

    def __getitem__(self, key):
        if key == EST:
            return BRUCE.estDictgetter
        elif key == PLAN:
            return BRUCE.planDictGetter


    def getBody(self, key):
        len = BODY_DICT[key][1]
        startIdx = BODY_DICT[key][0]
        return BRUCE.Body_SMArr[startIdx : startIdx + len]

    def setBody(self, key, value):
        if key in BODY_DICT:
            len = BODY_DICT[key][1]
            startIdx = BODY_DICT[key][0]
            BRUCE.Body_SMArr[startIdx : startIdx + len] = value

    def getEst(self, key):
        shape = EST_DICT[key][2]
        len = EST_DICT[key][1]
        startIdx = EST_DICT[key][0]
        return BRUCE.Est_SMArr[startIdx : startIdx + len].reshape(shape)

    def setEst(self, key, value):
        if key in EST_DICT:
            len = EST_DICT[key][1]
            startIdx = EST_DICT[key][0]
            BRUCE.Est_SMArr[startIdx : startIdx + len] = value.flatten()

    def setPlan(self, key, value):
        if key in PLAN_DICT:
            len = PLAN_DICT[key][1]
            startIdx = PLAN_DICT[key][0]
            BRUCE.Plan_SMArr[startIdx : startIdx + len] = value.flatten()

    def getPln(self, key):
        shape = PLAN_DICT[key][2]
        len = PLAN_DICT[key][1]
        startIdx = PLAN_DICT[key][0]
        return BRUCE.Plan_SMArr[startIdx : startIdx + len].reshape(shape)

    def setState(self, idxs, value):
        self.State_SMArr[idxs] = value

    def getState(self, idxs):
        return self.State_SMArr[idxs]
    


    def __init__(self):
        # PyBEAR settings
        self.BEAR_modes = {"torque": 0, "velocity": 1, "position": 2, "force": 3}

        # DXL settings
        self.DXL_modes = {
            "position": 3,
            "velocity": 1,
            "extended position": 4,
            "PWM": 16,
        }

        # Joint info
        self.joint = collections.defaultdict(lambda: collections.defaultdict())

        # Gamepad info
        self.gamepad = collections.defaultdict()


    def R_yaw(self):
        return MF.Rz(self.__getitem__(EST)["body_yaw_ang"][0])



    def update_input_status(self):
        input_data = MM.USER_COMMAND.get()

        self.walk = input_data["walk"][0]
        self.cmd_vxy_wg = input_data["com_xy_velocity"]
        self.cmd_yaw_rate = input_data["yaw_rate"][0]
        self.cmd_comPos_change = input_data["COMPos_change_scaled"]
        self.cmd_euler_change = input_data["body_euler_angle_change"]
        self.cmd_R_change = (
            MF.Rx(self.cmd_euler_change[0])
            @ MF.Ry(self.cmd_euler_change[1])
            @ MF.Rz(self.cmd_euler_change[2])
        )
        self.cmd_yaw_r_change = input_data["R_foot_yaw_angle_change"][0]
        self.cmd_yaw_l_change = input_data["L_foot_yaw_angle_change"][0]
        self.cmd_dy = input_data["foot_clearance"][0]
        self.cmd_cooling_speed = input_data["cooling_speed"][0]

    def update_leg_status(self):
        """
        Get leg states from shared memory.
        """

        q = self.State_SMArr[LEG_STATE_POS]
        dq = self.State_SMArr[LEG_STATE_VEL]

        # print(dq)

        # right leg
        self.joint[HIP_YAW_R]["q"] = q[0]
        self.joint[HIP_ROLL_R]["q"] = q[1]
        self.joint[HIP_PITCH_R]["q"] = q[2]
        self.joint[KNEE_PITCH_R]["q"] = q[3]
        self.joint[ANKLE_PITCH_R]["q"] = q[4]

        self.joint[HIP_YAW_R]["dq"] = dq[0]
        self.joint[HIP_ROLL_R]["dq"] = dq[1]
        self.joint[HIP_PITCH_R]["dq"] = dq[2]
        self.joint[KNEE_PITCH_R]["dq"] = dq[3]
        self.joint[ANKLE_PITCH_R]["dq"] = dq[4]

        # left leg
        self.joint[HIP_YAW_L]["q"] = q[5]
        self.joint[HIP_ROLL_L]["q"] = q[6]
        self.joint[HIP_PITCH_L]["q"] = q[7]
        self.joint[KNEE_PITCH_L]["q"] = q[8]
        self.joint[ANKLE_PITCH_L]["q"] = q[9]

        self.joint[HIP_YAW_L]["dq"] = dq[5]
        self.joint[HIP_ROLL_L]["dq"] = dq[6]
        self.joint[HIP_PITCH_L]["dq"] = dq[7]
        self.joint[KNEE_PITCH_L]["dq"] = dq[8]
        self.joint[ANKLE_PITCH_L]["dq"] = dq[9]

    def set_command_leg_positions(self):
        """
        Set command leg joint positions to shared memory.
        """
        commands = {
            "BEAR_enable": np.array([1.0]),
            "BEAR_mode": np.array([self.BEAR_modes["position"]]),
            "goal_positions": np.array(
                [
                    self.joint[HIP_YAW_R]["q_goal"],
                    self.joint[HIP_ROLL_R]["q_goal"],
                    self.joint[HIP_PITCH_R]["q_goal"],
                    self.joint[KNEE_PITCH_R]["q_goal"],
                    self.joint[ANKLE_PITCH_R]["q_goal"],
                    self.joint[HIP_YAW_L]["q_goal"],
                    self.joint[HIP_ROLL_L]["q_goal"],
                    self.joint[HIP_PITCH_L]["q_goal"],
                    self.joint[KNEE_PITCH_L]["q_goal"],
                    self.joint[ANKLE_PITCH_L]["q_goal"],
                ]
            ),
        }
        BRUCE.All_PosCmd_SMArr[0] = 1.0
        BRUCE.All_PosCmd_SMArr[1] = self.BEAR_modes["position"]
        BRUCE.All_PosCmd_SMArr[2:12] = commands["goal_positions"]

    def set_command_leg_torques(self):
        """
        Set command leg joint torques to shared memory.
        """
        commands = {
            "BEAR_enable": np.array([1.0]),
            "BEAR_mode": np.array([self.BEAR_modes["torque"]]),
            "goal_torques": np.array(
                [
                    self.joint[HIP_YAW_R]["tau_goal"],
                    self.joint[HIP_ROLL_R]["tau_goal"],
                    self.joint[HIP_PITCH_R]["tau_goal"],
                    self.joint[KNEE_PITCH_R]["tau_goal"],
                    self.joint[ANKLE_PITCH_R]["tau_goal"],
                    self.joint[HIP_YAW_L]["tau_goal"],
                    self.joint[HIP_ROLL_L]["tau_goal"],
                    self.joint[HIP_PITCH_L]["tau_goal"],
                    self.joint[KNEE_PITCH_L]["tau_goal"],
                    self.joint[ANKLE_PITCH_L]["tau_goal"],
                ]
            ),
        }
        BRUCE.All_PosCmd_SMArr[0] = 1.0
        BRUCE.All_PosCmd_SMArr[1] = self.BEAR_modes["torque"]
        BRUCE.Leg_Torq_SMArr[2:12] = commands["goal_torques"]

    def set_command_leg_values(self):
        """
        Set command leg joint values to shared memory.
        """
        commands = {
            "BEAR_enable": np.array([1.0]),
            "BEAR_mode": np.array([self.BEAR_modes["force"]]),
            "goal_torques": np.array(
                [
                    self.joint[HIP_YAW_R]["tau_goal"],
                    self.joint[HIP_ROLL_R]["tau_goal"],
                    self.joint[HIP_PITCH_R]["tau_goal"],
                    self.joint[KNEE_PITCH_R]["tau_goal"],
                    self.joint[ANKLE_PITCH_R]["tau_goal"],
                    self.joint[HIP_YAW_L]["tau_goal"],
                    self.joint[HIP_ROLL_L]["tau_goal"],
                    self.joint[HIP_PITCH_L]["tau_goal"],
                    self.joint[KNEE_PITCH_L]["tau_goal"],
                    self.joint[ANKLE_PITCH_L]["tau_goal"],
                ]
            ),
            "goal_positions": np.array(
                [
                    self.joint[HIP_YAW_R]["q_goal"],
                    self.joint[HIP_ROLL_R]["q_goal"],
                    self.joint[HIP_PITCH_R]["q_goal"],
                    self.joint[KNEE_PITCH_R]["q_goal"],
                    self.joint[ANKLE_PITCH_R]["q_goal"],
                    self.joint[HIP_YAW_L]["q_goal"],
                    self.joint[HIP_ROLL_L]["q_goal"],
                    self.joint[HIP_PITCH_L]["q_goal"],
                    self.joint[KNEE_PITCH_L]["q_goal"],
                    self.joint[ANKLE_PITCH_L]["q_goal"],
                ]
            ),
            "goal_velocities": np.array(
                [
                    self.joint[HIP_YAW_R]["dq_goal"],
                    self.joint[HIP_ROLL_R]["dq_goal"],
                    self.joint[HIP_PITCH_R]["dq_goal"],
                    self.joint[KNEE_PITCH_R]["dq_goal"],
                    self.joint[ANKLE_PITCH_R]["dq_goal"],
                    self.joint[HIP_YAW_L]["dq_goal"],
                    self.joint[HIP_ROLL_L]["dq_goal"],
                    self.joint[HIP_PITCH_L]["dq_goal"],
                    self.joint[KNEE_PITCH_L]["dq_goal"],
                    self.joint[ANKLE_PITCH_L]["dq_goal"],
                ]
            ),
        }

        self.All_PosCmd_SMArr[0] = 1.0
        self.All_PosCmd_SMArr[1] = self.BEAR_modes["force"]
        self.Value_SMArr[LEG_TAR_POS] = commands["goal_positions"]
        self.Value_SMArr[LEG_TAR_VEL] = commands["goal_velocities"]
        self.Value_SMArr[LEG_TAR_TOR] = commands["goal_torques"]

    def update_arm_status(self):
        """
        Get arm states from shared memory.
        """
        q = self.State_SMArr[ARM_STATE_POS]
        dq = self.State_SMArr[ARM_STATE_VEL]

        # right arm
        self.joint[SHOULDER_PITCH_R]["q"] = q[0]
        self.joint[SHOULDER_ROLL_R]["q"] = q[1]
        self.joint[ELBOW_YAW_R]["q"] = q[2]

        self.joint[SHOULDER_PITCH_R]["dq"] = dq[0]
        self.joint[SHOULDER_ROLL_R]["dq"] = dq[1]
        self.joint[ELBOW_YAW_R]["dq"] = dq[2]

        # left arm
        self.joint[SHOULDER_PITCH_L]["q"] = q[3]
        self.joint[SHOULDER_ROLL_L]["q"] = q[4]
        self.joint[ELBOW_YAW_L]["q"] = q[5]

        self.joint[SHOULDER_PITCH_L]["dq"] = dq[3]
        self.joint[SHOULDER_ROLL_L]["dq"] = dq[4]
        self.joint[ELBOW_YAW_L]["dq"] = dq[5]

    def set_command_arm_positions(self):
        """
        Set command arm joint positions to shared memory.
        """
        commands = {
            "DXL_enable": np.array([1.0]),
            "DXL_mode": np.array([self.DXL_modes["position"]]),
            "goal_positions": np.array(
                [
                    self.joint[SHOULDER_PITCH_R]["q_goal"],
                    self.joint[SHOULDER_ROLL_R]["q_goal"],
                    self.joint[ELBOW_YAW_R]["q_goal"],
                    self.joint[SHOULDER_PITCH_L]["q_goal"],
                    self.joint[SHOULDER_ROLL_L]["q_goal"],
                    self.joint[ELBOW_YAW_L]["q_goal"],
                ]
            ),
        }
        BRUCE.All_PosCmd_SMArr[12] = 1.0
        BRUCE.All_PosCmd_SMArr[13] = self.DXL_modes["position"]
        BRUCE.All_PosCmd_SMArr[14:20] = commands["goal_positions"]

    # @property
    # def sair_armPos(self):

    # @sair_armPos.setter
    # def sair_armPos(self, value):

    def update_gamepad_status(self):
        """
        Get gamepad states from shared memory.
        """
        gamepad_data = MM.GAMEPAD_STATE.get()

        self.gamepad["U"] = gamepad_data["U"][0]
        self.gamepad["D"] = gamepad_data["D"][0]
        self.gamepad["L"] = gamepad_data["L"][0]
        self.gamepad["R"] = gamepad_data["R"][0]
        self.gamepad["A"] = gamepad_data["A"][0]
        self.gamepad["B"] = gamepad_data["B"][0]
        self.gamepad["X"] = gamepad_data["X"][0]
        self.gamepad["Y"] = gamepad_data["Y"][0]
        self.gamepad["LZ"] = gamepad_data["LZ"][0]
        self.gamepad["LS"] = gamepad_data["LS"][0]
        self.gamepad["LSP"] = gamepad_data["LSP"][0]
        self.gamepad["LSM"] = gamepad_data["LSM"][0]
        self.gamepad["RZ"] = gamepad_data["RZ"][0]
        self.gamepad["RS"] = gamepad_data["RS"][0]
        self.gamepad["RSP"] = gamepad_data["RSP"][0]
        self.gamepad["RSM"] = gamepad_data["RSM"][0]
        self.gamepad["ST"] = gamepad_data["ST"][0]
        self.gamepad["BK"] = gamepad_data["BK"][0]
        self.gamepad["ALT"] = gamepad_data["ALT"][0]
        self.gamepad["FN"] = gamepad_data["FN"][0]
        self.gamepad["LX"] = gamepad_data["LX"][0]
        self.gamepad["LY"] = gamepad_data["LY"][0]
        self.gamepad["RX"] = gamepad_data["RX"][0]
        self.gamepad["RY"] = gamepad_data["RY"][0]

    def thread_error(self):
        thread_data = MM.THREAD_STATE.get()
        thread_error = False
        if thread_data["dxl"][0] == 2.0:
            print(colored("DXL Thread Error! Terminate Now!", "red"))
            thread_error = True

        if thread_data["bear"][0] == 2.0:
            print(colored("BEAR Thread Error! Terminate Now!", "red"))
            thread_error = True
        elif thread_data["bear"][0] == 3.0:
            print(colored("BEAR IN ERROR! Terminate Now!", "red"))
            thread_error = True
        elif thread_data["bear"][0] == 4.0:
            print(colored("BEAR IN E-STOP! Terminate Now!", "red"))
            thread_error = True
            pass

        if thread_data["estimation"][0] == 2.0:
            print(colored("Estimation Thread Error! Terminate Now!", "red"))
            thread_error = True

        if thread_data["low_level"][0] == 2.0:
            print(colored("Low-Level Thread Error! Terminate Now!", "red"))
            thread_error = True

        if thread_data["high_level"][0] == 2.0:
            print(colored("High-Level Thread Error! Terminate Now!", "red"))
            thread_error = True

        if thread_data["top_level"][0] == 2.0:
            print(colored("Top-Level Thread Error! Terminate Now!", "red"))
            thread_error = True

        return thread_error

    @staticmethod
    def stop_robot():
        # BEAR_commands = {"BEAR_enable": np.array([0.0])}
        # MM.LEG_COMMAND.set(BEAR_commands)

        # DXL_commands = {"DXL_enable": np.array([0.0])}
        # MM.ARM_COMMAND.set(DXL_commands)
        raise NotImplementedError
        pass

    @staticmethod
    def damping_robot():
        raise NotImplementedError
        # commands = {"BEAR_enable": np.array([1.0]), "damping": np.array([1.0])}
        # MM.LEG_COMMAND.set(commands)

    @staticmethod
    def is_damping():
        return False
        leg_data = MM.LEG_STATE.get()
        return False if leg_data["damping"][0] == 0.0 else True

    def setModuleState(self, module, state):
        self.module_states[module] = state
    
    def getModuleState(self, module):
        return self.module_states[module]
    
    def get_time(self):
        # print(time.time(),self[EST]["time_stamp"][0])
        if self.module_states[MODULE_IFSIM]:
            return float(self.__getitem__(EST)["time_stamp"][0])
        else:
            return time.time()


    @staticmethod
    def stop_threading():
        THIS_IS_AN_INTENTIONAL_ERROR  # raise a stupid error to terminate the thread
