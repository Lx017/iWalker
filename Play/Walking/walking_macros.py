#!usr/bin/env python
__author__    = "Westwood Robotics Corporation"
__email__     = "info@westwoodrobotics.io"
__copyright__ = "Copyright 2023 Westwood Robotics Corporation"
__date__      = "November 1, 2023"
__project__   = "BRUCE"
__version__   = "0.0.4"
__status__    = "Product"

"""
Script that holds useful macros for DCM walking tuning
"""

import numpy as np
from collections import defaultdict


gamepad = False  # if using gamepad or not
run_KF  = True   # if KF or CF for estimation

# HIGH-LEVEL
# Walking
# foot swing trajectory
Ts_desired     = 0.16   # desired stance phase duration [s]####0.24
Ts_min = 0.14   # minimum stance duration       [s]
Ts_max = 0.18   # maximum stance duration       [s]
T_buff = 0.05   # stop plan before T - T_buff   [s]

Txi = 0.00      # x stay before Txi
Txn = 0.05      # x go to nominal before Txn
Txf = 0.00      # x arrive before T - Txf

Tyi = 0.00      # y stay before Tyi
Tyn = 0.05      # y go to nominal before Tyn
Tyf = 0.00      # y arrive before T - Tyf

Tzm = 0.06      # desired swing apex time [s]
Tzf = 0.00      # z arrive before T - Tzf

z_max = 0.01    # left  swing apex height [m]#### 0.025

z_final = -0.001 # final foot height [m]

hz = 0.345      # desired CoM height [m]#0.345

yaw_f_offset = 0.02  # foot yaw offset [rad]

# kinematic reachability [m]
lx  = 0.20      # max longitudinal step length
lyi = 0.02      # min lateral distance between feet
lyo = 0.20      # max lateral distance between feet

# velocity offset compensation [m]
bx_offset = 0.000  # set to negative if BRUCE tends to go forward 0.25
by_offset = 0.000  # set to negative if BRUCE tends to go left

# Stance
ka = -0.0       # x position of CoM from the center of foot, in scale of 1/2 foot length
                # ka = 1 puts CoM at the front tip of foot

# TOP-LEVEL
COMPos_X     = 0
COMPos_Y     = 1
COMPos_Z     = 2

BODY_ORIENTATION_X = 3
BODY_ORIENTATION_Y = 4
BODY_ORIENTATION_Z = 5

COMVel_X     = 6
COMVel_Y     = 7
BODY_YAW_RATE      = 8

FOOT_YAW_RIGHT     = 9
FOOT_YAW_LEFT      = 10

FOOT_CLEARANCE     = 11

COOLING_SPEED      = 12

PARAMETER_ID_LIST      = range(13)
PARAMETER_INCREMENT    = [ 0.05,  0.05,  0.002,       1,     1,     2,    0.01,  0.01,     1,       1,     1,    0.01,       1]
PARAMETER_DEFAULT      = [ 0.00,  0.00,  0.000,       0,     0,     0,     0.0,   0.0,     0,       0,     0,    0.06,       0]
PARAMETER_MAX          = [ 0.20,  0.50,  0.020,       8,    10,    20,    0.15,  0.25,    15,      10,    10,    0.10,       5]
PARAMETER_MIN          = [-0.20, -0.50, -0.030,      -8,   -10,   -20,   -0.15, -0.25,   -15,     -10,   -10,    0.04,       0]
PARAMETER_BUTTON_PLUS  = [  "g",   "j",    "l",     "y",   "i",   "p",     "w",   "a",   "q",     "x",   "v",     "m",     "="]
PARAMETER_BUTTON_MINUS = [  "f",   "h",    "k",     "t",   "u",   "o",     "s",   "d",   "e",     "z",   "c",     "n",     "-"]
PARAMETER_TYPE         = ["len", "len",  "len",   "ang", "ang", "ang",   "len", "len", "ang",   "ang", "ang",   "len",   "len"]
PARAMETER_RECOVER      = [  "y",   "y",    "y",     "y",   "y",   "y",     "y",   "y",   "y",     "y",   "y",     "y",     "n"]

BALANCE = 0
WALK    = 1
PARAMETER_MODE_LIST = {COMPos_X:     [BALANCE],
                       COMPos_Y:     [BALANCE],
                       COMPos_Z:     [BALANCE],
                       BODY_ORIENTATION_X: [BALANCE],
                       BODY_ORIENTATION_Y: [BALANCE],
                       BODY_ORIENTATION_Z: [BALANCE],
                       COMVel_X:     [WALK],
                       COMVel_Y:     [WALK],
                       BODY_YAW_RATE:      [WALK],
                       FOOT_YAW_RIGHT:     [WALK],
                       FOOT_YAW_LEFT:      [WALK],
                       FOOT_CLEARANCE:     [WALK],
                       COOLING_SPEED:      [BALANCE, WALK]
                       }



































# wave trajectory
arm_position_nominal = np.array([-0.7,  1.3,  2.0, 
                                  0.7, -1.3, -2.0])
arm_position_goal    = np.array([0.0, -1.2, 0.0,
                                 0.0,  1.2, 0.0])
arm_trajectory = defaultdict()

for i in range(6):
    arm_trajectory[i] = np.linspace(arm_position_nominal[i], arm_position_goal[i], 20, endpoint=True)

traj_time = np.linspace(0, 2.75 * 2 * np.pi, 30)
for tdx in traj_time:
    arm_trajectory[1] = np.append(arm_trajectory[1], arm_position_goal[1] - 0.3 * np.sin(tdx))
    arm_trajectory[4] = np.append(arm_trajectory[4], arm_position_goal[4] + 0.3 * np.sin(tdx))

    for i in [0, 2, 3, 5]:
        arm_trajectory[i] = np.append(arm_trajectory[i], arm_position_goal[i])

for i in range(6):
    arm_trajectory[i] = np.append(arm_trajectory[i], np.linspace(arm_trajectory[i][-1], arm_position_nominal[i], 20, endpoint=True))
