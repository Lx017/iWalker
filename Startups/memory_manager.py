#!usr/bin/env python
__author__    = "Westwood Robotics Corporation"
__email__     = "info@westwoodrobotics.io"
__copyright__ = "Copyright 2023 Westwood Robotics Corporation"
__date__      = "November 1, 2023"
__project__   = "BRUCE"
__version__   = "0.0.4"
__status__    = "Product"

"""
Pre-generate the shared memory segments before using them in the rest of the scripts
"""

import numpy as np
from Library.SHARED_MEMORY import Manager as shmx
from multiprocessing import shared_memory as sm


# Create Shared Memory Segments
# Thread state
THREAD_STATE = shmx.SHMEMSEG(robot_name="BRUCE", seg_name="THREAD_STATE", init=False)
THREAD_STATE.add_block(name="simulation", data=np.zeros(1))
THREAD_STATE.add_block(name="bear",       data=np.zeros(1))
THREAD_STATE.add_block(name="dxl",        data=np.zeros(1))
THREAD_STATE.add_block(name="estimation", data=np.zeros(1))
THREAD_STATE.add_block(name="low_level",  data=np.zeros(1))
THREAD_STATE.add_block(name="high_level", data=np.zeros(1))
THREAD_STATE.add_block(name="top_level",  data=np.zeros(1))


# Gamepad state
GAMEPAD_STATE = shmx.SHMEMSEG(robot_name="BRUCE", seg_name="GAMEPAD_STATE", init=False)
GAMEPAD_STATE.add_block(name="U",   data=np.zeros(1))
GAMEPAD_STATE.add_block(name="D",   data=np.zeros(1))
GAMEPAD_STATE.add_block(name="L",   data=np.zeros(1))
GAMEPAD_STATE.add_block(name="R",   data=np.zeros(1))
GAMEPAD_STATE.add_block(name="A",   data=np.zeros(1))
GAMEPAD_STATE.add_block(name="B",   data=np.zeros(1))
GAMEPAD_STATE.add_block(name="X",   data=np.zeros(1))
GAMEPAD_STATE.add_block(name="Y",   data=np.zeros(1))
GAMEPAD_STATE.add_block(name="LZ",  data=np.zeros(1))
GAMEPAD_STATE.add_block(name="LS",  data=np.zeros(1))
GAMEPAD_STATE.add_block(name="LSP", data=np.zeros(1))
GAMEPAD_STATE.add_block(name="LSM", data=np.zeros(1))
GAMEPAD_STATE.add_block(name="RZ",  data=np.zeros(1))
GAMEPAD_STATE.add_block(name="RS",  data=np.zeros(1))
GAMEPAD_STATE.add_block(name="RSP", data=np.zeros(1))
GAMEPAD_STATE.add_block(name="RSM", data=np.zeros(1))
GAMEPAD_STATE.add_block(name="ST",  data=np.zeros(1))
GAMEPAD_STATE.add_block(name="BK",  data=np.zeros(1))
GAMEPAD_STATE.add_block(name="ALT", data=np.zeros(1))
GAMEPAD_STATE.add_block(name="FN",  data=np.zeros(1))
GAMEPAD_STATE.add_block(name="LX",  data=np.zeros(1))
GAMEPAD_STATE.add_block(name="LY",  data=np.zeros(1))
GAMEPAD_STATE.add_block(name="RX",  data=np.zeros(1))
GAMEPAD_STATE.add_block(name="RY",  data=np.zeros(1))


# User Command
USER_COMMAND = shmx.SHMEMSEG(robot_name="BRUCE", seg_name="USER_COMMAND", init=False)
USER_COMMAND.add_block(name="time_stamp",                  data=np.zeros(1))
USER_COMMAND.add_block(name="walk",                        data=np.zeros(1))
USER_COMMAND.add_block(name="com_xy_velocity",             data=np.zeros(2))
USER_COMMAND.add_block(name="yaw_rate",                    data=np.zeros(1))
USER_COMMAND.add_block(name="COMPos_change_scaled",  data=np.zeros(3))
USER_COMMAND.add_block(name="body_euler_angle_change",     data=np.zeros(3))
USER_COMMAND.add_block(name="R_foot_yaw_angle_change", data=np.zeros(1))
USER_COMMAND.add_block(name="L_foot_yaw_angle_change",  data=np.zeros(1))
USER_COMMAND.add_block(name="foot_clearance",              data=np.zeros(1))
USER_COMMAND.add_block(name="cooling_speed",               data=np.zeros(1))


def init():
    """Init if main"""
    THREAD_STATE.initialize      = True
    GAMEPAD_STATE.initialize     = True
    USER_COMMAND.initialize      = True


def connect():
    """Connect and create segment"""
    THREAD_STATE.connect_segment()
    GAMEPAD_STATE.connect_segment()
    USER_COMMAND.connect_segment()


if __name__ == "__main__":
    init()
    connect()
else:
    connect()


