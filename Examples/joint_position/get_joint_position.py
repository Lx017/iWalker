#!usr/bin/env python
__author__    = "Westwood Robotics Corporation"
__email__     = "info@westwoodrobotics.io"
__copyright__ = "Copyright 2023 Westwood Robotics Corporation"
__date__      = "November 1, 2023"
__project__   = "BRUCE"
__version__   = "0.0.4"
__status__    = "Product"

"""
Get joint positions (change "q" to "dq" for joint velocities)
"""

import time
import Settings.Bruce as RDS
from Settings.BRUCE_macros import *


if __name__ == "__main__":
    # BRUCE setup
    bot = RDS.BRUCE()

    LINE_UP = "\033[1A"
    LINE_CLEAR = "\x1b[2K"
    while True:
        bot.update_arm_status()
        bot.update_leg_status()

        print("Right Arm: {:.2f} / {:.2f} / {:.2f}".format(bot.joint[SHOULDER_PITCH_R]["q"],
                                                           bot.joint[SHOULDER_ROLL_R]["q"],
                                                           bot.joint[ELBOW_YAW_R]["q"]))
        print(" Left Arm: {:.2f} / {:.2f} / {:.2f}".format(bot.joint[SHOULDER_PITCH_L]["q"],
                                                           bot.joint[SHOULDER_ROLL_L]["q"],
                                                           bot.joint[ELBOW_YAW_L]["q"]))
        print("Right Leg: {:.2f} / {:.2f} / {:.2f} / {:.2f} / {:.2f}".format(bot.joint[HIP_YAW_R]["q"],
                                                                             bot.joint[HIP_ROLL_R]["q"],
                                                                             bot.joint[HIP_PITCH_R]["q"],
                                                                             bot.joint[KNEE_PITCH_R]["q"],
                                                                             bot.joint[ANKLE_PITCH_R]["q"]))
        print(" Left Leg: {:.2f} / {:.2f} / {:.2f} / {:.2f} / {:.2f}".format(bot.joint[HIP_YAW_L]["q"],
                                                                             bot.joint[HIP_ROLL_L]["q"],
                                                                             bot.joint[HIP_PITCH_L]["q"],
                                                                             bot.joint[KNEE_PITCH_L]["q"],
                                                                             bot.joint[ANKLE_PITCH_L]["q"]))
        for _ in range(4):
            print(LINE_UP, end=LINE_CLEAR)

        time.sleep(0.01)
