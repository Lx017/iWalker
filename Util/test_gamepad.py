#!usr/bin/env python
__author__    = "Westwood Robotics Corporation"
__email__     = "info@westwoodrobotics.io"
__copyright__ = "Copyright 2023 Westwood Robotics Corporation"
__date__      = "November 1, 2023"
__project__   = "BRUCE"
__version__   = "0.0.4"
__status__    = "Product"

"""
Script for monitoring gamepad data
"""

import time
import Settings.bot as RDS


if __name__ == "__main__":
    bot = RDS.BRUCE()
    
    while True:
        bot.update_gamepad_status()

        for key in list(bot.gamepad.keys()):
            if key in ["LX", "LY", "RX", "RY"]:
                print(key + ": {:.2f}".format(bot.gamepad[key]))
            else:
                print(key + ": {:.0f}".format(bot.gamepad[key]))

        for _ in range(len(list(bot.gamepad))):
            print("\033[1A", end="\x1b[2K")

        time.sleep(0.01)
