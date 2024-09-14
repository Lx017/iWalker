#!usr/bin/env python
__author__    = "Westwood Robotics Corporation"
__email__     = "info@westwoodrobotics.io"
__copyright__ = "Copyright 2023 Westwood Robotics Corporation"
__date__      = "November 1, 2023"
__project__   = "BRUCE"
__version__   = "0.0.4"
__status__    = "Product"

"""
Reset thread status
"""

import Startups.memory_manager as MM


# if __name__ == "__main__":
thread_data = MM.THREAD_STATE.get()
for key in list(thread_data.keys()):
    thread_data[key][0] = 0.0
MM.THREAD_STATE.set(thread_data)
    