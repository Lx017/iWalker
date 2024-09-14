#!/bin/bash
<< 'COMMENT'
__author__    = "Westwood Robotics Corporation"
__email__     = "info@westwoodrobotics.io"
__copyright__ = "Copyright 2023 Westwood Robotics Corporation"
__date__      = "November 1, 2023"
__project__   = "BRUCE"
__version__   = "0.0.4"
__status__    = "Product"
COMMENT

# set password
readonly PASSWORD=khadas

# USB low latency setup
echo $PASSWORD | sudo -S sh -c "echo '1' > /sys/bus/usb-serial/devices/ttyUSB0/latency_timer"
echo $PASSWORD | sudo -S sh -c "echo '1' > /sys/bus/usb-serial/devices/ttyUSB1/latency_timer"
echo $PASSWORD | sudo -S sh -c "echo '1' > /sys/bus/usb-serial/devices/ttyACM0/latency_timer"