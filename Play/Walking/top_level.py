#!usr/bin/env python
__author__    = "Westwood Robotics Corporation"
__email__     = "info@westwoodrobotics.io"
__copyright__ = "Copyright 2023 Westwood Robotics Corporation"
__date__      = "November 1, 2023"
__project__   = "BRUCE"
__version__   = "0.0.4"
__status__    = "Product"

"""
User Input
"""

import os
import time
import select
import random
import subprocess
import tty, sys, termios
import Settings.Bruce as RDS
import Startups.memory_manager as MM
from termcolor import colored
from collections import defaultdict
from Settings.BRUCE_macros import *
from Play.Walking.walking_macros import *

# BRUCE SETUP
bot = RDS.BRUCE()

class keyEvent:
    lastKey = None

    @staticmethod
    def onPress(event):
        keyEvent.lastKey = event.name
        #print("input:", keyEvent.lastKey)

    @staticmethod
    def read():
        res = keyEvent.lastKey
        keyEvent.lastKey = None
        return res



class UserCommand(object):
    def __init__(self, controller="keyboard", command_frequency=20, display_frequency=10):
        # setup
        self.controller = controller
        self.loop_freq  = command_frequency
        self.display_dt = 1 / display_frequency
        self.last_display_time = time.time()
        self.loop_count = 0

        # user command
        self.in_operate     = True
        self.in_recover     = False
        self.in_wave        = False

        # operate parameter
        self.mode = BALANCE
        self.target_mode = BALANCE
        self.mode_name = {BALANCE: "BALANCE",
                          WALK:    "WALK"}
        self.parameter = defaultdict(lambda: defaultdict(int))
        for idx in PARAMETER_ID_LIST:
            self.parameter[idx]["value_raw"]    = PARAMETER_DEFAULT[idx]
            self.parameter[idx]["value_filter"] = PARAMETER_DEFAULT[idx]

        # misc
        self.tol         = 1e-3
        self.deg2rad     = np.pi / 180
        self.screen_flag = True if "bruce" in subprocess.check_output(["screen -ls; true"], shell=True).decode("utf-8") else False

        self.cooling_flag = False

        self.wave_count = 0
        self.wave_hand  = 1
        self.counter = 0
        # initialize the controller
        if self.controller == "keyboard":
            pass
            # self.filedes = termios.tcgetattr(sys.stdin)
            # tty.setcbreak(sys.stdin)
            # print("Press any key to continue!")
            # cmd = sys.stdin.read(1)[0]
            # termios.tcflush(sys.stdin, termios.TCIOFLUSH)  # clear buffer
        elif self.controller == "gamepad":
            bot.update_gamepad_status()
        
    def terminate(self):
        if self.controller == "keyboard":
            pass
            #termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.filedes)
        elif self.controller == "gamepad":
            pass

    def update(self,launch_time):
        if self.in_recover:
            self.recover()
        elif self.in_operate:
            self.operate(launch_time)

        if self.in_wave:
            self.wave()

        if time.time() - self.last_display_time > self.display_dt:
            self.last_display_time = time.time()
            self.display()

        self.loop_count = 0 if self.loop_count > 1e8 else self.loop_count + 1

    def recover(self):
        self.in_recover = False
        for idx in PARAMETER_ID_LIST:
            if PARAMETER_RECOVER[idx] == "y":
                if np.abs(PARAMETER_DEFAULT[idx] - self.parameter[idx]["value_raw"]) > self.tol:
                    self.in_recover = True
                    if self.loop_count % int(self.loop_freq / 20) == 0:  # 20 Hz max
                        self.parameter[idx]["value_raw"] = PARAMETER_DEFAULT[idx] if np.abs(PARAMETER_DEFAULT[idx] - self.parameter[idx]["value_raw"]) < PARAMETER_INCREMENT[idx] else self.parameter[idx]["value_raw"] + np.sign(PARAMETER_DEFAULT[idx] - self.parameter[idx]["value_raw"]) * PARAMETER_INCREMENT[idx]
        
        if self.in_recover is False:
            if self.mode != self.target_mode:
                # time.sleep(0.5)
                self.mode = self.target_mode

    def operate(self, launch_time):
        START_WALKING_TIME = 5
        if self.controller == "keyboard":
            # read keyboard
            is_input, _, _ = select.select([sys.stdin], [], [], 0.01)  # adding timeout to this keyboard input
            cmd = keyEvent.read()#sys.stdin.read(1)[0] if is_input else None
            # cmd = input()
            elapsed_time = time.time() - launch_time
            
            if elapsed_time>START_WALKING_TIME and self.counter == 0:
                self.target_mode = WALK
                self.in_recover = True
                self.counter = 1

                
                for idx in PARAMETER_ID_LIST:
                    if self.mode in PARAMETER_MODE_LIST[idx]:
                        if cmd == PARAMETER_BUTTON_PLUS[idx] and PARAMETER_MAX[idx] - self.parameter[idx]["value_raw"] > self.tol:
                            self.parameter[idx]["value_raw"] += PARAMETER_INCREMENT[idx]
                        elif cmd == PARAMETER_BUTTON_MINUS[idx] and self.parameter[idx]["value_raw"] - PARAMETER_MIN[idx] > self.tol:
                            self.parameter[idx]["value_raw"] -= PARAMETER_INCREMENT[idx]
        elif self.controller == "gamepad":
            bot.update_gamepad_status()

            if bot.gamepad["LZ"] and bot.gamepad["RZ"] :
                bot.stop_threading()

            if bot.gamepad["BK"]:
                self.in_recover = True

            if bot.gamepad["A"]:
                self.in_wave = True

            if bot.gamepad["ST"]:
                if bot.gamepad["BK"]:
                    self.target_mode = BALANCE
                elif bot.gamepad["Y"]:
                    self.target_mode = WALK
                
                if self.mode != self.target_mode:
                    self.in_recover = True
                    
                if bot.gamepad["FN"] and PARAMETER_MAX[COOLING_SPEED] - self.parameter[COOLING_SPEED]["value_raw"] > self.tol:
                    self.parameter[COOLING_SPEED]["value_raw"] += PARAMETER_INCREMENT[COOLING_SPEED]

                if self.mode == WALK:
                    if bot.gamepad["LSP"] and PARAMETER_MAX[FOOT_YAW_LEFT] - self.parameter[FOOT_YAW_LEFT]["value_raw"] > self.tol:
                        self.parameter[FOOT_YAW_LEFT]["value_raw"] += PARAMETER_INCREMENT[FOOT_YAW_LEFT]
                    elif bot.gamepad["LSM"] and self.parameter[FOOT_YAW_LEFT]["value_raw"] - PARAMETER_MIN[FOOT_YAW_LEFT] > self.tol:
                        self.parameter[FOOT_YAW_LEFT]["value_raw"] -= PARAMETER_INCREMENT[FOOT_YAW_LEFT]

                    if bot.gamepad["RSP"] and PARAMETER_MAX[FOOT_YAW_RIGHT] - self.parameter[FOOT_YAW_RIGHT]["value_raw"] > self.tol:
                        self.parameter[FOOT_YAW_RIGHT]["value_raw"] += PARAMETER_INCREMENT[FOOT_YAW_RIGHT]
                    elif bot.gamepad["RSM"] and self.parameter[FOOT_YAW_RIGHT]["value_raw"] - PARAMETER_MIN[FOOT_YAW_RIGHT] > self.tol:
                        self.parameter[FOOT_YAW_RIGHT]["value_raw"] -= PARAMETER_INCREMENT[FOOT_YAW_RIGHT]

                    if bot.gamepad["R"] and PARAMETER_MAX[FOOT_CLEARANCE] - self.parameter[FOOT_CLEARANCE]["value_raw"] > self.tol:
                        self.parameter[FOOT_CLEARANCE]["value_raw"] += PARAMETER_INCREMENT[FOOT_CLEARANCE]
                    elif bot.gamepad["L"] and self.parameter[FOOT_CLEARANCE]["value_raw"] - PARAMETER_MIN[FOOT_CLEARANCE] > self.tol:
                        self.parameter[FOOT_CLEARANCE]["value_raw"] -= PARAMETER_INCREMENT[FOOT_CLEARANCE]
            else:
                if bot.gamepad["FN"] and self.parameter[COOLING_SPEED]["value_raw"] - PARAMETER_MIN[COOLING_SPEED] > self.tol:
                    self.parameter[COOLING_SPEED]["value_raw"] -= PARAMETER_INCREMENT[COOLING_SPEED]

                if self.mode == BALANCE:
                    if bot.gamepad["U"] and PARAMETER_MAX[COMPos_X] - self.parameter[COMPos_X]["value_raw"] > self.tol:
                        self.parameter[COMPos_X]["value_raw"] += PARAMETER_INCREMENT[COMPos_X]
                    elif bot.gamepad["D"] and self.parameter[COMPos_X]["value_raw"] - PARAMETER_MIN[COMPos_X] > self.tol:
                        self.parameter[COMPos_X]["value_raw"] -= PARAMETER_INCREMENT[COMPos_X]
                    self.parameter[COMPos_Y]["value_raw"] = self.axis2value(-bot.gamepad["LX"], PARAMETER_MAX[COMPos_Y], PARAMETER_MIN[COMPos_Y])
                    self.parameter[COMPos_Z]["value_raw"] = self.axis2value(bot.gamepad["LY"], PARAMETER_MAX[COMPos_Z], PARAMETER_MIN[COMPos_Z])

                    self.parameter[BODY_ORIENTATION_X]["value_raw"] = self.axis2value(bot.gamepad["RX"], PARAMETER_MAX[BODY_ORIENTATION_X], PARAMETER_MIN[BODY_ORIENTATION_X])
                    self.parameter[BODY_ORIENTATION_Y]["value_raw"] = self.axis2value(bot.gamepad["RY"], PARAMETER_MAX[BODY_ORIENTATION_Y], PARAMETER_MIN[BODY_ORIENTATION_Y])
                    if bot.gamepad["RSP"] and PARAMETER_MAX[BODY_ORIENTATION_Z] - self.parameter[BODY_ORIENTATION_Z]["value_raw"] > self.tol:
                        self.parameter[BODY_ORIENTATION_Z]["value_raw"] += PARAMETER_INCREMENT[BODY_ORIENTATION_Z]
                    elif bot.gamepad["RSM"] and self.parameter[BODY_ORIENTATION_Z]["value_raw"] - PARAMETER_MIN[BODY_ORIENTATION_Z] > self.tol:
                        self.parameter[BODY_ORIENTATION_Z]["value_raw"] -= PARAMETER_INCREMENT[BODY_ORIENTATION_Z]
                elif self.mode == WALK:
                    self.parameter[COMVel_X]["value_raw"] = self.axis2value(bot.gamepad["LY"], PARAMETER_MAX[COMVel_X], PARAMETER_MIN[COMVel_X])
                    self.parameter[COMVel_Y]["value_raw"] = self.axis2value(-bot.gamepad["LX"], PARAMETER_MAX[COMVel_Y], PARAMETER_MIN[COMVel_Y])

                    if bot.gamepad["RSP"]:
                        self.parameter[BODY_YAW_RATE]["value_raw"] = PARAMETER_MAX[BODY_YAW_RATE]
                    elif bot.gamepad["RSM"]:
                        self.parameter[BODY_YAW_RATE]["value_raw"] = PARAMETER_MIN[BODY_YAW_RATE]
                    else:
                        self.parameter[BODY_YAW_RATE]["value_raw"] = 0

        # set user input to shared memory
        for idx in PARAMETER_ID_LIST:
            self.parameter[idx]["value_filter"] = MF.exp_filter(self.parameter[idx]["value_filter"], self.parameter[idx]["value_raw"], 0.85)
        input_data = {"walk":                        np.array([self.mode]),
                      "com_xy_velocity":             np.array([self.parameter[COMVel_X]["value_filter"], self.parameter[COMVel_Y]["value_filter"]]),
                      "yaw_rate":                    np.array([self.parameter[BODY_YAW_RATE]["value_filter"]]) * self.deg2rad,
                      "COMPos_change_scaled":  np.array([self.parameter[COMPos_X]["value_filter"], self.parameter[COMPos_Y]["value_filter"], self.parameter[COMPos_Z]["value_filter"]]),
                      "body_euler_angle_change":     np.array([self.parameter[BODY_ORIENTATION_X]["value_filter"], self.parameter[BODY_ORIENTATION_Y]["value_filter"], self.parameter[BODY_ORIENTATION_Z]["value_filter"]]) * self.deg2rad,
                      "R_foot_yaw_angle_change": np.array([self.parameter[FOOT_YAW_RIGHT]["value_filter"]]) * self.deg2rad,
                      "L_foot_yaw_angle_change":  np.array([self.parameter[FOOT_YAW_LEFT]["value_filter"]])  * self.deg2rad,
                      "foot_clearance":              np.array([self.parameter[FOOT_CLEARANCE]["value_filter"]]),
                      "cooling_speed":               np.array([self.parameter[COOLING_SPEED]["value_raw"]]),
                      }
        MM.USER_COMMAND.set(input_data)

    def wave(self):
        if self.wave_count < arm_trajectory[0].size:
            if self.wave_hand == -1:
                for idx, joint in enumerate(ARM_JOINT_LIST[0:3]):
                    bot.joint[joint]["q_goal"] = arm_trajectory[idx][self.wave_count]
                for idx, joint in enumerate(ARM_JOINT_LIST[3:6]):
                    bot.joint[joint]["q_goal"] = arm_position_nominal[idx + 3]
            elif self.wave_hand == 1:
                for idx, joint in enumerate(ARM_JOINT_LIST[0:3]):
                    bot.joint[joint]["q_goal"] = arm_position_nominal[idx]
                for idx, joint in enumerate(ARM_JOINT_LIST[3:6]):
                    bot.joint[joint]["q_goal"] = arm_trajectory[idx + 3][self.wave_count]
            else:
                for idx, joint in enumerate(ARM_JOINT_LIST):
                    bot.joint[joint]["q_goal"] = arm_trajectory[idx][self.wave_count]
            bot.set_command_arm_positions()
            self.wave_count += 1
        else:
            self.wave_count = 0
            self.in_wave = False
            val = random.random()
            if val > 0.666:
                self.wave_hand = -1
            elif val > 0.333:
                self.wave_hand = 1
            else:
                self.wave_hand = 2

    def display(self):
        # get BEAR info
        return
        leg_data = MM.LEG_STATE.get()
        vol_msg  = colored("  LOW!!!", "red") if leg_data["voltage"][0] < 14.5 else ""
        if leg_data["temperature"][0] >= 75.0:
            tem_msg = colored("HIGH!!!",  "red")
            if self.parameter[COOLING_SPEED]["value_raw"] >= 2:
                self.cooling_flag = True
            else:
                if not self.cooling_flag:
                    self.parameter[COOLING_SPEED]["value_raw"] = 2
                    self.cooling_flag = True
        else:
            tem_msg = ""
            if leg_data["temperature"][0] <= 65.0:
                if self.parameter[COOLING_SPEED]["value_raw"] == 0:
                    self.cooling_flag = False
                else:
                    if self.cooling_flag:
                        self.parameter[COOLING_SPEED]["value_raw"] = 0
                        self.cooling_flag = False

        if 0:
            os.system("clear")
            print("====== The User Input Thread is running at ???? Hz... ======")
            print("")
            print("BEAR Voltage:           ", self.float2str(leg_data["voltage"][0]),      "[V]", vol_msg)
            print("BEAR Temperature:       ", self.float2str(leg_data["temperature"][0]), "[°C]", tem_msg)
            print("BEAR Cooling Speed(-/+):", self.float2str(self.parameter[COOLING_SPEED]["value_raw"] * 10), "[%]")
            print("_________")
            print("")
            print("IN", self.mode_name[self.mode], "(press SPACEBAR + 0/1 to balance/walk)")
            print("_________")
            print("")
            if self.mode == BALANCE:
                print("Body Orientation")
                print("-      roll (T/Y):",  self.float2str(self.parameter[BODY_ORIENTATION_X]["value_raw"], "ang"), "[deg]")
                print("-     pitch (U/I):",  self.float2str(self.parameter[BODY_ORIENTATION_Y]["value_raw"], "ang"), "[deg]")
                print("-       yaw (O/P):",  self.float2str(self.parameter[BODY_ORIENTATION_Z]["value_raw"], "ang"), "[deg]")
                print("")
                print("CoM Position")
                print("-  sagittal (F/G):",  self.float2str(self.parameter[COMPos_X]["value_raw"], "len"), "[%]")
                print("-   lateral (H/J):",  self.float2str(self.parameter[COMPos_Y]["value_raw"], "len"), "[%]")
                print("-  vertical (K/L):",  self.float2str(self.parameter[COMPos_Z]["value_raw"], "len"), "[cm]")
            elif self.mode == WALK:
                print("CoM Velocity")
                print("-   sagittal (W/S):", self.float2str(self.parameter[COMVel_X]["value_raw"], "len"), "[cm/s]")
                print("-    lateral (A/D):", self.float2str(self.parameter[COMVel_Y]["value_raw"], "len"), "[cm/s]")
                print("-        yaw (Q/E):", self.float2str(self.parameter[BODY_YAW_RATE]["value_raw"],  "ang"), "[deg/s]")
                print("")
                print("Swing Foot")
                print("-  right yaw (Z/X):", self.float2str(self.parameter[FOOT_YAW_RIGHT]["value_raw"], "ang"), "[deg]")
                print("-   left yaw (C/V):", self.float2str(self.parameter[FOOT_YAW_LEFT]["value_raw"],  "ang"), "[deg]")
                print("- foot clr x (N/M):", self.float2str(self.parameter[FOOT_CLEARANCE]["value_raw"], "len"), "[cm]")
            print("")
            print("Press R to recover.")
            if self.screen_flag:
                print(colored("Press Ctrl+A+D to go back to the terminal.", "yellow"))

    @staticmethod
    def axis2value(x, x_max, x_min):
        n = 1
        if x > 0:
            val = x**n * x_max
        else:
            val = (-x)**n * x_min
        return val
    
    @staticmethod
    def float2str(f, t="default"):
        if t == "len":
            f = round(f * 1000) / 10
        else:
            f = round(f * 100) / 100

        if f >= 100:
            s = "+" + str(f)[:-1]
        elif f >= 10:
            s = "+" + str(f)
        elif 0 < f < 10:
            s = "+" + str(f) + "0"
        elif f == 0:
            s = " " + str(f) + "0"
        elif -10 < f < 0:
            s = str(f) + "0"
        else:
            s = str(f)

        return s
    
def main_loop(uc, loop_freq, loop_duration):
    t0 = time.time()
    thread_run = False
    try:
        while True:
            # time info
            loop_start_time = time.time() - t0

            if time.time() - t0 > 1:
                if not thread_run:
                    MM.THREAD_STATE.set({"top_level": np.array([1.0])}, opt="only")  # thread is running
                    thread_run = True

                # check thread error
                if bot.thread_error():
                    pass#bot.stop_threading()

            # update user command
            uc.update(t0)

            # check time to ensure the thread stays at a consistent running loop
            loop_target_time = loop_start_time + loop_duration
            while time.time() - t0 < loop_target_time:
                pass
    finally:
        uc.terminate()


def topMain():
    #keyboard.on_press(keyEvent.onPress)
    # BRUCE Setup
    bot = RDS.BRUCE()

    # Control Frequency
    loop_freq     = 20  # run     at 20 Hz
    display_freq  = 10  # display at 10 Hz
    loop_duration = 1 / loop_freq


    # User Command Input Setup
    ctrl = "gamepad" if gamepad else "keyboard"
    uc = UserCommand(controller=ctrl, display_frequency=display_freq)
    
    try:
        MM.THREAD_STATE.set({"top_level": np.array([1.0])}, opt="only")  # thread is running
        main_loop(uc, loop_freq, loop_duration)
    except (NameError, KeyboardInterrupt) as error:
        MM.THREAD_STATE.set({"top_level": np.array([0.0])}, opt="only")  # thread is stopped
    except Exception as error:
        print(error)
        MM.THREAD_STATE.set({"top_level": np.array([2.0])}, opt="only")  # thread in error