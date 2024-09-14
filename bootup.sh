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

# terminate threads
for i in $(seq 1 5)
do
  screen -S bruce -p bear       -X stuff "^C"
  screen -S bruce -p dxl        -X stuff "^C"
  screen -S bruce -p estimation -X stuff "^C"
  screen -S bruce -p low_level  -X stuff "^C"
  screen -S bruce -p high_level -X stuff "^C"
  screen -S bruce -p top_level  -X stuff "^C"
  sleep 0.1s
done

# reset thread status
screen -S bruce -p dxl -X stuff 'python3 -m Startups.reset_thread^M'

# run_dxl
read -p $'\e[31mATTENTION: Press ENTER to start DXL control! DXL LED will be on!\e[0m'
screen -S bruce -p dxl -X stuff 'python3 -m Startups.run_dxl^M'
sleep 1.5s
echo '====== Dynamixel Motor Online ======'
echo ''

# run_bear
read -p $'\e[31mATTENTION: Press ENTER to start BEAR control! BEAR LED will be on!\e[0m'
screen -S bruce -p bear -X stuff 'python3 -m Startups.run_bear^M'
sleep 1.5s
echo '====== BEAR Motor Online ======'
echo ''

# initialize & run_estimation
read -p $'\e[31mATTENTION: Press START to initialize BRUCE! Limbs will move!\e[0m'
screen -S bruce -p estimation -X stuff 'python3 -m Play.initialize^M'
sleep 0.5s
screen -S bruce -p estimation -X stuff "s^M"
sleep 2s
echo '====== BRUCE Initialized ======'
echo ''

read -p $'\e[31mATTENTION: Place BRUCE on the ground! Press Enter to start estimation! Wait until arms move!\e[0m'
screen -S bruce -p estimation -X stuff 'python3 -m Startups.run_estimation^M'
sleep 1.5s
echo '====== State Estimation Online ======'
echo ''

# low_level
screen -S bruce -p low_level -X stuff 'python3 -m Play.Walking.low_level^M'
sleep 0.5s
read -p $'\e[31mATTENTION: Press ENTER to start low-level control! BRUCE may twitch a bit!\e[0m'
sleep 0.5s
screen -S bruce -p low_level -X stuff "y^M"
sleep 0.5s
echo '====== Low-Level Controller Online ======'
echo ''

# high_level
screen -S bruce -p high_level -X stuff 'python3 -m Play.Walking.high_level^M'
sleep 0.5s
read -p $'\e[31mATTENTION: Press ENTER to start high-level control! BRUCE will rise a bit!\e[0m'
sleep 0.5s
screen -S bruce -p high_level -X stuff "y^M"
sleep 0.5s
echo '====== High-Level Controller Online ======'
echo ''

# top_level
read -p $'\e[31mATTENTION: Press ENTER to enter cockpit!\e[0m'
screen -S bruce -p top_level -X stuff 'python3 -m Play.Walking.top_level^M'
sleep 0.5s
echo '====== Top-Level Controller Online ======'
echo ''

input='n'
while [ $input != 'y' ]
do
  screen -r bruce -p top_level
  sleep 0.2s
  read -p 'Exit? (y/n)' input
done

# terminate
./terminate.sh