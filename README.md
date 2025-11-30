This repository contains the software for a real-time robotic teleoperation system designed to provide precise, intuitive 3D control over a physical SO-101 industrial manipulator. 
It utilizes a standard smartphone as an accessible 3D input device, serving as a low-cost, high-reliability alternative to specialized motion capture equipment. 
The core technical achievement is the application of advanced filtering to achieve stable motion.

Key Features & Sophistication:
This project achieves stability and precision through the following sophisticated engineering solutions:

1. Extended Kalman Filter (EKF) Implementation:
The raw positional data from the smartphone is inherently noisy.
A 6-state Extended Kalman Filter is implemented to fuse noisy measurements with a constant velocity prediction model, resulting in smooth, jitter-free robot movement.

2. Thread-Safe Concurrency:
Data fetching from the mujoco_ar connector runs on a dedicated thread (getPhoneData).
All data access and updates are protected by a threading.Lock() to prevent a situation where the main control loop and getPhoneData method access the same variable at the same time,
ensuring data integrity in the main control loop.

3. Intuitive Control Mapping:
The system correctly scales, swaps, and inverts the phone's 3D motion into the robot's coordinate system ([X, -Y, Z] mapping) for an intuitive, human-centered control experience.

4. Fail-Safe Architecture: If an error is detected when sending a command to a robot, the program immediately shutsdown, including putting the robot to sleep.

Prerequisites:
Before starting the program, make sure to install the imports in each code file using pip or pip3 on your computer. 
Additionally, download the MujocoAR app on iOS or Android. 
Lastly, ensure that your computer is connected to your phone's hostpot.
Refer to this website for further details regarding the SO-101 robot: https://docs.phospho.ai/welcome
To run this code, you will need:
  SO-101 Robot: Must be powered on and accessible on the local network.
  Robot API: The robot's control API must be running at the specified ROBOT_IP:ROBOT_PORT.
  Smartphone: Running the MuJoCo AR app and connected to the same network as computer (i.e. hotspot).
  Python Environment: install all necessary imports using pip

Execute the Python script. The script will automatically connect to the phone and begin teleoperation upon device connection.

Operation & Safety
This script operates in two modes:

TELEOP: Real-time movement driven by the phone's position delta.

HOMING: Initiated by pressing the phone's designated button. Moves the robot to the predefined home_position using joint-space control.

Disclaimer: This version is provided for demonstration and educational purposes. It relies on explicit user control for safety as well. Use with caution.

External Value and Contribution
This project contributes to robotics scholarship by providing a reliable, low-cost model for Human-Robot Interaction (HRI). 
The use of an Extended Kalman Filter on commercial hardware demonstrates a practical method for achieving industrial-grade smoothness using accessible devices like smartphones.
