import time
import requests
import threading
import numpy as np
from mujoco_ar import MujocoARConnector
import time

print("Imports completed")

ROBOT_IP: str = "192.168.18.64"
ROBOT_PORT: int = 80

def call_to_api(endpoint: str, data: dict = {}):
    try:
        response = requests.post(f"http://{ROBOT_IP}:{ROBOT_PORT}/{endpoint}", json=data, timeout=5)
        status_code = response.status_code

        if (status_code == 200):
            response_data = response.json()
            print(f"API Response ({endpoint}): {response.json()}")
            return response.json()
        else:
            print("API FAILURE DETECTED")
            print("STOP THE PROGRAM NOW.")
            print("YOU HAVE 20 SECONDS")
            time.sleep(20)
            return {}
    except requests.exceptions.RequestException as e:
        print(f"CRITICAL: Network/Connection failure for {endpoint}. Reason: {e}")
        print("STOP THE PROGRAM NOW.")
        print("YOU HAVE 20 SECONDS")
        time.sleep(20)
        return {}



#-------------------------------------------THREADING FUNCTIONS-----------------------------------------------------

def getPhoneData(stop_event, lock, variables, phone):
    while not stop_event.is_set():
        phone_pose = phone.get_latest_data()
        if phone_pose is not None:
            with lock:
                variables['new_phone_position'] = np.asarray(phone_pose['position'])
                variables['phone_toggle'] = phone_pose['toggle']
                variables['phone_button'] = phone_pose['button']
        #CHANGE THIS VALUE TO ADJUST THE HZ
        time.sleep(0.01)

#---------------------------------------------CREATING THE WORLD--------------------------------------------------

def ekf_filter(position_measurement, dt, ekf_state, ekf_cov):
    """
    Filters 3D position using EKF with constant velocity model.

    Args:
        position_measurement (np.array): shape (3,), new position [x, y, z]
        dt (float): timestep in seconds (e.g., 1/500)
        ekf_state (np.array): shape (6, 1), current state [x, y, z, vx, vy, vz]
        ekf_cov (np.array): shape (6, 6), current covariance matrix

    Returns:
        filtered_position (np.array): shape (3,)
        new_ekf_state (np.array): shape (6, 1)
        new_ekf_cov (np.array): shape (6, 6)
    """
    # Predict
    F = np.eye(6)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt

    Q = np.eye(6) * 1e-4  # process noise
    pred_state = F @ ekf_state
    pred_cov = F @ ekf_cov @ F.T + Q

    # Measurement update
    H = np.zeros((3, 6))
    H[0, 0] = H[1, 1] = H[2, 2] = 1  # measuring only position

    R = np.eye(3) * 1e-2  # measurement noise
    z = position_measurement.reshape((3, 1))

    y = z - H @ pred_state
    S = H @ pred_cov @ H.T + R
    K = pred_cov @ H.T @ np.linalg.inv(S)

    updated_state = pred_state + K @ y
    updated_cov = (np.eye(6) - K @ H) @ pred_cov

    return updated_state[:3].flatten(), updated_state, updated_cov

#------------------------------------------CONNECTING TO SO-101 ROBOT-------------------------------------------------
print("Turning on gravity compensation")
call_to_api("/gravity/start", {})
time.sleep(2)

print("Initializing robot")
call_to_api("move/init")
time.sleep(2)

print("Reading Home Position")
pose_dict = call_to_api("/end-effector/read", {})
POSE_KEYS = ['x', 'y', 'z']
pose_list = [pose_dict[key] for key in POSE_KEYS]
home_position = np.array(pose_list)

prev_position = home_position

print("Initial End Effector Position: ", prev_position)

joints_dict = call_to_api("/joints/read", {})
joints_array = np.array(joints_dict['angles'])

SCALE = 0.5
variables = {"new_phone_position": None, "phone_toggle": False, "phone_button": False}
state_lock = threading.Lock()
stop_event = threading.Event()

#Extended Kalman Filter variables
ekf_state = np.zeros((6, 1))         # Initial position and velocity
ekf_cov = np.eye(6) * 1e-3           # Initial uncertainty
last_ekf_time = time.time()

control_mode = "TELEOP"

#--------------------------------------------CONNECTING TO PHONE-----------------------------------------------------

phone = MujocoARConnector(port=8888, debug=False)
phone.start()
print("Waiting for device to connect...")
while not phone.connected_clients:
    time.sleep(0.050)
print("MujocoAR Initialized")
time.sleep(5)

#------------------------------------------------THREADING-----------------------------------------------------------

phone_thread = threading.Thread(target = getPhoneData, args=(stop_event, state_lock, variables, phone))
phone_thread.start()

#------------------------------------INITIALIZING FIRST PHONE VARIABLES----------------------------------------------

while True:
    with state_lock:
        if variables['new_phone_position'] is not None:
            prev_phone_position = variables['new_phone_position']
            break
    time.sleep(0.1)

prev_toggle_state = False

try:
    while not stop_event.is_set():
        #1. Read the new phone data
        with state_lock:
            current_phone_position = variables["new_phone_position"].copy()
            current_toggle = variables['phone_toggle']
            current_button = variables['phone_button']

        if current_button and control_mode == "TELEOP":
            print("Switching to HOMING mode...")
            control_mode = "HOMING"

        if control_mode == "HOMING":
            call_to_api("joints/write", {"angles": joints_array.tolist()})
            pose_dict = call_to_api("/end-effector/read", {"sync": True})
            if 'x' in pose_dict:
                POSE_KEYS = ['x', 'y', 'z']
                actual_position_array = np.array([pose_dict[key] for key in POSE_KEYS])

                distance_to_home = np.linalg.norm(home_position - actual_position_array)
                print(f"Distance to Home: {distance_to_home:.4f}")
            
                if distance_to_home < 0.015: 
                    print("Homing complete. Switching back to TELEOPERATION mode.")
                    control_mode = "TELEOP"
                    prev_phone_position = current_phone_position
                    prev_position = actual_position_array
            else:
                print("FAILED TO READ END EFFECTOR POSITION")
        
        elif control_mode == "TELEOP":
            #2. Calculate position and rotation deltas
            pos_delta_raw = current_phone_position - prev_phone_position

            #3. Filter position and rotation delta
            current_time = time.time()
            dt = max(current_time - last_ekf_time, 1e-6)
            last_ekf_time = current_time
            pos_delta_filtered, ekf_state, ekf_cov = ekf_filter(pos_delta_raw, dt, ekf_state, ekf_cov)

            #4. Scale position and rotation deltas
            pos_delta_scaled = pos_delta_filtered * SCALE

            #5. Swap and invert positions and rotations
            pos_delta_swapped = np.array([pos_delta_scaled[0], -pos_delta_scaled[1], pos_delta_scaled[2]])

            # 6. Add to old tcp position and rotation
            final_target_position = prev_position + pos_delta_swapped

            #7. Command Robots
            call_to_api("move/absolute", {"x": final_target_position[0], "y": final_target_position[1], "z": final_target_position[2]})


            #8. Update variables
            prev_phone_position = current_phone_position

            actual_pose_dict = call_to_api("/end-effector/read", {})
            if 'x' in actual_pose_dict:
                actual_position_array = np.array([actual_pose_dict[key] for key in POSE_KEYS])
                prev_position = actual_position_array
            else:
                print("Warning: Failed to read pose after movement.")

            #9. Gripper Control via Phone Toggle
            if current_toggle != prev_toggle_state:
                if current_toggle:
                    print("CLOSING GRIPPER")
                    call_to_api("move/absolute", {"open": 0})
                else:
                    print("OPENING GRIPPER")
                    call_to_api("move/absolute", {"open": 1})
                    
                prev_toggle_state = current_toggle

except KeyboardInterrupt:
    print("KEYBOARD INTERRUPT DETECTED")

finally:
    print("Shutting down...")
    print("Threads ending their loops...")
    stop_event.set()
    print("Phone thread rejoining...")
    phone_thread.join()
    print("All threads stopped successfully")
    print("Robot stopping movement...")
    call_to_api("move/sleep", {})
    print("Program Stopped")