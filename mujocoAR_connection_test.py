#pip install mujoco_ar

import time
from mujoco_ar import MujocoARConnector

#Creates a MujocoARConnector object
connector = MujocoARConnector(port=8888, debug=True)
connector.start()

#Establishes a connection to the device
print("Waiting for device to connect...")
while not connector.connected_clients:
    time.sleep(0.1)

time.sleep(5)
#Gathers and prints device's positional and
#rotational data in 3D spcae at 1 sec intervals
while True:
    data = connector.get_latest_data()
    if data:
        print(data)
