#! /usr/bin/env python3
# Import ROS.
import rospy
# Import the API.
from msc.py_gnc_functions import *
# To print colours (optional).
from msc.PrintColours import *

def main():
    # Initializing ROS node.
    rospy.init_node("drone_controller", anonymous=True)

    # Create an object for the API.
    drone = gnc_api()
    # Wait for FCU connection.
    drone.wait4connect()
    # Wait for the mode to be switched.
    drone.wait4start()

    # Create local reference frame.
    drone.initialize_local_frame()
    # Request takeoff with an altitude of 3m.
    drone.takeoff(3)
    # Specify control loop rate. We recommend a low frequency to not over load the FCU with messages. Too many messages will cause the drone to be sluggish.


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()
