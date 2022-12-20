#! /usr/bin/python3
import rospy
from msc.py_gnc_functions import *
from msc.PrintColours import *
from std_msgs.msg import Bool

def takeoff():
    nh = rospy.Publisher("takeoff", Bool, queue_size=10)
    rospy.init_node("takeoff", anonymous=True)
    rate = rospy.Rate(3) # 3Hz

    drone = gnc_api()
    drone.wait4connect()
    drone.set_mode("GUIDED")
    drone.wait4start()
    drone.initialize_local_frame()
    drone.takeoff(3)

    takeoff_complete = Bool()

    while not rospy.is_shutdown():
        # não posso usar o check do waypoint porque isso faz o drone aterrar
        takeoff_complete.data = drone.check_waypoint_reached()
        drone.set_destination(x=0, y=0, z=3, psi=0)
        nh.publish(takeoff_complete)
        rate.sleep()

if __name__ == '__main__':
    try:
        takeoff()
    except rospy.ROSInterruptException:
        pass