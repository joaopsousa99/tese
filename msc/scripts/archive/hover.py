#! /usr/bin/python3
from __future__ import print_function
import rospy
from msc.py_gnc_functions import *
from msc.PrintColours import *
from geometry_msgs.msg import Point, TwistStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import Image
from scipy.spatial.distance import euclidean


class landingController:
  def __init__(self):
    self.distanceToCentreThreshold = 5

    # parâmetros do PID
    self.px = 0.001
    self.py = 0.001

    self.height = None
    self.width = None
    self.deltaXENU = None
    self.deltaYENU = None

  def setup(self):
    initialAltitude = 5
    self.drone = gnc_api()
    self.drone.wait4connect()
    self.drone.wait4start()
    self.drone.initialize_local_frame()
    self.drone.takeoff(initialAltitude)
    self.drone.set_destination(x=0, y=0, z=initialAltitude, psi=0)

    self.imageSub = rospy.Subscriber("pylon_camera_node/image_raw", Image, self.imgCallback)
    self.targetSub = rospy.Subscriber("target/centre", Point, self.targetCallback)

    self.pub = rospy.Publisher("mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=10)

  def targetCallback(self, data):
    # para verificar se recebeu bem os dados
    invalidX = data.x is None
    invalidY = data.y is None
    invalidHeight = self.height is None
    invalidWidth = self.width is None

    if invalidX or invalidY or invalidHeight or invalidWidth:
      self.deltaXENU = 0
      self.deltaYENU = 0

    else:
      imgCentre = (self.width/2, self.height/2)
      targetCentre = (data.x, data.y)

      centreDist = euclidean(imgCentre, targetCentre)

      # está alinhado com o alvo
      if centreDist < self.distanceToCentreThreshold:
        self.deltaXENU = 0
        self.deltaYENU = 0
      else:
        # no opencv, x=w e y=h
        self.deltaXENU = -(self.width/2 - data.x)
        self.deltaYENU =  self.height/2 - data.y

    velCmd = TwistStamped()
    velCmd.header.stamp = rospy.Time.now()
    velCmd.twist.linear.x = self.px*self.deltaXENU
    velCmd.twist.linear.y = self.py*self.deltaYENU
    velCmd.twist.linear.z = 0
    self.pub.publish(velCmd)

  def imgCallback(self, data):
    self.height = data.height
    self.width = data.width


def main():
  rospy.init_node("landing_controller", anonymous=True)
  rate = rospy.Rate(3)

  hc = landingController()
  hc.setup()

  while not rospy.is_shutdown():
    try:
      rospy.spin()
      rate.sleep()
    except rospy.ROSInterruptException:
      print("shutting down")


if __name__ == "__main__":
  try:
    main()
  except rospy.ROSInterruptException:
    exit()
