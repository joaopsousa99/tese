#! /usr/bin/python3
from __future__ import print_function
from concurrent.futures import process
import rospy
from iq_gnc.py_gnc_functions import *
from iq_gnc.PrintColours import *
from geometry_msgs.msg import Point, TwistStamped, PoseStamped
# from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from scipy.spatial.distance import euclidean
import numpy as np


class landingController:
  def __init__(self):
    self.imageSub = rospy.Subscriber("pylon_camera_node/image_raw", Image, self.imgCallback)
    self.targetSub = rospy.Subscriber("target/center", Point, self.targetCallback)
    # self.poseSub = rospy.Subscriber("mavros/global_position/local", Odometry, self.poseCallback)
    self.poseSub = rospy.Subscriber("mavros/local_position/pose", PoseStamped, self.poseCallback)

    self.pub = rospy.Publisher("mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=10)
 
    self.MAX_DIST_TO_CENTER_THRESH = 250
    self.MIN_DIST_TO_CENTER_THRESH = 25
    self.DISTANCE_TO_CENTER_THRESH = self.MIN_DIST_TO_CENTER_THRESH

    # PID
    #   constantes calculadas com o método de ziegler-nichols
    #   Ku = 2.5e-6
    #   Tux = 8.7037/3
    #   Tuy = 8.4286/3
    self.KPX = 0.8e-6
    self.KPY = 0.8e-6
    self.KPZ = -5e-4

    self.KIX = 1e-7
    self.KIY = 1e-7
    self.KIZ = 0#1e-12

    self.KDX = 0#1e-7
    self.KDY = 0#1e-7
    self.KDZ = 0#1e-9


    self.differentialErrorX = 0.0
    self.differentialErrorY = 0.0
    self.differentialErrorZ = 0.0

    self.integralErrorX = 0
    self.integralErrorY = 0
    self.integralErrorZ = 0

    self.IMG_HEIGHT = None
    self.IMG_WIDTH = None

    # cm/s
    self.MIN_VEL_X     = -50.0
    self.MAX_VEL_X     =  50.0
    self.MIN_VEL_Y     = -50.0
    self.MAX_VEL_Y     =  50.0
    self.MIN_ABS_VEL_Z =  0.001
    self.MAX_ABS_VEL_Z =  50.0

    self.startedLanding = False

    self.relAlt = 0.0 # em centímetros

    self.landingAlt = 20.0 # em centímetros

  def setup(self):
    self.drone = gnc_api()
    self.drone.wait4connect()
    self.drone.initialize_local_frame()
    self.drone.wait4start()
    
  def targetCallback(self, data):
    # para verificar se recebeu bem os dados
    invalidX = data.x is None
    invalidY = data.y is None
    invalidZ = data.z is None or data.z < 0
    invalidHeight = self.IMG_HEIGHT is None
    invalidWidth = self.IMG_WIDTH is None

    if invalidX or invalidY or invalidZ or invalidHeight or invalidWidth:
      return

    else:
      imgCenter = (self.IMG_WIDTH/2, self.IMG_HEIGHT/2)
      targetCenter = (data.x, data.y)

      centerDist = euclidean(imgCenter, targetCenter)
      if centerDist <= self.DISTANCE_TO_CENTER_THRESH:
        centerDist = 0

      # está alinhado com o alvo
      if centerDist < self.DISTANCE_TO_CENTER_THRESH:
        deltaZENU = self.relAlt
        if self.relAlt < self.landingAlt:
          self.drone.land()
      else:
        deltaZENU = 0

      # o OpenCV usa ESD (east-south-down)
      # o MAVLink usa ENU (east-north-up)
      deltaXENU = -(self.IMG_WIDTH/2 - data.x)
      deltaYENU =  self.IMG_HEIGHT/2 - data.y

      self.differentialErrorX = deltaXENU - self.differentialErrorX
      self.differentialErrorY = deltaYENU - self.differentialErrorY
      self.differentialErrorZ = deltaZENU - self.differentialErrorZ

      self.integralErrorX = self.integralErrorX + deltaXENU
      self.integralErrorY = self.integralErrorY + deltaYENU
      self.integralErrorZ = self.integralErrorZ + deltaZENU

      velX = self.KPX*deltaXENU + self.KIX*self.integralErrorX + self.KDX*self.differentialErrorX
      velY = self.KPY*deltaYENU + self.KIY*self.integralErrorY + self.KDY*self.differentialErrorY
      velZ = self.KPZ*deltaZENU + self.KIZ*self.integralErrorZ + self.KDZ*self.differentialErrorZ

      # print("-"*15 + f"\n\t{deltaXENU:.2f}\n\t{self.integralErrorX:.2f}\n\t{self.differentialErrorX:.2f}\n\t{deltaYENU:.2f}\n\t{self.integralErrorY:.2f}\n\t{self.differentialErrorY:.2f}\n\t{deltaZENU:.2f}\n\t{self.integralErrorZ:.2f}\n\t{self.differentialErrorZ:.2f}")

      velX = clamp(velX*self.relAlt, self.MIN_VEL_X, self.MAX_VEL_X)
      velY = clamp(velY*self.relAlt, self.MIN_VEL_Y, self.MAX_VEL_Y)
      velZ = np.sign(velZ)*clamp(abs(velZ), self.MIN_ABS_VEL_Z, self.MAX_ABS_VEL_Z)
      # if velZ == 0.001:
      #   print(f"{velZ} = {np.sign(velZ)}clamp({self.KPZ*deltaZENU} + {self.KIZ*self.integralErrorZ} + {self.KDZ*self.differentialErrorZ})")

      velCmd = TwistStamped()
      velCmd.header.stamp = rospy.Time.now()
      velCmd.twist.linear.x = velX
      velCmd.twist.linear.y = velY
      velCmd.twist.linear.z = velZ

      self.pub.publish(velCmd)

  def pid(self, pe, ie, de, P, I, D):
    output = P*pe + I*ie + D*de

    clampedOutput = clamp(output, -1e9, 1e9)

    if output != clampedOutput and np.sign(ie) == np.sign():
      return 0

    return clampedOutput

  def imgCallback(self, data):
    self.IMG_HEIGHT = data.height
    self.IMG_WIDTH = data.width

  def poseCallback(self, data):
    # self.relAlt = data.pose.pose.position.z*100 # em centímetros
    self.relAlt = data.pose.position.z*100 # em centímetros
    self.DISTANCE_TO_CENTER_THRESH = max(min(-0.307692307692*self.relAlt + 265.384615385, self.MAX_DIST_TO_CENTER_THRESH), self.MIN_DIST_TO_CENTER_THRESH)


def clamp(x, minimum, maximum):
  return max(min(x, maximum), minimum) if x != 0 else 0


def main():
  rospy.init_node("landing_controller", anonymous=True)
  # rate = rospy.Rate(3)

  hc = landingController()
  hc.setup()

  while not rospy.is_shutdown():
    rospy.spin()

if __name__ == "__main__":
  main()

