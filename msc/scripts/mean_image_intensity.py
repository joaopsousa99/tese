#! /usr/bin/python3
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import yaml
from yaml.loader import SafeLoader
import numpy as np


class ArUcoDetector:
  def __init__(self):
    self.sub = rospy.Subscriber("pylon_camera_node/image_rect", Image, self.callback)
    self.bridge = CvBridge()
    
  def ros2cv(self, ros_img_msg, cv_format):
    try:
        return self.bridge.imgmsg_to_cv2(ros_img_msg, cv_format)
    except CvBridgeError as e:
        print(e)

  def callback(self, img):
    img = self.ros2cv(img, "mono8")
    # img = cv2.cvtColor(img, cv2.COLOR_BayerGR2GRAY)
    print(np.mean(img))
    # print(img.shape)


if __name__ == "__main__":
  rospy.init_node("landing_controller", anonymous=True)
  # rate = rospy.Rate(3)

  _ = ArUcoDetector()

  while not rospy.is_shutdown():
    rospy.spin()