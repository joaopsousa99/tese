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
    self.pub = rospy.Publisher("aruco/result", Image, queue_size=10)

    self.bridge = CvBridge()

    self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    # self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
    self.arucoParams = cv2.aruco.DetectorParameters_create()

    with open('/home/jp/catkin_ws/src/pylon-ros-camera/pylon_camera/calibration/pylon_camera1.yaml') as f:
      data = yaml.load(f, Loader=SafeLoader)
      self.cameraMatrix = data["camera_matrix"]["data"]
      self.cameraMatrix = np.reshape(np.array(self.cameraMatrix), (3,3))
      self.distortionCoefficients = data["distortion_coefficients"]["data"]
      self.distortionCoefficients = np.array(self.distortionCoefficients)
    
  def ros2cv(self, ros_img_msg, cv_format):
        try:
            return self.bridge.imgmsg_to_cv2(ros_img_msg, cv_format)
        except CvBridgeError as e:
            print(e)

  def callback(self, img):
    img = self.ros2cv(img, "mono8")
    img = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)

    (corners, ids, _) = cv2.aruco.detectMarkers(img, self.arucoDict, parameters=self.arucoParams)
    if len(corners) > 0:
      # flatten the ArUco IDs list
      ids = ids.flatten()

      for i in range(0, len(ids)):
        R, t, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.154, self.cameraMatrix, self.distortionCoefficients)
        rospy.loginfo(f"Marker detected. Corners: {corners[i][0]}")
        cv2.aruco.drawDetectedMarkers(img, corners)
        cv2.drawFrameAxes(img, self.cameraMatrix, self.distortionCoefficients, R, t, 0.05)
        x = 0.93605953573875675805*t[0][0][0]+0.35184164840490857136*t[0][0][1]+2.2048868272539550439
        z = -0.35184164840490857136*t[0][0][2]+0.93605953573875675805*t[0][0][1]+1.0001810231155480956
        # print(f'[{x:.3f}, {y:.3f}]')
        cv2.putText(img, str(ids[i]), (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    # show the output img
    self.pub.publish(self.bridge.cv2_to_imgmsg(img, encoding="passthrough"))
    # cv2.imwrite("/home/jp/Desktop/asdf.jpg", img)


if __name__ == "__main__":
  rospy.init_node("landing_controller", anonymous=True)
  # rate = rospy.Rate(3)

  _ = ArUcoDetector()

  while not rospy.is_shutdown():
    rospy.spin()