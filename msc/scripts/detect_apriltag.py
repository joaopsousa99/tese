#! /usr/bin/python3
from __future__ import print_function
import cv2
import numpy as np
import rospy
from cv_bridge import (CvBridge, CvBridgeError)
from geometry_msgs.msg import Point, PoseStamped
from sensor_msgs.msg import Image
import apriltag
import yaml
from yaml.loader import SafeLoader
from mavros_msgs import StatusText

# hack que encontrei no youtube que salta mensagens
# enquanto não terminar de processar a atual
processing = False
new_msg = False
msg = None


class AprilTagDetector:
	def __init__(self):
		self.bridge = CvBridge()

		# ROS
		self.imageSub = rospy.Subscriber("pylon_camera_node/image_rect", Image, self.callback)
		self.poseSub = rospy.Subscriber("mavros/local_position/pose", PoseStamped, self.poseCallback)

		self.tagCentrePub = rospy.Publisher("apriltag/centre", Point, queue_size=1)
		self.debugPub = rospy.Publisher("apriltag/debug", Image, queue_size=1)
		self.status = rospy.Publisher("mavros/statustext/send", StatusText, queue_size=1)

		# AprilTag
		apriltagOptions = apriltag.DetectorOptions(families="tag16h5")
		self.detector = apriltag.Detector(apriltagOptions)

		# Dados da calibração da câmara
		with open('/home/jp/catkin_ws/src/pylon-ros-camera/pylon_camera/calibration/pylon_camera_2.yaml') as f:
			calibrationData = yaml.load(f, Loader=SafeLoader)

			self.cameraMatrix = calibrationData["camera_matrix"]["data"]
			self.cameraMatrix = np.reshape(np.array(self.cameraMatrix), (3,3))

			self.distortionCoefficients = calibrationData["distortion_coefficients"]["data"]
			self.distortionCoefficients = np.array(self.distortionCoefficients)

		tagSizeMeters = 0.15341
		self.tagCorners = np.array([[-tagSizeMeters/2,  tagSizeMeters/2, 0],
				      									[ tagSizeMeters/2,  tagSizeMeters/2, 0],
																[ tagSizeMeters/2, -tagSizeMeters/2, 0],
																[-tagSizeMeters/2, -tagSizeMeters/2, 0]], dtype=float)

		self.relativeAltitude = 0.0  # em metros

	def ros2cv(self, ros_img_msg, cv_format):
		try:
			return self.bridge.imgmsg_to_cv2(ros_img_msg, cv_format)
		except CvBridgeError as e:
			print(e)

	def callback(self, data):
		global processing, new_msg, msg

		if not processing:
			new_msg = True
			msg = data

	def msg_processing(self, msg):
		img = self.ros2cv(msg, "mono8")

		if img is None:
			print("ERROR: img is None")
			self.tagCentrePub.publish(Point(0, 0, -1))
			return

		img = cv2.medianBlur(img, 5)
		results = self.detector.detect(img)

		img = cv2.merge((img, img, img))

		for r in results:
			corners = r.corners
			tagId = r.tag_id

			if len(corners) > 0:
				if tagId == 0:
					(_, rpy, t) = cv2.solvePnP(self.tagCorners, corners,
					                           self.cameraMatrix,
																		 self.distortionCoefficients,
																		 flags=cv2.SOLVEPNP_IPPE_SQUARE)

					# rpy = np.ravel(rpy, (3,1))
					# t = np.ravel(t, (3,1))

					c1 = (corners[0][0].astype(int), corners[0][1].astype(int))
					c2 = (corners[1][0].astype(int), corners[1][1].astype(int))
					c3 = (corners[2][0].astype(int), corners[2][1].astype(int))
					c4 = (corners[3][0].astype(int), corners[3][1].astype(int))

					tagCentre = (((c1[0] + c2[0] + c3[0] + c4[0])/4).astype(int),
												((c1[1] + c2[1] + c3[1] + c4[1])/4).astype(int))

					img = cv2.line(img, (0, tagCentre[1]), (2000, tagCentre[1]), (255,0,0), 3)
					img = cv2.line(img, (tagCentre[0], 0), (tagCentre[0], 2000), (255,0,0), 3)

					tagCentrePoint = Point(tagCentre[0], tagCentre[1], self.relativeAltitude)

					self.debugPub.publish(self.bridge.cv2_to_imgmsg(img, encoding="passthrough"))
					self.tagCentrePub.publish(tagCentrePoint)

					break

	def poseCallback(self, data):
		self.relativeAltitude = data.pose.position.z


def main():
	global processing, new_msg, msg

	rospy.init_node("apriltag", anonymous=True)
	rate = rospy.Rate(30)
	td = AprilTagDetector()

	while not rospy.is_shutdown():
		if new_msg:
			processing = True
			new_msg = False
			td.msg_processing(msg)
			rate.sleep()
			processing = False


if __name__ == '__main__':
	main()