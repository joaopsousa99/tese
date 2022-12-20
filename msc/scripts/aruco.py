#! /usr/bin/python3
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError


class ArUcoDetector:
  def __init__(self):
    self.sub = rospy.Subscriber("pylon_camera_node/image_rect", Image, self.callback)
    self.pub = rospy.Publisher("result", Image, queue_size=10)

    self.bridge = CvBridge()

    self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    self.arucoParams = cv2.aruco.DetectorParameters_create()
    
  def ros2cv(self, ros_img_msg, cv_format):
        try:
            return self.bridge.imgmsg_to_cv2(ros_img_msg, cv_format)
        except CvBridgeError as e:
            print(e)

  def callback(self, img):
    img = self.ros2cv(img, "bayer_gbrg8")
    img = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)

    (corners, ids, rejected) = cv2.aruco.detectMarkers(img, self.arucoDict, parameters=self.arucoParams)
    if len(corners) > 0:
      # flatten the ArUco IDs list
      ids = ids.flatten()

      # loop over the detected ArUCo corners
      for (markerCorner, markerID) in zip(corners, ids):
        # extract the marker corners (which are always returned in
        # top-left, top-right, bottom-right, and bottom-left order)
        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners

        # convert each of the (x, y)-coordinate pairs to integers
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))

        # draw the bounding box of the ArUCo detection
        cv2.line(img, topLeft, topRight, (0, 255, 0), 2)
        cv2.line(img, topRight, bottomRight, (0, 255, 0), 2)
        cv2.line(img, bottomRight, bottomLeft, (0, 255, 0), 2)
        cv2.line(img, bottomLeft, topLeft, (0, 255, 0), 2)

        # compute and draw the center (x, y)-coordinates of the ArUco
        # marker
        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        cv2.circle(img, (cX, cY), 4, (0, 0, 255), -1)

        # draw the ArUco marker ID on the img
        cv2.putText(img, str(markerID),
          (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
          0.5, (0, 255, 0), 2)

        print("[INFO] ArUco marker ID: {}".format(markerID))
        
    # show the output img
    self.pub.publish(self.bridge.cv2_to_imgmsg(img, encoding="passthrough"))
    cv2.imwrite("/home/jp/Desktop/asdf.jpg", img)
        


def main():
  rospy.init_node("landing_controller", anonymous=True)
  # rate = rospy.Rate(3)

  _ = ArUcoDetector()

  while not rospy.is_shutdown():
    rospy.spin()

if __name__ == "__main__":
  main()

