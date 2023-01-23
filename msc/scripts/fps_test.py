#! /usr/bin/python3
import rospy
from sensor_msgs.msg import Image
import time


global old 
old = time.time()

class FPS:
  def __init__(self):
    self.sub = rospy.Subscriber("pylon_camera_node/image_rect", Image, self.callback)

  def callback(self, img):
    global old
    # img = self.ros2cv(img, "bayer_gbrg8")
    # img = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
    # img = cv2.merge((img, img, img))
    now = time.time()
    print(f"{1/(now-old)}")
    old = time.time()
    # img = cv2.putText(img, f"{rospy.get_time()}", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    # self.pub.publish(self.bridge.cv2_to_imgmsg(img, encoding="passthrough"))


if __name__ == "__main__":
  rospy.init_node("fps_tester", anonymous=True)
  # rate = rospy.Rate(3)

  _ = FPS()

  while not rospy.is_shutdown():
    rospy.spin()