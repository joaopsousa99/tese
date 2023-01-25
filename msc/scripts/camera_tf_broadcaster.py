#!/usr/bin/env python3
import rospy
import tf2_ros
import tf2_msgs.msg
import geometry_msgs.msg


class FixedTFBroadcaster:
    def __init__(self):
        self.pub_tf = rospy.Publisher("/camera_tf", tf2_msgs.msg.TFMessage, queue_size=1)

        while not rospy.is_shutdown():
            rospy.sleep(1/3)

            t = geometry_msgs.msg.TransformStamped()
            t.header.frame_id = "base_link"
            t.header.stamp = rospy.Time.now()
            t.child_frame_id = "camera_link"
            t.transform.translation.x = 0
            t.transform.translation.y = 0
            t.transform.translation.z = 0#0.03

            t.transform.rotation.w = 0
            t.transform.rotation.x = 1
            t.transform.rotation.y = 0
            t.transform.rotation.z = 0

            tfm = tf2_msgs.msg.TFMessage([t])
            self.pub_tf.publish(tfm)

if __name__ == '__main__':
    rospy.init_node('camera_tf_broadcaster')
    tfb = FixedTFBroadcaster()
    rospy.spin()