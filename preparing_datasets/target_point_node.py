import rospy 
from geometry_msgs.msg import PoseStamped
import math

target_point_topic_name = "/target_point_pose"
pub = rospy.Publisher(target_point_topic_name, PoseStamped, queue_size=10)

print("Input position in x axis")
target_pose_x = input()
print("Input position in y axis")
target_pose_y = input()

rospy.init_node("target_point")
rate = rospy.Rate(10)

from tf.transformations import *

q_origin = quaternion_from_euler(0, 0, 0)
q_rot = quaternion_from_euler(0, 0, 0)
q_new = quaternion_multiply(q_rot, q_origin)

target_point_msg = PoseStamped()
target_point_msg.pose.position.x = target_pose_x
target_point_msg.pose.position.y = target_pose_y
target_point_msg.pose.position.z = 0
target_point_msg.pose.orientation.x = q_new[0]
target_point_msg.pose.orientation.y = q_new[1]
target_point_msg.pose.orientation.z = q_new[2]
target_point_msg.pose.orientation.w = q_new[3]
target_point_msg.header.frame_id = "map"
target_point_msg.header.stamp = rospy.get_rostime()

def pub_target_point():
    while not rospy.is_shutdown():
        pub.publish(target_point_msg)
        print("-"*15)
        print(target_point_msg)
    rate.sleep()

if __name__ =='__main__':
    try:
        pub_target_point()
    except rospy.ROSInterruptException:
        pass
