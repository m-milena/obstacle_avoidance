import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

import os
from cv_bridge import CvBridge, CvBridgeError
import cv2

global target_point_data
target_point_data = None
global labbot_odometry_data
labbot_odometry_data = None
global control_data
control_data = None
global depth_img_data
depth_img_data = None

def target_callback(data):
    global target_point_data
    target_point_data = data

def labbot_odometry_callback(data):
    global labbot_odometry_data
    labbot_odometry_data = data

def control_callback(data):
    global control_data
    control_data = data

def depth_img_callback(data):
    global depth_img_data
    depth_img_data = data


def subscribers():
    depth_img_topic_name = "/Throttle_camera_depth_img_raw"
    rospy.Subscriber(depth_img_topic_name, Image, depth_img_callback)
    target_topic_name = "/Throttle_target_point_pose"
    rospy.Subscriber(target_topic_name, PoseStamped, target_callback)
    labbot_odometry_topic_name = "/Throttle_labbot_odometry"
    rospy.Subscriber(labbot_odometry_topic_name, Odometry, labbot_odometry_callback)
    control_topic_name = "/Throttle_twist"
    rospy.Subscriber(control_topic_name, Twist, control_callback)

def calculate_labbot_position():
    return [labbot_odometry_data.pose.pose.position.x, labbot_odometry_data.pose.pose.position.y, labbot_odometry_data.pose.pose.orientation.z, labbot_odometry_data.pose.pose.orientation.w, target_point_data.pose.position.x, target_point_data.pose.position.y]

def main():
    node_name = 'collect_data_node'
    rospy.init_node(node_name, anonymous=True)
    rate = rospy.Rate(2)
    subscribers()
	
    current_img_number = 0
    img_folder = './dataset_v3/'
    img_files = [f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))]
    if img_files:
        img_numbers = [img[:5] for img in img_files]
        last_img_number = sorted(img_numbers)
        current_img_number = int(last_img_number[-1]) + 1

    bridge = CvBridge()
    while not rospy.is_shutdown():
        if control_data != None:
            pose_info = 'pose_x%.2f_pose_y%.2f_orient_z%.2f_orient_w%.2f_point_x%.2f_point_y%.2f' % tuple(calculate_labbot_position())
            control_info = 'lin_x%.2f_ang_z%.2f' % (control_data.linear.x, control_data.angular.z)
            image = bridge.imgmsg_to_cv2(depth_img_data, "32FC1")
            image = (image/10000)*255
            cv2.imwrite('%(folder)s%(#)05d_%(pose)s_%(control)s.png'%{'folder': img_folder, 'pose': pose_info, 'control': control_info, '#':current_img_number}, image)
            current_img_number = current_img_number + 1
        else:
            print('Waiting ...')
        rate.sleep()

if __name__ == '__main__':
    main()

