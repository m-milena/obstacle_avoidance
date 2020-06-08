#!/usr/bin/env python

import rospy
import rospkg
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

import keras
import tensorflow as tf
from keras.models import model_from_json

import math
import cv2
import numpy as np
from cv_bridge import CvBridge


#####################  VARIABLES  #####################################

node_name = 'robot_moving_node'
camera_topic_name = '/camera/depth/image_raw'
odometry_topic_name = '/labbot_odometry'
control_topic_name = '/cmd_joy'

rospack = rospkg.RosPack()
obstacle_json_model_name = rospack.get_path('labbot_neural_control') + \
        '/networks/obstacle_avoidance/train_v003_model.json'
obstacle_h5_model_name = rospack.get_path('labbot_neural_control') + \
        '/networks/obstacle_avoidance/train_v003_model.h5'
navigation_json_model_name = rospack.get_path('labbot_neural_control') + \
        '/networks/navigation/train_v001_model.json'
navigation_h5_model_name = rospack.get_path('labbot_neural_control') + \
        '/networks/navigation/train_v001_model.h5'

img_width = 160
img_height = 120

obstacle_control_switch = {
    0: [0.25, -0.2],
    1: [0.25, 0],
    2: [0.25, 0.2] 
}
obstacle_control_info = {
    0: 'Left',
    1: 'Straightforward',
    2: 'Right'
}

navigation_control_switch = {
    0: [0.0, 0.14],
    1: [0.2, 0.0],
    2: [0.0, -0.14],
    3: [0.0, 0.0]
}
navigation_control_info = {
    0: 'Right',
    1: 'Straightforward',
    2: 'Left',
    3: 'Stop'
}

navigation_control_history = {
    0: 2,
    1: 1,
    2: 0
}

########################  FUNCTIONS  ##################################

def load_model(model_json, model_weights):
    json_file = open(model_json, 'r')
    model_json = json_file.read()
    model = model_from_json(model_json, custom_objects = \
        {"GlorotUniform": tf.keras.initializers.glorot_uniform})
    model.load_weights(model_weights)
    return model

def odometry_callback(data):
    global labbot_odometry 
    labbot_odometry = data.pose.pose

def robot_yaw_angle(z, w):
    x = y = 0
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return math.degrees(yaw)

def calculate_target_angle(target_x, target_y):
    dx = target_x - labbot_odometry.position.x
    dy = target_y - labbot_odometry.position.y
    if dx and dy:
        if dx < 0 and dy < 0:
            angle = -(180 - math.degrees(math.atan(dy/dx)))
        else:
            angle = math.degrees(math.atan(dy/dx))
    else:
        angle = math.degrees(math.atan(dy/dx))
    return angle

def process_depth_image(bridge):
    image = bridge.imgmsg_to_cv2(image_data, desired_encoding="32FC1")
    image = (image/10000)*255
    image = cv2.resize(image, (img_width, img_height))
    model_input = image.reshape(1, img_height, img_width, 1)
    return model_input

def camera_callback(data):
    global image_data
    image_data = data

#####################  MAIN FUNCTION  #################################

def main():
    # Get target point x and y
    print('Input target point position.\nInput x:')
    target_x = float(input())
    print('Input y:')
    target_y = float(input())
    # ROS init
    rospy.init_node(node_name, anonymous=True)
    rate = rospy.Rate(10)  # 5hz
    rospy.sleep(1)
    # Publishers and Subscribers
    rospy.Subscriber(odometry_topic_name, Odometry, odometry_callback)
    rospy.Subscriber(camera_topic_name, Image, camera_callback)
    vel_pub = rospy.Publisher(control_topic_name, Twist, queue_size=10)
    # Loading CNN model
    obstacle_model = load_model(obstacle_json_model_name, obstacle_h5_model_name)
    navigation_model = load_model(navigation_json_model_name, navigation_h5_model_name)
    bridge = CvBridge()
    control_history = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    history_to_array = np.array(control_history)
    counter = 0
    # Main loop
    while not rospy.is_shutdown():
        # Calculate robot yaw and angle between robot orientation and target point
        labbot_angle = robot_yaw_angle(labbot_odometry.orientation.z, \
                    labbot_odometry.orientation.w)
        target_angle = calculate_target_angle(target_x, target_y)
        # Input of network: dx, dy, d_angle
        dx = target_x - labbot_odometry.position.x
        dy = target_y - labbot_odometry.position.y
        d_angle = target_angle - labbot_angle
        navigation_network_input = [[dx, dy, d_angle]]
        navigation_network_input = np.array(navigation_network_input) 
        # Predict robot control
        navigation_prediction = navigation_model.predict(navigation_network_input)
        navigation_control = np.argmax(navigation_prediction)   
        # Process image from camera
        image_input = process_depth_image(bridge)
        # Predict control
        obstacle_prediction = obstacle_model.predict([image_input, history_to_array])
        obstacle_control = np.argmax(obstacle_prediction)
        vel_msg = Twist()
        if navigation_control != 3:
            if (obstacle_control == 1 or navigation_control != 1) and not counter:
                vel_msg.linear.x, vel_msg.angular.z = navigation_control_switch.get(navigation_control)
                print('%.3f, %.3f ----- %.3f ----- %s' % (labbot_angle, \
                    target_angle, d_angle, navigation_control_info.get(navigation_control)))
                last_control = [0, 0, 0]
                last_control[navigation_control_history.get(navigation_control)] = 1
            else:
                vel_msg.linear.x, vel_msg.angular.z = obstacle_control_switch.get(obstacle_control)
                print('!!! Obstacle: ' + str(obstacle_control_info.get(obstacle_control)))
                last_control = [0, 0, 0]
                last_control[obstacle_control] = 1
                if counter > 30:
                    counter = 0
                else:
                    counter = counter + 1
            # Save last control to array

            control_history = [last_control + control_history[0][:-3]]
            history_to_array = np.array(control_history)
        # Publish control to robot
        vel_pub.publish(vel_msg)

        rate.sleep()

    vel_msg = Twist()
    vel_msg.linear.x = 0
    vel_msg.angular.z = 0
    vel_pub.publish(vel_msg)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass