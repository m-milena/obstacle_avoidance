#!/usr/bin/env python

import rospy
import rospkg
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

import keras
import tensorflow as tf
from keras.models import model_from_json

import cv2
import numpy as np
from cv_bridge import CvBridge


#####################  VARIABLES  #####################################

node_name = 'robot_moving_node'
camera_topic_name = '/camera/depth/image_raw'
control_topic_name = '/cmd_joy'

rospack = rospkg.RosPack()
json_model_name = rospack.get_path('labbot_neural_control') + \
        '/networks/obstacle_avoidance/train_v003_model.json'
h5_model_name = rospack.get_path('labbot_neural_control') + \
        '/networks/obstacle_avoidance/train_v003_model.h5'

img_width = 160
img_height = 120

control_switch = {
    0: -1,
    1: 0,
    2: 1 
}
control_info = {
    0: 'Left',
    1: 'Straightforward',
    2: 'Right'
}

########################  FUNCTIONS  ##################################

def load_model(model_json, model_weights):
    json_file = open(model_json, 'r')
    model_json = json_file.read()
    model = model_from_json(model_json, custom_objects = \
        {"GlorotUniform": tf.keras.initializers.glorot_uniform})
    model.load_weights(model_weights)
    return model

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
    # ROS init
    rospy.init_node(node_name, anonymous=True)
    rate = rospy.Rate(5)  # 5hz
    rospy.sleep(1)
    # Publishers and Subscribers
    rospy.Subscriber(camera_topic_name, Image, camera_callback)
    vel_pub = rospy.Publisher(control_topic_name, Twist, queue_size=10)
    # Loading CNN model
    model = load_model(json_model_name, h5_model_name)

    bridge = CvBridge()
    control_history = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    history_to_array = np.array(control_history)
    # Main loop
    while not rospy.is_shutdown():    
        # Process image from camera
        image_input = process_depth_image(bridge)
        # Predict control
        prediction = model.predict([image_input, history_to_array])
        control = np.argmax(prediction)
        speed = control_switch.get(control)
        print(control_info.get(control))
        # Save last control to array
        last_control = [0, 0, 0]
        last_control[control] = 1
        control_history = [last_control + control_history[0][:-3]]
        history_to_array = np.array(control_history)
        # Publish control to robot
        vel_msg = Twist()
        vel_msg.linear.x = 0.25
        vel_msg.angular.z = 0.4 * speed
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
