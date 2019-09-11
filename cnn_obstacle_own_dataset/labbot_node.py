import os
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

import keras
import tensorflow as tf
from keras.models import model_from_json

img_width = 160
img_height = 120
json_model_name = './training/train_v108/train_v108_model.json'
h5_model_name = './training/train_v108/train_v108_model.h5'

node_name = 'robot_moving_node'
camera_topic_name = '/camera/depth/image_raw'
speed_topic_name = '/cmd_joy'

control = {
    0: -1,
    1: 0,
    2: 1
}
control_info = {
    0: 'Left',
    1: 'Straightforward',
    2: 'Right'
}

def load_model(model_json, model_weights):
    global model
    json_file = open(model_json, 'r')
    model_json = json_file.read()
    model = model_from_json(model_json, custom_objects = \
        {"GlorotUniform": tf.keras.initializers.glorot_uniform})
    model.load_weights(model_weights)

def callback(data):
    global image_data
    image_data = data

def ros_initialize():
    # Init ROS node
    rospy.init_node(node_name, anonymous=True)
    # Subscribe depth image topic
    rospy.Subscriber(camera_topic_name, Image, callback)
    # Publish twist data to robot
    return rospy.Publisher(speed_topic_name, Twist, queue_size=10) 

def process_depth_image(bridge):
    image = bridge.imgmsg_to_cv2(image_data, desired_encoding="passthrough")
    image = cv2.resize(image, (img_width, img_height))
    nn_input = image.reshape(1, img_height, img_width, 1)
    return nn_input

def robot_node():
    # Load trained Neural Network
    load_model(json_model_name, h5_model_name)

    vel_pub = ros_initialize()
    rate = rospy.Rate(10)  # 10hz
    rospy.sleep(1)
    bridge = CvBridge()

    while not rospy.is_shutdown():
		# Predict control using the Neural Network	
        image_input = process_depth_image(bridge)
        prediction = model.predict(image_input)
        control = np.argmax(prediction)
        speed = control.get(control)
        print(control_info(control))

        # Publish control to robot
        vel_msg = Twist()
        vel_msg.linear.x = 0.25
        vel_msg.angular.z = 0.4 * speed
        pub.publish(vel_msg)

        rate.sleep()


if __name__ == '__main__':
    try:
        robot_node()
    except rospy.ROSInterruptException:
        pass

