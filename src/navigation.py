#!/usr/bin/env python

import math
import numpy as np

import rospy
import rospkg

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

import keras
import tensorflow as tf
from keras.models import model_from_json

#####################  VARIABLES  #####################################

node_name = 'robot_moving_node'
odometry_topic_name = '/labbot_odometry'
control_topic_name = '/cmd_joy'

rospack = rospkg.RosPack()
json_model_name = rospack.get_path('labbot_neural_control') + \
        '/networks/navigation/train_v001_model.json'
h5_model_name = rospack.get_path('labbot_neural_control') + \
        '/networks/navigation/train_v001_model.h5'

control_switch = {
    0: [0.0, 0.14],
    1: [0.2, 0.0],
    2: [0.0, -0.14],
    3: [0.0, 0.0]
}
control_info = {
    0: 'Right',
    1: 'Straightforward',
    2: 'Left',
    3: 'Stop'
}

########################  FUNCTIONS  ##################################

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
    if dx < 0 and dy < 0:
            angle = -(180 - math.degrees(math.atan(dy/dx)))
    elif dx < 0 and dy > 0:
            angle = 180 + math.degrees(math.atan(dy/dx))
    else:
        angle = math.degrees(math.atan(dy/dx))

    return angle

def load_model(model_json, model_weights):
    json_file = open(model_json, 'r')
    model_json = json_file.read()
    model = model_from_json(model_json, custom_objects = \
        {"GlorotUniform": tf.keras.initializers.glorot_uniform})
    model.load_weights(model_weights)
    return model

#####################  MAIN FUNCTION  #################################

def main():
    # Get target point x and y
    print('Input target point position.\nInput x:')
    target_x = float(input())
    print('Input y:')
    target_y = float(input())
    # ROS init
    rospy.init_node(node_name, anonymous=True)
    rate = rospy.Rate(5)  # 5hz
    rospy.sleep(1)
    # Publishers and Subscribers
    rospy.Subscriber(odometry_topic_name, Odometry, odometry_callback)
    vel_pub = rospy.Publisher(control_topic_name, Twist, queue_size=10)
    # Loading CNN model
    model = load_model(json_model_name, h5_model_name)
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
        if d_angle > 180:
            d_angle = -180 + (d_angle - 180)
        elif d_angle < -180:
            d_angle = 180 + (d_angle + 180)
        network_input = [[dx, dy, d_angle]]
        network_input = np.array(network_input)
        # Predict robot control
        prediction = model.predict(network_input)
        control = np.argmax(prediction)
        print('%.3f, %.3f ----- %3.f ----- %s' % (labbot_angle, \
                target_angle, d_angle, control_info.get(control)))
        # Publish control to robot
        vel_msg = Twist()
        vel_msg.linear.x, vel_msg.angular.z = control_switch.get(control)
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
