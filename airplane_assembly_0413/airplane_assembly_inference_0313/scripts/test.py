#!/usr/bin/env python
import rospy

import roslib; roslib.load_manifest('airplane_assembly_0313')
import tf

from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import MultiArrayDimension

import array
import numpy

import socket
import time
import sys
import pylab
import struct

from geometry_msgs.msg import *
pylab.ion()

from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from ar_track_alvar.msg import *

USE_ROS_RATE        = False

ROS_TOPIC_LEFTHAND  = "/left_hand"
ROS_TOPIC_RIGHTHAND = "/right_hand"
ROS_TOPIC_BINMARKES = "/ar_pose_marker"
ROS_TOPIC_IMAGE     = "/kinect/color_image"

TF_WORLD     = "base_link"
TF_KINECT    = "kinect0_rgb_optical_frame"
TF_WEBCAM    = "lifecam1_optical_frame"

BIN_NUM      = 20

T            = 900

FPS          = 30

MAX_NAME_LENGTH = 20

PORT = 12341





v = Float32MultiArray()
v.layout.dim.append(MultiArrayDimension())
v.layout.dim.append(MultiArrayDimension())
v.layout.dim[0].label = 'x'
v.layout.dim[1].label = 'y'
v.layout.dim[0].size = 2
v.layout.dim[1].size = 3
v.data = [1, 2, 3, 4, 5, 6]
print(v)


rospy.init_node('inference_from_matlab')
pub = rospy.Publisher('/testme', Float32MultiArray, latch=True)
pub.publish(v)


r           = rospy.Rate(FPS)
while (not rospy.is_shutdown()):
    r.sleep()


