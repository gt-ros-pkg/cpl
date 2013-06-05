#!/usr/bin/env python
import rospy

import roslib; roslib.load_manifest('airplane_assembly_inference_0313')
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

PORT = 12345


#####################################################
# 
#####################################################

def get_FPS():
     return FPS

def get_downsampleratio():
     return 5

begin_time = None

def begin_time_cb(msg):
     global begin_time
     begin_time = msg.data

def get_begin_time():
     if begin_time is None:
          s = rospy.Subscriber("/inference/begin_time", std_msgs.msg.Time, begin_time_cb)
          rospy.sleep(0.1)
          s.unregister()
     return begin_time
            
#####################################################
# MAIN
#####################################################

def main():
    print 'Nothing here'


if __name__ == '__main__':
    main()










