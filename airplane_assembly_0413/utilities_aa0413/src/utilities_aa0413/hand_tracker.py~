#! /usr/bin/python

import rospy
import roslib
roslib.load_manifest('utilities_aa0413')
import tf


from geometry_msgs.msg import *

ROS_TOPIC_LEFTHAND  = "left_hand"
ROS_TOPIC_RIGHTHAND = "right_hand"

####################################################
# var
####################################################
lefthand_sub 	= None
righthand_sub	= None

latest_lefthand_msg 	= None
latest_righthand_msg 	= None

####################################################
# functions
####################################################

def init():
    global lefthand_sub, righthand_sub
    lefthand_sub  = rospy.Subscriber(ROS_TOPIC_LEFTHAND, PoseStamped, left_hand_callback)
    righthand_sub = rospy.Subscriber(ROS_TOPIC_RIGHTHAND, PoseStamped, right_hand_callback)

def left_hand_callback(msg):
    global latest_lefthand_msg
    latest_lefthand_msg = msg

def right_hand_callback(msg):
    global latest_righthand_msg
    latest_righthand_msg = msg

def get_latest_msg():
    return latest_lefthand_msg, latest_righthand_msg 



####################################################
# test
####################################################

def main():
    print 'nothing'

if __name__ == '__main__' :
    main()























































































