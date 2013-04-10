#!/usr/bin/env python
import rospy
import sys

import roslib
roslib.load_manifest('project_simulation')
import time


from geometry_msgs.msg import *
from std_msgs.msg import *
from project_simulation.msg import *
from visualization_msgs.msg import *

import math
import copy
import tf

#publish Hz
PUB_RATE = 5



if __name__=='__main__':
    rospy.init_node('test_prob')
    pub_prob = rospy.Publisher('inference_dist', rospy_tutorials.msg.Floats)
    
    loop_rate = rospy.Rate(PUB_RATE)
    
    temp_msg = rospy_tutorials.msg.Floats()
    temp_msg.data = [1.0, 2., 3.2, 4.4]
    
    while not rospy.is_shutdown():
        pub_prob.publish(temp_msg)
        temp_msg.data[0] += 1.
        loop_rate.sleep()
        
