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
import random

#publish Hz
PUB_RATE = 5



if __name__=='__main__':
    rospy.init_node('test_prob')
    pub_prob = rospy.Publisher('all_distributions', project_simulation.msg.BinInference)
    
    loop_rate = rospy.Rate(PUB_RATE)
    starting_time = rospy.Time.now()

    count =0 
    while not rospy.is_shutdown():
        temp_pub_msg = project_simulation.msg.BinInference()
        temp_pub_msg.header.seq = count
        count +=1
        temp_pub_msg.header.stamp = rospy.Time.now()
        temp_pub_msg.start_time = starting_time
        temp_pub_msg.T_len = 300
        temp_pub_msg.period = 0.2
        temp_pub_msg.t_cur = 0 
        temp_arr = [random.random() for i in range(300*40)]
        temp_pub_msg.distributions = temp_arr
        pub_prob.publish(temp_pub_msg)
        loop_rate.sleep()
