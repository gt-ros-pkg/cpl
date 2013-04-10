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
PUB_RATE = 60

'''
Collects following information:
- Probability distribution from the inference engine
- Bin - alvar_markers or bin-id & loc-names
- endfactor position, as well as if its carrying a bin(bin_id in that case)
'''


endf_recd = False
bin_loc_recd = False
prob_dist_recd = False

bins_locs_cur =None
prob_dist_cur = None 
endf_cur = None
 
def pub_as_one():
    global endf_recd, bin_loc_recd, prob_dist_recd, bins_locs_cur, prob_dist_cur, endf_cur

    if (endf_recd and bin_loc_recd and prob_dist_recd):
        msg_pub = project_simulation.msg.state_estimate()
        if not endf_cur == None:
            msg_pub.endfactor_pos_n_bin = endf_cur
        else:
            return
        if not bin_loc_recd == None:
            msg_pub.bin_locations = bins_locs_cur
        else:
            return
        if not prob_dist_cur == None:
            msg_pub.inf_prob_dist = prob_dist_cur
        else:
            return
        pub_cur_state.publish(msg_pub)

def endf_listen(endf_msg):
    global endf_recd, endf_cur
    endf_recd = True
    endf_cur = endf_msg
    return

#populate bin lists and empty lists
def bin_loc_listen(bin_loc_msg):
    global bin_loc_recd, bins_locs_cur
    bin_loc_recd = True
    bins_locs_cur = bin_loc_msg
    return

def inf_prob_listen(prob_msg):
    global prob_dist_recd, prob_dist_cur
    prob_dist_recd = True
    prob_dist_cur = prob_msg

if __name__ == '__main__':

    rospy.init_node('state_estimator')

    #bin positions
    sub_bins = rospy.Subscriber('bins_robo_sim', 
                                project_simulation.msg.bins_loc, bin_loc_listen)
    
    #end-factor position
    sub_endf = rospy.Subscriber('endfactor_pose', 
                                project_simulation.msg.endf_bin, endf_listen)
    
    #nam's inference
    sub_inf_prob = rospy.Subscriber('inference_dist', rospy_tutorials.msg.Floats, 
                                    inf_prob_listen)
    
    #publish as one
    pub_cur_state = rospy.Publisher('current_state', 
                                    project_simulation.msg.state_estimate)
    
    while not rospy.is_shutdown():

        pub_as_one()
