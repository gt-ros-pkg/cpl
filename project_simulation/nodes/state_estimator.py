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
PUB_RATE = 10

'''
Collects following information:
- Probability distribution from the inference engine
- Bin - alvar_markers or bin-id & loc-names
- endfactor position, as well as if its carrying a bin(bin_id in that case)
'''


endf_recd = False
bin_loc_recd = False
prob_dist_recd = False
work_bins_recd = False

bins_locs_cur =None
prob_dist_cur = None 
endf_cur = None
work_bins_cur = None

cur_msg_no = 0 
 
def pub_as_one():
    global endf_recd, bin_loc_recd, prob_dist_recd, work_bins_recd, bins_locs_cur, prob_dist_cur, endf_cur, work_bins_cur, cur_msg_no

    if (endf_recd and bin_loc_recd and prob_dist_recd and work_bins_recd):
        msg_pub = project_simulation.msg.BinStateEstimate()
        if not endf_cur == None:
            msg_pub.robot_is_moving = endf_cur.performing_task.data
            msg_pub.robot_bin_grasped = endf_cur.bin_id.data
        else:
            return
        if not bin_loc_recd == None:
            msg_pub.bin_slots = bins_locs_cur
        else:
            return
        if not prob_dist_cur == None:
            msg_pub.inference = prob_dist_cur
        else:
            return
        if not work_bins_cur == None:
            msg_pub.reachable_slots = work_bins_cur

        cur_msg_no +=1
        
        msg_pub.header.stamp = rospy.Time.now()
        msg_pub.header.seq = cur_msg_no
        pub_cur_state.publish(msg_pub)
        
        #debug
        print "Pblishin"
        return

def endf_listen(endf_msg):
    global endf_recd, endf_cur
    if not endf_recd:
        print "Endfactor"
    endf_recd = True
    endf_cur = endf_msg
    return

#populate bin lists and empty lists
def bin_loc_listen(bin_loc_msg):
    global bin_loc_recd, bins_locs_cur
    if not bin_loc_recd:
        print "Populate bins"
    bin_loc_recd = True
    n = bin_loc_msg.bin_array.__len__()
    bins_locs_cur = [-1]*n
    #bin_temp = int8()
    for bin_loc in bin_loc_msg.bin_array:
        index = int(bin_loc.location.data[1:])
       # bin_temp.data = bin_loc.bin_id.data
        #bins_locs_cur[index] = bin_temp.data
        bins_locs_cur[index] = bin_loc.bin_id.data

    return

def inf_prob_listen(prob_msg):
    global prob_dist_recd, prob_dist_cur
    if not prob_dist_recd:
        print "Inference probability"
    prob_dist_recd = True
    prob_dist_cur = prob_msg
    return

def listen_work_bins(work_bin_msg):
    global work_bins_recd, work_bins_cur
    if not work_bins_recd:
        print "Listened Work bins"
    work_bins_recd = True
    work_bins_cur = []
    for work_loc in work_bin_msg.string_array:
        work_bins_cur.append(int(work_loc.data[1:]))
    return

if __name__ == '__main__':

    rospy.init_node('state_estimator')

    #bin positions
    sub_bins = rospy.Subscriber('bins_robo_sim', 
                                project_simulation.msg.bins_loc, bin_loc_listen)
    
    #end-factor position
    sub_endf = rospy.Subscriber('endfactor_bin', 
                                project_simulation.msg.endf_bin, endf_listen)
    
    #nam's inference
    sub_inf_prob = rospy.Subscriber('/inference/all_distributions', 
                                    project_simulation.msg.BinInference
                                    ,inf_prob_listen)
    #bins in the workspace
    sub_reach_bins = rospy.Subscriber('reachable_bins', 
                                      project_simulation.msg.StringArray,
                                      listen_work_bins)
    
    #publish as one
    pub_cur_state = rospy.Publisher('current_state', 
                                    project_simulation.msg.BinStateEstimate)
    
    loop_rate = rospy.Rate(PUB_RATE)
    while not rospy.is_shutdown():
        pub_as_one()
        loop_rate.sleep()
        r.sleep()
