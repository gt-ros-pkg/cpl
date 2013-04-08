#!/usr/bin/env python

import rospy

import roslib
roslib.load_manifest('project_simulation')
import time

from geometry_msgs.msg import *
from project_simulation.msg import *
from visualization_msgs.msg import *
from std_msgs.msg import *
import random
from math import exp
import numpy as np
import copy
from transformations import quaternion_matrix, quaternion_inverse
import tf
import numpy as np

#publish at Hz 
PUB_RATE = 30

#Human-Workspace
workspace = ['L0', 'L1', 'L2']

#noise
location_noise = 0.006
orientation_noise = 0.025

#possible locations
slocations = [ 
         { 'name' : 'L5' , 'position' : (-0.694954464364,0.264302079631,1.99317972681) , 'orientation' : (0.0864803654147,0.979654035105,-0.178136413992,0.0326578614002)},
         { 'name' : 'L3' , 'position' :  (-0.847151201376,0.297786050824,1.96464817216) , 'orientation' :  (0.099286275045,0.979143250933,-0.172672549803,0.040061456355)},
         { 'name' : 'L4' , 'position' : (-0.356133281188,0.200652416089,2.00604817833) , 'orientation' : (0.0864803654147,0.979654035105,-0.178136413992,0.0326578614002)},
         { 'name' : 'L6' , 'position' : (-0.526133281188,0.200652416089,2.00604817833) , 'orientation' : (0.0864803654147,0.979654035105,-0.178136413992,0.0326578614002)},

         { 'name' : 'L2' , 'position' : 
(-0.806263331563,-0.225740063166,2.17398472964), 'orientation' : 
  (0.0802827646415,0.978381335494,-0.188814418718,0.0259567976382)},
         { 'name' : 'L1' , 'position' : 
  (-0.663868350375,-0.237326254339,2.18771858991)
, 'orientation' : 
  (0.0831702389541,0.979029851475,-0.18352138094,0.0300526872406)},
         { 'name' : 'L0' , 'position' : 
  (-0.511209542213,-0.257163236559,2.18809949502)
, 'orientation' : 
(0.0864803654147,0.979654035105,-0.178136413992,0.0326578614002)
},
         { 'name' : 'L7' , 'position' : 
(0.738845303768,-0.216998119566,1.99359502486)
, 'orientation' : 
(-0.20227136896,0.962479020114,-0.179088366973,-0.025451639519)},
         { 'name' : 'L8' , 'position' : 
  (0.578156534057,0.0753144815163,1.88095830843)
, 'orientation' : 
(-0.155360904198,0.971362830021,-0.178368800013,-0.0224011848603)},
         { 'name' : 'L9' , 'position' : 
(0.394281491914,0.0255446236486,1.88485177488)
, 'orientation' : 
  (-0.168410139014,0.968065944316,-0.184818178623,-0.0181271449139)},

         { 'name' : 'L10' , 'position' : 
  (0.214970708732,-0.0341420080945,1.89436220148)
, 'orientation' : 
  (-0.167669713811,0.967651750635,-0.188267888617,-0.00959993117272)},
         { 'name' : 'L11' , 'position' : 
  (0.388313527453,-0.395303418346,2.04776721087)
, 'orientation' : 
  (-0.172801370397,0.967825570325,-0.181500892043,-0.0226003982657)},
         { 'name' : 'L12' , 'position' : 
  (0.588363257575,-0.288420043663,2.01260385776)
, 'orientation' :   (-0.201361148913,0.961335468405,-0.186586019109,-0.0217591904042)},
     { 'name' : 'L13' , 'position' : 
  (0.458363257575,-0.288420043663,2.01260385776)
, 'orientation' :   (-0.201361148913,0.961335468405,-0.186586019109,-0.0217591904042)}
         ]

bins_locs = [
    {'bin_id':2, 'location':'L0'}
    ,{'bin_id':3, 'location':'L1'}
    ,{'bin_id':11, 'location':'L2'}
    ,{'bin_id':10, 'location':'L3'}
    ,{'bin_id':12, 'location':'L4'}
    ,{'bin_id':7, 'location':'L5'}
    ,{'bin_id':14, 'location':'L6'}
    ,{'bin_id':15, 'location':'L7'}
    ,{'bin_id':13, 'location':'L8'}
    #,{'bin_id':16, 'location':'L9'}
    #,{'bin_id':17, 'location':'L10'}
    #,{'bin_id':18, 'location':'L11'}
    #,{'bin_id':19, 'location':'L12'}
    #,{'bin_id':20, 'location':'L13'}
    ]

#frame for markers
frame_of_reference = '/lifecam1_optical_frame'
#frame_of_reference = '/base_link'
#which bin is to be removed
bin_for_removal = None

#noise for hand-touch in bin-frame
hand_t_mean = [0.0091831, -0.13022, -0.022461]
hand_t_cov = [[0.00020556, 1.7289e-05, 0.00015445], [1.7289e-05, 0.00052374, -3.6877e-05], [0.00015445, -3.6877e-05, 0.00058416]]

#inverse transforms for workspace
inv_trans = []

#get the inverse transform of the work-bin frames
def set_wbs_transf():
    global workspace, slocations, inv_trans
    inv_trans = []

    for work_loc in workspace:
        for slocation in slocations:
            if work_loc == slocation['name']:
                translation = copy.deepcopy(slocation['position'])
                translation = np.array(translation)
                quaternion = copy.deepcopy(slocation['orientation'])
                rot_mat = quaternion_matrix(quaternion)
                inv_trans.append({'name':slocation['name'], 
                                  'translation': translation, 
                                  'rotation': rot_mat})
                break
    
    #debug
    print inv_trans
    return

#samples from gaussian for hand-offset in marker's frame
#returns the same number of samples as bins in workspace
def samp_hand_offset():
    global hand_t_mean, hand_t_cov, workspace
    n = workspace.__len__()
    samples = np.random.multivariate_normal(hand_t_mean,hand_t_cov,(n))
    return samples

#generate positions of workspace bins for the human
def gen_locs_human():
    global workspace, inv_trans
    hand_t_locs = []
    n = workspace.__len__()
    sample_offs = samp_hand_offset()

    for i in range(n):
        cur_loc = workspace[i]
        cur_off = list(sample_offs[i])
        for trans in inv_trans:
            if trans['name'] == cur_loc:
                #homogenize
                cur_off.append(1.)
                homo_pos = np.array(cur_off)
                #rotate
                homo_pos = list(np.dot(homo_pos, trans['rotation']))
                #translate
                new_pos = np.array([homo_pos[0]/homo_pos[3], 
                                    homo_pos[1]/homo_pos[3], 
                                    homo_pos[2]/homo_pos[3]])
                tran_by = trans['translation']
                new_pos = [new_pos[i]+tran_by[i] for i in range(new_pos.__len__())]
                hand_t_locs.append({'name':cur_loc, 'position':new_pos})
                break
    
    return hand_t_locs

#receive message to remove bin from publishing list
def bin_rmv(bin_to_rmv_msg):

    bin_to_rmv = bin_to_rmv_msg.data
    global bins_locs, bin_for_removal
    
    no_of_bins = bins_locs.__len__()
    
    for i in range(no_of_bins):
        if bins_locs[i]['bin_id']== bin_to_rmv:
            del bins_locs[i]
            bin_for_removal = bin_to_rmv
            return
    
    #incase it didn't return then bin was not found
    print "bin no: "+ str(bin_to_rmv) + "was not found"

def add_bin_id(bin_n_loc):
    global bins_locs
    temp_bin = {'bin_id':bin_n_loc.bin_id.data, 'location':bin_n_loc.location.data}
    bins_locs.append(temp_bin)
    return

def pub_bins():

    global bin_locs, slocations, PUB_RATE, bin_for_removal, location_noise, orientation_noise
    

    br = tf.TransformBroadcaster()
    #perturbed by gaussian noise, thought to simulate tracking noise
    pub = rospy.Publisher('ar_pose_marker', project_simulation.msg.AlvarMarkers)
    #human gets actual bin positions
    pub_human = rospy.Publisher('ar_pose_marker_hum', project_simulation.msg.AlvarMarkers)
    #location of bins to robot simulator
    pub_robo = rospy.Publisher('bins_robo_sim', project_simulation.msg.bins_loc)
    #ROS-viz
    viz_pub = rospy.Publisher('ar_poses_visual', visualization_msgs.msg.MarkerArray)

    #subscribers- add/remove bin from list
    bin_rmv_sub = rospy.Subscriber('remove_bin', std_msgs.msg.UInt8, bin_rmv)
    bin_add_sub = rospy.Subscriber('add_bin', project_simulation.msg.bin_loc, add_bin_id)

    
    r = rospy.Rate(PUB_RATE)

    while not rospy.is_shutdown() :
        ar_markers = []
        hum_ar_markers = []
        ar_viz_markers = []
        markers_robo = []

        #generate human work positions
        hum_locs = gen_locs_human()
        #debug 
        print "Human locations:" + str(hum_locs)
        time.sleep(1000)

        
        for bin in bins_locs:
            
            #message for robo-sim
            temp_bin_loc = project_simulation.msg.bin_loc()
            temp_bin_loc.bin_id.data = bin['bin_id']
            temp_bin_loc.location.data = bin['location']
            markers_robo.append(temp_bin_loc)

            marker = project_simulation.msg.AlvarMarker()
            marker.header.frame_id = frame_of_reference
            marker.pose.header.frame_id = frame_of_reference
            marker.id = bin['bin_id']
            

            temp_msg = visualization_msgs.msg.Marker()
            set_viz_marker(temp_msg, marker.id)
            temp_msg.id = marker.id
            
            for slocation in slocations:
                if slocation['name']==bin['location']:
                    marker.pose.pose.position.x = slocation['position'][0]
                    marker.pose.pose.position.y = slocation['position'][1]
                    marker.pose.pose.position.z = slocation['position'][2]
                    marker.pose.pose.orientation.x = slocation['orientation'][0]
                    marker.pose.pose.orientation.y = slocation['orientation'][1]
                    marker.pose.pose.orientation.z = slocation['orientation'][2]
                    marker.pose.pose.orientation.w = slocation['orientation'][3]
                    
                    #add only work bins in human list
                    for hum_loc in hum_locs:
                        if hum_loc['name'] == slocation['name']:
                            marker_human = copy.deepcopy(marker)
                            marker_human.pose.pose.position.x = hum_loc['position'][0]          
                            marker_human.pose.pose.position.y = hum_loc['position'][1]                       
                            marker_human.pose.pose.position.z = hum_loc['position'][2]
                            hum_ar_markers.append(marker_human)
                            break
                    
                    #add gaussian noise
                    #marker =add_gauss_noise(marker)
                    
                    #visual marker
                    temp_msg.pose = copy.deepcopy(marker.pose.pose)
                    
                    #add to ar_pose list and visual list
                    ar_markers.append(marker)
                    ar_viz_markers.append(temp_msg)
                    

                    '''br.sendTransform( slocation['position'], 
                                      slocation['orientation'], 
                                      rospy.Time.now(), 
                                      'Bin_'+str(bin['bin_id'])+'__'
                                      +str(bin['location']), 
                                      frame_of_reference)'''
                    #post to tf
                    temp_tf_pos = (marker.pose.pose.position.x, 
                                   marker.pose.pose.position.y, 
                                   marker.pose.pose.position.z)
                    temp_tf_orient = (marker.pose.pose.orientation.x, 
                                      marker.pose.pose.orientation.y, 
                                      marker.pose.pose.orientation.z,
                                      marker.pose.pose.orientation.w)
                    br.sendTransform( temp_tf_pos, 
                                      temp_tf_orient, 
                                      rospy.Time.now(), 
                                      'ar_marker_'+str(bin['bin_id']), 
                                      frame_of_reference)

                    

                    '''#debug
                    print ar_viz_markers
                    time.sleep(10)'''

        #add a delete visual marker for bin removed
        if not bin_for_removal == None:
            temp_marker = gen_delete_bin(bin_for_removal)
            ar_viz_markers.append(temp_marker)
            bin_for_removal = None
                    
        #publish markers to 'ar_pose_marker'
        msg = project_simulation.msg.AlvarMarkers()
        msg.header.frame_id = frame_of_reference
        msg.markers = ar_markers
        pub.publish(msg)

        #publish markers to 'hum_ar_pose_marker'
        msg_hum = project_simulation.msg.AlvarMarkers()
        msg_hum.header.frame_id = frame_of_reference
        msg_hum.markers = hum_ar_markers
        pub_human.publish(msg_hum)

        #publish to robot simulator
        robo_msg = project_simulation.msg.bins_loc()
        robo_msg.bin_array = markers_robo        
        pub_robo.publish(robo_msg)
        
        '''#debug
        print ar_viz_markers
        time.sleep(40)'''
        
        #publish visual markers
        viz_msg = visualization_msgs.msg.MarkerArray()
        viz_msg.markers = ar_viz_markers
        viz_pub.publish(viz_msg)
        
        r.sleep()

#add gaussian noise to msg
def add_gauss_noise(marker):
    marker.pose.pose.position.x += random.gauss(0.0, location_noise)
    marker.pose.pose.position.y += random.gauss(0.0, location_noise)
    marker.pose.pose.position.z += random.gauss(0.0, location_noise)
    marker.pose.pose.orientation.x += random.gauss(0.0, orientation_noise)
    marker.pose.pose.orientation.y += random.gauss(0.0, orientation_noise)
    marker.pose.pose.orientation.z += random.gauss(0.0, orientation_noise)
    marker.pose.pose.orientation.w += random.gauss(0.0, orientation_noise)
    
    return marker
        
#return a visualization marker that deletes argument bin id
def gen_delete_bin(rmv_bin_id):
    temp_msg  = visualization_msgs.msg.Marker()
    temp_msg.id = rmv_bin_id
    temp_msg.header.stamp = rospy.Time()
    temp_msg.ns = 'marker_sim'
    temp_msg.action = visualization_msgs.msg.Marker.DELETE
    
    temp_msg.header.frame_id = frame_of_reference
    return temp_msg

    
def set_viz_marker(temp_msg, bin_id):
    marker_shape = visualization_msgs.msg.Marker.CUBE
    
    temp_msg.header.stamp = rospy.Time()
    temp_msg.ns = 'marker_sim'
    temp_msg.type = marker_shape
    temp_msg.action = visualization_msgs.msg.Marker.ADD
    
    temp_msg.header.frame_id = frame_of_reference
    
    temp_msg.color.r = 0.0#exp(-bin_id)
    temp_msg.color.g = 0.5#exp(-bin_id)
    temp_msg.color.b = 0.8#exp(-bin_id)
    temp_msg.color.a = 0.7#exp(-bin_id)
    
    temp_msg.scale.x = 0.05
    temp_msg.scale.y = 0.1
    temp_msg.scale.z = 0.03
    temp_msg.lifetime = rospy.Duration()
    return



#MAIN
if __name__ == '__main__':

    # init 
    rospy.init_node('ar_pose_markers_pub')

    #set transform for work-space locations
    set_wbs_transf()

    pub_bins()
