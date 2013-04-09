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

#constant robot velocity(m/s)
ROBO_VEL = 1

#time taken to pick-up put down bin (s)
ROBO_PICK = 0.5
ROBO_PUT = 0.5

task_done = False
task_cnt =0
frame_of_reference = '/lifecam1_optical_frame'

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

#current bins and their locations
cur_bin_list = []

#locations not currently containing bins
empty_locations = []

#list of tasks to perform
task_list = []

#endfactor position
endfactor_rest = [0.226263331563,0.225740063166,1.17398472964]
endfactor_cur_pos = [0.226263331563,0.225740063166,1.17398472964]

endfactor_fix_orientation = [0.0, 1.0, 0.0, 0.0]

def listen_bin_loc(bin_loc_msg):
    global cur_bin_list, empty_locations, slocations

    #add bins to current bin list
    cur_bin_list = []
    for temp_bin_msg in bin_loc_msg.bin_array:
        for slocation in slocations:
            if slocation['name'] == temp_bin_msg.location.data:
                temp_bin = {'id':temp_bin_msg.bin_id.data, 'location':slocation['name']}
                cur_bin_list.append(temp_bin)
                break
    
    #populate empty locations
    empty_locations = []
    for slocation in slocations:
        bin_in_loc = False
        for cur_bin in cur_bin_list:
            if cur_bin['location'] == slocation['name']:
                bin_in_loc = True
                break
        if not bin_in_loc:
            empty_locations.append(slocation['name'])

    #    do_tasks()
    return

    '''
    #debug
    print "******BINS********"
    for t_bin in cur_bin_list:
        print "bin, loc" + str(t_bin['id']) + ' , ' + str(t_bin['location']) 
        
    print "******EMPTY********"
    for e_loc in empty_locations:
        print "loc  : " + e_loc 
    '''

#check whether position received matches to stored location within 15 percent
def check_pos_match(pos_tup, pos_point):
    perc_x = 5*abs(pos_point.x/100)
    perc_y = 5*abs(pos_point.y/100)
    perc_z = 5*abs(pos_point.z/100)
    
    if (abs(pos_tup[0]-pos_point.x) > perc_x):
        return False
    if (abs(pos_tup[1]-pos_point.y) > perc_y):
        return False
    if (abs(pos_tup[2]-pos_point.z) > perc_z):
        return False
    
    return True

#perform tasks in list
def do_tasks():
    global cur_bin_list, task_cnt, task_list

    if task_cnt < task_list.__len__():
        temp_task = task_list[task_cnt]
        perform_task(temp_task)
        
        #debug
        print 'performed'
    else:
        #switch off node if task list done?
        #if not task_done:
        print 'Tasks complete'
        #task_done = True
        return True

    #increment global task counter
    task_cnt+=1
    return False

#move bin to a new position
def perform_task(cur_task):
    
    global cur_bin_list, empty_locations, ROBO_PICK, ROBO_PUT

    targ_bin_id = cur_task['bin_id']
    dest_loc = cur_task['location']
    before_task_delay = cur_task['delay']
    
    #wait for the task delay
    #TODO: compute task completion task and delay represents when task complete?
    #maybe I should be displaying endfactor while I wait
    time.sleep(before_task_delay)

    #ensure bin not already positioned correctly
    bin_not_found = True
    
    '''    #delay
    print 'The current list is'
    print cur_bin_list'''

    #find target bin
    for cur_bin in cur_bin_list:
        if cur_bin['id'] == targ_bin_id:
            targ_bin_cur_loc = cur_bin['location']
            bin_not_found = False
            break
    
    if bin_not_found:
        print "Bin id: " + str(targ_bin_id) + " was not found. Skipping task."
        return

    if targ_bin_cur_loc == dest_loc:
        print "Bin id: " + str(targ_bin_id) + "already at destination. Skipping task."
        return

    #check if target position empty
    location_not_empty = True
    for e_loc in empty_locations:
        if e_loc == dest_loc:
            location_not_empty = False
            break

    if location_not_empty:
        print "Location: " + str(dest_loc) + "is currently occupied. Skipping task."
        return

    #in case both those conditions met- do it!

    #move endfactor to bin
    move_endf(targ_bin_cur_loc)
    #pick up bin
    time.sleep(ROBO_PICK)
    #removing bin from ar_pose_markers
    pub_remove_bin(targ_bin_id)

    #move bin to target
    pub_endf_w_bin(targ_bin_id, dest_loc)
    #put down bin
    time.sleep(ROBO_PUT)
    pub_add_bin(targ_bin_id, dest_loc)
    #move endfactor to rest position
    move_endf(1, True)

#get position(x,y,z) associated with a location name
def get_location(loc_name):
    for slocation in slocations:
        if slocation['name'] == loc_name:
            return list(slocation['position'])
        
    print 'Error: Queried location name not in list, skipping task'

#check if endfactor at target ,bin otherwise bring it there
#move endfactor and associated bin to target
def pub_endf_w_bin(target_bin_id, destination_name):

    global endfactor_cur_pos, PUB_RATE, ROBO_VEL    
    destination = get_location(destination_name)
    tot_dist = calc_euclid(endfactor_cur_pos, destination)
    
    #if endfactor already in position
    if tot_dist ==0:
        return
    
    total_time_steps = math.floor((tot_dist/ROBO_VEL)*PUB_RATE)
    
    '''#debug
    print 'target location  steps' + str(destination) + '   ' + str(total_time_steps)    
#    time.sleep(2)'''

    step_x = (destination[0]-endfactor_cur_pos[0])/(total_time_steps)
    step_y = (destination[1]-endfactor_cur_pos[1])/(total_time_steps)
    step_z = (destination[2]-endfactor_cur_pos[2])/(total_time_steps)

    time_step = 0
    loop_rate = rospy.Rate(PUB_RATE)
    
    while(time_step<total_time_steps):
        endfactor_cur_pos[0] += step_x
        endfactor_cur_pos[1] += step_y
        endfactor_cur_pos[2] += step_z
        pub_endfactor()
        time_step+=1
        loop_rate.sleep()
        
    #put endfactor to target if it didn't get there
    endfactor_cur_pos = destination

#message to ar_poses simulator to remove the moving bin from its list
def pub_remove_bin(bin_to_rmv):
    rmv_msg = std_msgs.msg.UInt8()
    rmv_msg.data = bin_to_rmv
    rmv_bin_pub.publish(rmv_msg)

#message to include bin at given location into ar_poses list
#only position, no orientation
def pub_add_bin(bin_to_add, bin_location):
    msg = project_simulation.msg.bin_loc()
    msg.bin_id.data = bin_to_add
    msg.location.data = bin_location
    
    pub_bin_adder.publish(msg)

    '''bin_position = get_location(bin_location)

    temp_msg = proj_simul.msg.AlvarMarker()
    temp_msg.id = bin_to_add
    temp_msg.header.frame_id = frame_of_reference
    temp_msg.pose.header.frame_id = frame_of_reference


    temp_msg.pose.pose.orientation.x = endfactor_fix_orientation[0]
    temp_msg.pose.pose.orientation.y = endfactor_fix_orientation[1]
    temp_msg.pose.pose.orientation.z = endfactor_fix_orientation[2]
    temp_msg.pose.pose.orientation.w = endfactor_fix_orientation[3]

    temp_msg.pose.pose.position.x = bin_position[0]
    temp_msg.pose.pose.position.y = bin_position[1]
    temp_msg.pose.pose.position.z = bin_position[2]   

    #debug
    print 'The message'
    print temp_msg
    time.sleep(100)
    pub_bin_adder.publish(temp_msg)'''

#move the endfactor, no bin to move
def move_endf(target_loc_name, to_rest=False):
    global endfactor_cur_pos, PUB_RATE, ROBO_VEL, endfactor_rest

    if not to_rest:
        target_loc = get_location(target_loc_name)
    else:
        target_loc = copy.deepcopy(endfactor_rest)
    
    tot_dist = calc_euclid(endfactor_cur_pos, target_loc)
    total_time_steps = math.floor((tot_dist/ROBO_VEL)*PUB_RATE)

    if total_time_steps == 0:
        return

    '''#debug
    print 'distance = '+ str(tot_dist)
    print 'time steps = '+ str(total_time_steps)
    print 'target location  steps' + str(target_loc) + '   ' + str(total_time_steps)'''    

    step_x = (target_loc[0]-endfactor_cur_pos[0])/(total_time_steps)
    step_y = (target_loc[1]-endfactor_cur_pos[1])/(total_time_steps)
    step_z = (target_loc[2]-endfactor_cur_pos[2])/(total_time_steps)

    time_step = 0
    loop_rate = rospy.Rate(PUB_RATE)
    
    while (time_step<total_time_steps):
        endfactor_cur_pos[0] += step_x
        endfactor_cur_pos[1] += step_y
        endfactor_cur_pos[2] += step_z
        pub_endfactor()
        time_step+=1
        loop_rate.sleep()
        
    #put endfactor to target if it didn't get there
    endfactor_cur_pos = target_loc

#calculate euclidean distance between two vectors
def calc_euclid(vec_one, vec_two):
    if vec_one.__len__() == vec_two.__len__():
        vec_size = vec_one.__len__()
        distance = 0.0
        for i in range(vec_size):
            distance+= math.pow(vec_one[i]-vec_two[i],2)
        
        return math.sqrt(distance)

    #sizes don't match
    print "Sizes don't match can't compute euclidean distance"

#publish current endfactor position
def pub_endfactor():
    global endfactor_cur_pos, endfactor_fix_orientation

    temp_msg = geometry_msgs.msg.PoseStamped()
    temp_msg.header.frame_id = frame_of_reference
    temp_msg.pose.orientation.x = endfactor_fix_orientation[0]
    temp_msg.pose.orientation.y = endfactor_fix_orientation[1]
    temp_msg.pose.orientation.z = endfactor_fix_orientation[2]
    temp_msg.pose.orientation.w = endfactor_fix_orientation[3]

    temp_msg.pose.position.x = endfactor_cur_pos[0]
    temp_msg.pose.position.y = endfactor_cur_pos[1]
    temp_msg.pose.position.z = endfactor_cur_pos[2]
    
    endf_pub.publish(temp_msg)

    #visualize
    pub_viz_enf()

def pub_viz_enf():
    
    marker_shape = visualization_msgs.msg.Marker.CYLINDER
    
    global endfactor_cur_pos, endfactor_fix_orientation, endf_pub

    temp_msg = visualization_msgs.msg.Marker()
    temp_msg.header.stamp = rospy.Time.now()
    temp_msg.ns = 'robo_sim'
    temp_msg.id = 0
    temp_msg.type = marker_shape
    temp_msg.action = visualization_msgs.msg.Marker.ADD

    temp_msg.header.frame_id = frame_of_reference
    temp_msg.pose.orientation.x = 1.0#endfactor_fix_orientation[0]
    temp_msg.pose.orientation.y = 0.0#endfactor_fix_orientation[1]
    temp_msg.pose.orientation.z = 0.0#endfactor_fix_orientation[2]
    temp_msg.pose.orientation.w = 0.0#endfactor_fix_orientation[3]

    temp_msg.pose.position.x = endfactor_cur_pos[0]
    temp_msg.pose.position.y = endfactor_cur_pos[1]
    temp_msg.pose.position.z = endfactor_cur_pos[2]

    temp_msg.color.r = 1.0
    temp_msg.color.g = 0.0
    temp_msg.color.b = 0.0
    temp_msg.color.a = 0.9

    temp_msg.scale.x = 0.1
    temp_msg.scale.y = 0.1
    temp_msg.scale.z = 0.1
    temp_msg.lifetime = rospy.Duration()
    

    temp_msg.text = 'ENDFACTOR'
    endf_viz_pub.publish(temp_msg)


if __name__=='__main__':
    
    #ros
    rospy.init_node('robo_bin_mover')

    bin_loc_sub = rospy.Subscriber('bins_robo_sim', project_simulation.msg.bins_loc, listen_bin_loc)
    
    endf_pub = rospy.Publisher('endfactor_pose', geometry_msgs.msg.PoseStamped)
    
    rmv_bin_pub = rospy.Publisher('remove_bin', std_msgs.msg.UInt8)

    pub_bin_adder = rospy.Publisher('add_bin', project_simulation.msg.bin_loc)
    
    endf_viz_pub = rospy.Publisher('endfactor_visual', visualization_msgs.msg.Marker)


    #input tasks
    keep_doing = True
    
    while(keep_doing):
        more = False
        inps = raw_input('Add Task: <bin_id> <location> <delay>')
        inps = inps.rsplit(' ')
        #TODO: take inputs proper
        task_list.append({'bin_id':int(inps[0]), 'location':inps[1], 'delay':float(inps[2])})
        
        inps = raw_input('Add more(y/n)')
        if inps[0] == 'y':
            more = True
        
        done = False
        #do tasks in list
        if not more:
            while ((not rospy.is_shutdown()) and (not done)):
                done = do_tasks()
            inps = raw_input('Do more tasks?(y/n)')
            if not inps == 'y':
                keep_doing = False
            else:
                task_list = []
                task_cnt = 0
    
    #TODO: listen to hand positions
