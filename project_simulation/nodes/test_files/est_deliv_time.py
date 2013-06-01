#!/usr/bin/env python

#########################
# Estimate the average distance between bin in workspace versus, bin not there
########################

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

#constant robot velocity(m/s)
ROBO_VEL = 0.75

#time taken to pick-up put down bin (s)
ROBO_PICK = 1.0
ROBO_PUT = 1.0

rest_above_by = 0.2


#work-space
work_space = ['L0', 'L1', 'L2']

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




if __name__ == '__main__':
    not_work_pos = []
    work_pos = []
    
    for slocation in slocations:
        is_work = False
        for work_loc in work_space:
            if work_loc == slocation['name']:
                work_pos.append(slocation['position'])
                is_work = True
        if not is_work:
            not_work_pos.append(slocation['position'])
                

    total_distance = 0
    no_of_cases = 0

    for workP in work_pos:
        for not_workP in not_work_pos:
            total_distance += calc_euclid(list(workP), list(not_workP))
            no_of_cases += 1

    avg_dist = (total_distance)/no_of_cases + rest_above_by
    avg_time = avg_dist/ROBO_VEL + ROBO_PICK + ROBO_PUT

    print "Average Distance = " + str(avg_dist)
    print "Average Time = " + str(avg_time)
