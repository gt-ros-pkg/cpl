#!/usr/bin/env python
import rospy
import roslib; roslib.load_manifest('airplane_assembly_inference_0313')
import tf

from std_msgs.msg import *

import array
import numpy

import socket
import time
import sys
import pylab
import struct

from lxml import etree
from copy import deepcopy

from geometry_msgs.msg import *
from project_simulation.msg import *

from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from ar_track_alvar.msg import *


# from hrl_geom.pose_converter import PoseConv

from airplane_assembly_inference_0313.inference_utilities import *


ROS_TOPIC_LEFTHAND  = "left_hand"
ROS_TOPIC_RIGHTHAND = "right_hand"
ROS_TOPIC_BINMARKES = "ar_pose_marker"
ROS_TOPIC_IMAGE     = "kinect/color_image"
ROS_TOPIC_ROBOT_BIN = "endfactor_bin"

TF_WORLD     = "base_link"
TF_KINECT    = "kinect0_rgb_optical_frame"
TF_WEBCAM    = "lifecam1_optical_frame"

BIN_NUM      = 20

T            = 900

FPS                = 30
DOWN_SAMPLE_FACTOR = 7

MAX_NAME_LENGTH = 20

PORT = 54321



#####################################################
# Set up server, waiting for matlab to connect
#####################################################

s 	= None
conn 	= None
addr 	= None

def set_up_server():
    # set up server
    print 'Starting server, waiting for matlab to connect......'
    global s
    global conn
    global addr
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 
                 s.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) | 1)
    s.bind(('', PORT))
    s.listen(1)
    conn, addr = s.accept()

    print 'connected by ', addr


#####################################################
# Update hands & bin data
#####################################################
running          = False
framecount       = 0

webcam_to_w      = 0
kinect_to_w      = 0

lefthand         = None
righthand        = None
bins             = None
robot_moving     = None

lefthand_msgnum  = 0
righthand_msgnum = 0
bin_msgnum       = 0

inference_num    = 0;
    
def bins_callback(data):
    global bins
    bins = data.markers
    if running:
        global bin_msgnum
        bin_msgnum = bin_msgnum + 1

def left_hand_callback(data):
    global lefthand
    lefthand = data
    if running:
        global lefthand_msgnum
        lefthand_msgnum = lefthand_msgnum + 1

def right_hand_callback(data):
    global righthand
    righthand = data
    if running:
        global righthand_msgnum
        righthand_msgnum = righthand_msgnum + 1


def endf_bin_callback(data):
    global robot_moving
    robot_moving = data.performing_task.data



#####################################################
# PLANNING
#####################################################

matlab_plan  = None
my_plan      = etree.Element('myplan')
move_bin_pub = None

def get_data(bytenum):
    data = ''
    while (len(data) < bytenum):
        moredata = conn.recv(bytenum - len(data))
        data = data + moredata

        if (len(moredata) == 0):
            print 'exit'
            conn.close()
            s.close()
            sys.exit()
            
    return data;

def get_string(length):
    data = get_data(length)
    return data

def get_int():
    data = get_data(4)
    return struct.unpack('>i', data)[0]

def get_next_string():

        # check data available
        try:
            conn.settimeout(0)
            conn.recv(1, socket.MSG_PEEK)
            conn.settimeout(None)
   
        except socket.error as msg:
            #print 'data not available'
            return None
        
        # publish
        l = get_int()
        s = get_string(l)
        return s
        

def check_for_new_planning():
    global matlab_plan
    while True:
        s = get_next_string()
        if s is None:
             break;
        else:
             new_matlab_plan = etree.fromstring(s)
             valid_new_plan = True
             for e1 in my_plan:
                 for e2 in new_matlab_plan:
                     if e1.tag == 'events' and e2.tag == 'events' and e1.find('bin_id').text == e2.find('bin_id').text :
                         # valid_new_plan = False
                         aaaaaaaa = 1

             if len(my_plan) > 0 and len(new_matlab_plan) > 0 and my_plan[-1].find('sname').text == new_matlab_plan[0].find('sname').text:
                 valid_new_plan = False

             if valid_new_plan:
                 matlab_plan = new_matlab_plan
                 print 'received a valid plan'
             else:
                 print 'received an invalid plan'

def update_my_plan():
    global my_plan
    #if my_plan is None:
    #   my_plan = matlab_plan
    
    # check plan validity
    # todo
    #my_plan = matlab_plan



def execute_plan():
    if matlab_plan is None:
        return
    e = matlab_plan.find('events')
    if e is None:
        return
    
    execute_time = int(e.find('optimal_t').text) - int(e.find('pre_duration').text) 
    execute_time = execute_time * DOWN_SAMPLE_FACTOR / FPS + get_begin_time().secs
    current_time = rospy.Time.now().to_nsec() / 1000000000.0
    
    if execute_time-0.1 <= current_time:

        # add to my plan
        my_plan.append(e) # remove e from matlab_plan and add to my_plan
        my_plan[-1].append(etree.Element('execute_time'))
        my_plan[-1].append(etree.Element('matlab_execute_time'))
        my_plan[-1].find('execute_time').text = str(execute_time)
        my_plan[-1].find('matlab_execute_time').text = str(int((execute_time - get_begin_time().secs) * FPS / DOWN_SAMPLE_FACTOR))
        my_plan[-1].find('matlab_execute_time').set('rows', '1')
        my_plan[-1].find('matlab_execute_time').set('cols', '1')
        
        # send to matlab
        s = etree.tostring(my_plan)
        conn.sendall(struct.pack('>i', len(s)))
        conn.sendall(s)

        # lets the robot MOVE
        supposed_duration = float(e.find('pre_duration').text) + float(e.find('post_duration').text) 
        supposed_duration = supposed_duration * DOWN_SAMPLE_FACTOR / FPS
        print '>>> start action: ' + my_plan[-1].find('sname').text + ' (' + str(supposed_duration) + 's)'
        move_bin_msg = move_bin()
        move_bin_msg.bin_id = int(my_plan[-1].find('bin_id').text);
        move_bin_msg.move_near_human = my_plan[-1].find('sname').text[0] == 'A';
        move_bin_pub.publish(move_bin_msg);
        a1 = rospy.Time.now().to_nsec() / 1000000000.0
        #while not robot_moving and (rospy.is_shutdown() == False):
        #    rospy.sleep(0.1)
        #print 'ok robot started moving'
        rospy.sleep(1)
        while robot_moving and (rospy.is_shutdown() == False):
            rospy.sleep(0.1)
        a2 = rospy.Time.now().to_nsec() / 1000000000.0
        print '>>> finish action in ', (a2 - a1), ' (s) '
            
#####################################################
# MAIN
#####################################################


def printplan():
        if not my_plan is None:
          for e in my_plan:
            if e.tag == 'events':
                exe_t = float(e.find('execute_time').text)
                remain_time = exe_t - rospy.Time.now().to_nsec() / 1000000000.0
                print e.find('name').text, ' executed in ', remain_time, ' (s)'
                #print etree.tostring(e)
        if not matlab_plan is None:
          for e in matlab_plan:
            if e.tag == 'events':
                optimal_t = int(e.find('optimal_t').text)
                remain_time = optimal_t * DOWN_SAMPLE_FACTOR / FPS - (rospy.Time.now().to_nsec() - get_begin_time().nsecs) / 1000000000.0
                print e.find('name').text, ' in ', remain_time, ' (s), optimal_t ', optimal_t
                #print etree.tostring(e)


def main():


    # init
    rospy.init_node('planning_from_matlab')

    # sub
    #rospy.Subscriber(ROS_TOPIC_BINMARKES, AlvarMarkers, bins_callback)
    #rospy.Subscriber(ROS_TOPIC_LEFTHAND, PoseStamped, left_hand_callback)
    #rospy.Subscriber(ROS_TOPIC_RIGHTHAND, PoseStamped, right_hand_callback)
    rospy.Subscriber(ROS_TOPIC_ROBOT_BIN , endf_bin, endf_bin_callback)

    # pub
    global move_bin_pub
    move_bin_pub = rospy.Publisher('move_bin', move_bin)
    #move_bin_msg = move_bin()
    #move_bin_msg.bin_id = 14;
    #move_bin_msg.move_near_human = True;
    #rospy.sleep(0.1)
    #move_bin_pub.publish(move_bin_msg);
    #rospy.sleep(10)
    #return

    # wait for client
    set_up_server()

    
    # wait for start
    print 'Wait for the task to start'
    while get_begin_time() is None:

        rospy.sleep(0.1)

        if rospy.is_shutdown():
           sys.exit()


    # processing loop
    print 'Task started....'
    r = rospy.Rate(10)
    while (rospy.is_shutdown() == False):

        check_for_new_planning()
        # update_my_plan()
        execute_plan()

        # print plan  
        # printplan()

        # sleep till next frame
        if USE_ROS_RATE:
            r.sleep()


if __name__ == '__main__':
    try:
       main()
    except Exception as e:
       print e
    finally:
       conn.sendall('exit!'); # exist signal
       conn.close()
       s.close()
       sys.exit()











