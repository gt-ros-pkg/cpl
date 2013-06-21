#!/usr/bin/env python
import rospy

import roslib; roslib.load_manifest('airplane_assembly_inference_0313')
import tf

from std_msgs.msg import *

from geometry_msgs.msg import *
from project_simulation.msg import BinInference

from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from ar_track_alvar.msg import *

from math import *

import array
import numpy

import socket
import time
import sys
import pylab
import struct
from lxml import etree

pylab.ion()


USE_ROS_RATE        = False

PUBLISH_MULTIARRAY_ALLDISTRIBUTIONS = False
PUBLISH_INDIVIDUAL_DISTRIBUTION     = True

ROS_TOPIC_LEFTHAND  = "left_hand"
ROS_TOPIC_RIGHTHAND = "right_hand"
ROS_TOPIC_BINMARKES = "ar_pose_marker"
ROS_TOPIC_IMAGE     = "kinect/color_image"

ROS_TOPIC_ACTION_NAME_GT = "action_name"
ROS_TOPIC_WORKSPACE_BINS = "workspace_bins"

TF_WORLD     = "base_link"
TF_KINECT    = "kinect0_rgb_optical_frame"
TF_WEBCAM    = "lifecam1_optical_frame"

BIN_NUM      = 20
MAX_WS_BINS  = 20

T            = 1000

FPS          = 30
DOWN_SAMPLING_RATE = 7

MAX_NAME_LENGTH = 20

PORT = 12341


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
action_name      = "N/A"
workspace_bins   = None

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

def action_name_callback(data):
    global action_name
    action_name = data.data

def workspace_bins_callback(data):
    global workspace_bins
    workspace_bins = data.data


#####################################################
# Send data to Matlab
#####################################################

# br = tf.TransformBroadcaster()


def lookup_transforms():
    global webcam_to_w
    global kinect_to_w

    lr = tf.TransformListener();

    while (rospy.is_shutdown() == False):
        try:
            (trans, rot) = lr.lookupTransform(TF_WORLD, TF_KINECT, rospy.Time(0))
            kinect_to_w  = lr.fromTranslationRotation(trans, rot)
            (trans, rot) = lr.lookupTransform(TF_WORLD, TF_WEBCAM, rospy.Time(0))
            webcam_to_w  = lr.fromTranslationRotation(trans, rot)
            print 'LookupTransform succeeded'
            break

        except tf.LookupException as e:
            print 'LookupTransform fail'
            rospy.sleep(1)


def send_matlab_floats(v):
    s = struct.pack('>%sf' % len(v), *v)
    conn.sendall(s)

def send_data_to_matlab():

    if (lefthand is None  or righthand is None or bins is None or workspace_bins is None):
        return

    global framecount
    framecount = framecount + 1

    # send hands
    p = lefthand.pose.position
    if lefthand.header.frame_id == TF_WORLD:
        p = [p.x, p.y, p.z, 1]
    else:
        p = kinect_to_w.dot([p.x, p.y, p.z, 1])
    send_matlab_floats([float(p[0]), float(p[1]), float(p[2])])
    p = righthand.pose.position
    if righthand.header.frame_id == TF_WORLD:
        p = [p.x, p.y, p.z, 1]
    else:
        p = kinect_to_w.dot([p.x, p.y, p.z, 1])
    send_matlab_floats([float(p[0]), float(p[1]), float(p[2])])

    # send bins
    for i in range(1, BIN_NUM+1):
        
        bin_exist = False

        for bin in bins:
            if bin.id == i:

                bin_exist = True

                p = bin.pose.pose.position
                p = webcam_to_w.dot([p.x, p.y, p.z, 1])
                o = bin.pose.pose.orientation
                o = tf.transformations.quaternion_multiply(tf.transformations.quaternion_from_matrix(webcam_to_w), [o.x, o.y, o.z, o.w])
                send_matlab_floats([float(p[0]), float(p[1]), float(p[2]), o[0], o[1], o[2], o[3]])              

        if bin_exist == False:
            send_matlab_floats([0, 0, 0, 0, 0, 0, 0])

    # send workspace bins
    send_matlab_floats([float(len(workspace_bins))] + 
                       [float(ord(ws_bin)) for ws_bin in workspace_bins] +
                       [float(0) for i in range(len(workspace_bins),MAX_WS_BINS)])

    # send additional xml string
    additional_data = etree.Element('data')
    additional_data.append(etree.Element('current_action_name'))
    additional_data[0].text = action_name
    s = etree.tostring(additional_data)
    conn.sendall(struct.pack('>i', len(s)))
    conn.sendall(s)


#####################################################
# get data from Matlab & publish
#####################################################

multiarray_float_pub = None
distributions = {}

multiarray_float_msg = Float32MultiArray()
multiarray_float_msg.layout.dim.append(MultiArrayDimension())
multiarray_float_msg.layout.dim.append(MultiArrayDimension())
multiarray_float_msg.layout.dim[0].label = 'x'
multiarray_float_msg.layout.dim[1].label = 'y'
multiarray_float_msg.layout.dim[0].size = BIN_NUM
multiarray_float_msg.layout.dim[1].size = T
multiarray_float_msg.data = [0] * (multiarray_float_msg.layout.dim[0].size * multiarray_float_msg.layout.dim[1].size)

bin_start_2_distribution_name = [0] * BIN_NUM
bin_end_2_distribution_name   = [0] * BIN_NUM

bin_start_2_distribution_name[3]	= 'Body_start'
bin_end_2_distribution_name[3]   	= 'body6_start'
bin_start_2_distribution_name[11]	= 'Nose_A_start'
bin_end_2_distribution_name[11]   	= 'nose_a4_start'
bin_start_2_distribution_name[10]	= 'Nose_H_start'
bin_end_2_distribution_name[10]   	= 'nose_h3_start'
bin_start_2_distribution_name[12]	= 'Wing_AT_start'
bin_end_2_distribution_name[12]   	= 'wing_at3_start'
bin_start_2_distribution_name[2]	= 'Wing_AD_start'
bin_end_2_distribution_name[2]   	= 'wing_ad4_start'
bin_start_2_distribution_name[7]	= 'Wing_H_start'
bin_end_2_distribution_name[7]   	= 'wing_h6_start'
bin_start_2_distribution_name[14]	= 'Tail_AT_start'
bin_end_2_distribution_name[14]   	= 'tail_at3_start'
bin_start_2_distribution_name[15]	= 'Tail_AD_start'
bin_end_2_distribution_name[15]   	= 'tail_ad4_start'
bin_start_2_distribution_name[13]	= 'Tail_H_start'
bin_end_2_distribution_name[13]   	= 'tail_h6_start'



bin_inference_pub = None

bin_inference_msg 			= BinInference()
bin_inference_msg.header.seq 		= -1
bin_inference_msg.header.stamp 		= rospy.Time()
bin_inference_msg.header.frame_id 	= "None"
bin_inference_msg.start_time 		= rospy.Time()
bin_inference_msg.distributions 	= [0] * (2 * BIN_NUM * T)
bin_inference_msg.T_len			= T
bin_inference_msg.period		= 1.0 / 30 * 7
bin_inference_msg.t_cur			= -1


discription = '>'
while (len(discription) <= T):
    discription = discription + 'f'
pubs = {}


def get_pub(key):
    if (pubs.has_key(key) == False):
        pubs[key] = rospy.Publisher('/inference/' + key, numpy_msg(Floats))

    return pubs[key];


def get_floats(i):
    data = ''
    while (len(data) < 4 * i):
        moredata = conn.recv(4 * i - len(data))
        data = data + moredata

        if (len(moredata) == 0):
            #print 'exit'
            #conn.close()
            #s.close()
            #sys.exit()
            raise Exception('spam', 'eggs')
            
    data = struct.unpack(discription, data)
    return data;

def get_name():
    data = ''
    while (len(data) < MAX_NAME_LENGTH):
        moredata = conn.recv(MAX_NAME_LENGTH - len(data))
        data = data + moredata

        if (len(moredata) == 0):
            #print 'exit'
            #conn.close()
            #s.close()
            #sys.exit()
            raise Exception('spam', 'eggs')
            
    return data;

def publish_multifloat():

    #for i in range(BIN_NUM):
        #multiarray_float_msg.data[2*i*T+0:2*i*T+T]     = distributions[bin_start_2_distribution_name[i]]
        #multiarray_float_msg.data[2*i*T+T+0:2*i*T+T+T] = distributions[bin_end_2_distribution_name[i]]

    multiarray_float_pub.publish(multiarray_float_msg)

def check_and_publish_inference():
    while (True):
        
        # check data available
        try:
            conn.settimeout(0)
            conn.recv(1, socket.MSG_PEEK)
            conn.settimeout(None)
   
        except socket.error as msg:
            #print 'data not available'
            break;
        
        # get data
        name = get_name().replace(' ', '')
        data = get_floats(T)
        assert(len(data) == T)

        if name == 'S_end':
            print 'Inference received'
            global inference_num
            inference_num = inference_num + 1

        # publish individual
        if PUBLISH_INDIVIDUAL_DISTRIBUTION :
            get_pub(name).publish(numpy.array(data, dtype=numpy.float32))

        # publish all
        if PUBLISH_MULTIARRAY_ALLDISTRIBUTIONS :
          global multiarray_float_msg
          for i in range(1, BIN_NUM):
              if bin_start_2_distribution_name[i] == name :
                  k = 2*(i-1)*T
                  multiarray_float_msg.data[k+0:k+T] = data
              if bin_end_2_distribution_name[i] == name :
                  k = 2*(i-1)*T+T
                  multiarray_float_msg.data[k+0:k+T] = data

          if name == 'S_end':
             multiarray_float_pub.publish(multiarray_float_msg)

        # publish all
        if True :
          global bin_inference_msg 
          for i in range(1, BIN_NUM):
              if bin_start_2_distribution_name[i] == name :
                  k = 2*(i-1)*T
                  bin_inference_msg.distributions[k+0:k+T] = data
              if bin_end_2_distribution_name[i] == name :
                  k = 2*(i-1)*T+T
                  bin_inference_msg.distributions[k+0:k+T] = data

          if name == 'S_end':
             global bin_inference_msg
             bin_inference_msg.header.stamp = rospy.Time.now()
             bin_inference_msg.t_cur = ceil((rospy.Time.now() - bin_inference_msg.start_time).to_sec() / bin_inference_msg.period)
             assert(len(bin_inference_msg.distributions) == 2 * T * BIN_NUM)
             bin_inference_pub.publish(bin_inference_msg)
        

      
#####################################################
# MAIN
#####################################################

def main():

    set_up_server()

    # init
    rospy.init_node('inference_from_matlab')

    # subscribe
    rospy.Subscriber(ROS_TOPIC_BINMARKES, AlvarMarkers, bins_callback)
    rospy.Subscriber(ROS_TOPIC_LEFTHAND, PoseStamped, left_hand_callback)
    rospy.Subscriber(ROS_TOPIC_RIGHTHAND, PoseStamped, right_hand_callback)
    rospy.Subscriber(ROS_TOPIC_ACTION_NAME_GT, String, action_name_callback)
    rospy.Subscriber(ROS_TOPIC_WORKSPACE_BINS, UInt8MultiArray, workspace_bins_callback)

    # set up pub
    if PUBLISH_MULTIARRAY_ALLDISTRIBUTIONS :
        global multiarray_float_pub
        multiarray_float_pub = rospy.Publisher('inference/all_distributions', Float32MultiArray)

    global bin_inference_pub
    bin_inference_pub = rospy.Publisher('inference/all_distributions', BinInference)


    # publish time interval
    time_interval_length_pub = rospy.Publisher('inference/time_interval_length', std_msgs.msg.Duration, latch=True)
    time_interval_length_pub.publish(std_msgs.msg.Duration(rospy.Duration.from_sec(1.0 / FPS * DOWN_SAMPLING_RATE)))


    # look up transform of world <---> kinect & webcam
    rospy.sleep(0.1)   
    lookup_transforms()
    if rospy.is_shutdown():
         return

    # ready?
    # raw_input("ready?")
    print 'wait to receive first hand & bin msg'
    while lefthand is None  or righthand is None or bins is None:
        rospy.sleep(0.1)
    print 'ok start!'

    # begin_time
    begin_time  = rospy.Time.now().to_nsec()
    timepub     = rospy.Publisher('inference/begin_time', std_msgs.msg.Time, latch=True)

    timepub.publish(rospy.Time.now())
    global bin_inference_msg
    bin_inference_msg.start_time = rospy.Time.now()

    # variables for checking timing
    elapsedtime = 0
    waittime    = 0
    r           = rospy.Rate(FPS)

    task_completed_frame = -1

    # processing loop
    global running
    running = True
    while (rospy.is_shutdown() == False):
         
        # send & receive data
        send_data_to_matlab()
	check_and_publish_inference()

        # print 'Frame ', framecount, ', lefthand_msgnum ', lefthand_msgnum, ', righthand_msgnum ', righthand_msgnum, ', bin_msgnum ', bin_msgnum, 'Inference_num ', inference_num, ', ', (rospy.Time.now().to_nsec() / 1000000000.0)

        # sleep till next frame
        if USE_ROS_RATE:
            r.sleep()
        else:
            elapsedtime = rospy.Time.now().to_nsec() - begin_time
            waittime    = framecount * 1000000000 / FPS - elapsedtime;
            rospy.sleep(waittime / 1000000000.0)

        # check for the end
        if framecount >= (T - 20) * DOWN_SAMPLING_RATE:
            print 'Max time exceeded. Quit inference_from_matlab'
            break
        if action_name == 'Complete' and task_completed_frame < 0:
            task_completed_frame = framecount
        if task_completed_frame > 0 and framecount - task_completed_frame > 100:
            print 'Task completed. Quit inference_from_matlab'
            break
      


    running = False

    # final sumary
    print 'Frame ', framecount 
    print 'Time ', ((rospy.Time.now().to_nsec() - begin_time) / 1000000000.0)
    print 'FPS ', framecount  / ((rospy.Time.now().to_nsec() - begin_time) / 1000000000.0)





if __name__ == '__main__':

     try:
        main()
     except Exception as e:
        print e
     finally:
        conn.sendall('exit!'); # exist signal
        rospy.sleep(2)
        conn.close()
        s.close()
        sys.exit()











