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

ROS_TOPIC_BINMARKES       = "ar_pose_marker"
ROS_TOPIC_BINMARKES_CLEAN = "ar_pose_marker_clean"


TF_WORLD     = "base_link"
TF_KINECT    = "kinect1_rgb_optical_frame"
TF_WEBCAM    = "lifecam1_optical_frame"

BIN_NUM      = 20
MAX_WS_BINS  = 20

T            = 1000

FPS          = 50


#####################################################
# Update hands & bin data
#####################################################


webcam_to_w      = 0
kinect_to_w      = 0

bins             = None
bins_clean       = None
    
def lookup_transforms():
    return
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


def bins_callback(data):
    global bins
    bins = data.markers

def bins_callback_clean(data):
    global bins_clean
    bins_clean = data.markers



def publish_clean_tf():
    global bins
    global bins_clean

    if bins is None or bins_clean is None:
        return

    for bin in bins_clean:
        
        fake_bin = True

        for bin_real in bins:
            if bin_real.id == bin.id:

                fake_bin = False

        if fake_bin or True: # publish tf
            br = tf.TransformBroadcaster()
            br.sendTransform((bin.pose.pose.position.x, bin.pose.pose.position.y, bin.pose.pose.position.z), (bin.pose.pose.orientation.x, bin.pose.pose.orientation.y, bin.pose.pose.orientation.z, bin.pose.pose.orientation.w), rospy.Time.now(), "ar_" + str(bin.id) + "_clean",  "lifecam1_optical_frame") 
           
#####################################################
# MAIN
#####################################################

def main():


    # init
    rospy.init_node('show_tf_for_ar_clean')

    # subscribe
    rospy.Subscriber(ROS_TOPIC_BINMARKES, AlvarMarkers, bins_callback)
    rospy.Subscriber(ROS_TOPIC_BINMARKES_CLEAN, AlvarMarkers, bins_callback_clean)

    # look up transform of world <---> kinect & webcam
    rospy.sleep(0.1)   
    lookup_transforms()
    if rospy.is_shutdown():
         return


    # processing loop
    r = rospy.Rate(FPS)
    while (rospy.is_shutdown() == False):
         
	publish_clean_tf()

        # sleep till next frame
        if USE_ROS_RATE:
            r.sleep()



if __name__ == '__main__':

     try:
        main()
     except Exception as e:
        print e
     finally:
        sys.exit()











