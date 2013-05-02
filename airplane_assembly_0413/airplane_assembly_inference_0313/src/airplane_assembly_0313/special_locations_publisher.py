#!/usr/bin/env python
import rospy

import roslib
roslib.load_manifest('airplane_assembly_0313')
import tf

from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import MultiArrayDimension

import array
import numpy

import socket
import time
import sys
import pylab
import struct

from geometry_msgs.msg import *
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from ar_track_alvar.msg import *

RATE     = 30
TF_WORLD = "base_link"


slocations = [ 
         { 'name' : 'L0' , 'position' : (-0.673186, -1.09836, -0.166633) , 'orientation' : (0.000564658, 0.00105971, -0.458399, 0.888746)},
         { 'name' : 'L11' , 'position' : (-0.51749, -0.779265, -0.0629973) , 'orientation' : (0.00204842, -0.0234259, -0.446821, 0.894314)},
         { 'name' : 'L12' , 'position' : (-0.408021, -0.921468, -0.0661017) , 'orientation' : (-0.00895636, -0.00378763, -0.471837, 0.881632)},
         { 'name' : 'L13' , 'position' : (-0.314413, -1.0677, -0.0705261) , 'orientation' : (0.00391229, -0.000605329, -0.475785, 0.879553)},

         { 'name' : 'L21' , 'position' : (-0.241755, -0.479651, -0.0566505) , 'orientation' : (0.0145793, -0.00559917, -0.479727, 0.877279)},
         { 'name' : 'L22' , 'position' : (-0.181439, -0.562769, -0.0398458) , 'orientation' : (0.0103439, -0.0097846, -0.476231, 0.879205)},
         { 'name' : 'L23' , 'position' : (-0.125371, -0.651957, -0.0474464) , 'orientation' : (0.00236152, -0.0141837, -0.470724, 0.882163)},
         { 'name' : 'L24' , 'position' : (-0.0678047, -0.743967, -0.0544225) , 'orientation' : (0.0045029, -0.0123035, -0.474586, 0.880111)},
         { 'name' : 'L25' , 'position' : (-0.0111795, -0.841586, -0.0736639) , 'orientation' : (0.000220135, 0.00643288, -0.47017, 0.882553)},
         { 'name' : 'L26' , 'position' : (0.0552281, -0.954989, -0.0835534) , 'orientation' : (-0.0100673, 0.0179283, -0.493724, 0.869376)},

         { 'name' : 'L31' , 'position' : (0.0798513, -0.29116, -0.0505636) , 'orientation' : (0.0261526, -0.0297672, -0.488093, 0.871892)},
         { 'name' : 'L32' , 'position' : (0.137721, -0.378104, -0.0498443) , 'orientation' : (0.0227562, -0.0148598, -0.487837, 0.872511)},
         { 'name' : 'L33' , 'position' : (0.189021, -0.468398, -0.0502239) , 'orientation' : (-0.00248678, -0.0135991, -0.491639, 0.87069)},
         { 'name' : 'L34' , 'position' : (0.245162, -0.559758, -0.0608967) , 'orientation' : (-0.00410395, -0.0150506, -0.483665, 0.875114)},
         { 'name' : 'L35' , 'position' : (0.300318, -0.65233, -0.0543972) , 'orientation' : (-0.00915256, -0.00879721, -0.487721, 0.872907)},
         { 'name' : 'L36' , 'position' : (0.354257, -0.729629, -0.0317831) , 'orientation' : (0.0146889, -0.0318996, -0.481777, 0.87559)}

         ]


#####################################################
# GET SPECIAL LOCATIONS
#####################################################

def get_special_locations():
      return slocations

def get_special_location_by_name(name) :
    for slocation in slocations:
        if slocation['name'] == name :
             return slocation['position'], slocation['orientation']
    return False, False

            
#####################################################
# MAIN
#####################################################

if __name__ == '__main__':

    # init 
    rospy.init_node('special_locations_publisher')
    
    br = tf.TransformBroadcaster()

    r = rospy.Rate(RATE)

    while not rospy.is_shutdown() :
        for slocation in slocations:

            br.sendTransform( slocation['position'], slocation['orientation'], rospy.Time.now(), slocation['name'], TF_WORLD)

        r.sleep()












