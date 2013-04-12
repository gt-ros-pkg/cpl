#! /usr/bin/python

import rospy
import roslib
roslib.load_manifest('utilities_aa0413')
import tf

from ar_track_alvar.msg import *

import utilities_aa0413.locations
import random

PUB_RATE = 10
REF_FRAME = '/lifecam1_optical_frame'

LOCATION_NOISE = 0.0001 

####################################################
# create msg
####################################################


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

msg 			= AlvarMarkers()
msg.header.frame_id 	= REF_FRAME
msg.markers 		= []

for bin in bins_locs:

    marker 			= AlvarMarker()
    marker.header.frame_id 	= REF_FRAME
    marker.pose.header.frame_id = REF_FRAME
    marker.id 			= bin['bin_id']

    l, o = utilities_aa0413.locations.get_location_by_name(bin['location'])

    marker.pose.pose.position.x = l[0]
    marker.pose.pose.position.y = l[1]
    marker.pose.pose.position.z = l[2]

    marker.pose.pose.orientation.x = o[0]
    marker.pose.pose.orientation.y = o[1]
    marker.pose.pose.orientation.z = o[2]
    marker.pose.pose.orientation.w = o[3]

    msg.markers.append(marker)
            


####################################################
# main
####################################################

def the_cb(m):
    global msg
    msg = m

def main():

    print 'running ....'

    rospy.init_node('nam_ar_tag_tracker_sim')

    sub = rospy.Subscriber("update_ar_pose_marker", AlvarMarkers, the_cb)

    pub = rospy.Publisher('ar_pose_marker', AlvarMarkers)    

    br = tf.TransformBroadcaster()

    r = rospy.Rate(PUB_RATE)

    while not rospy.is_shutdown():
         
        # create msg
        sentmsg = msg
        sentmsg.header.stamp = rospy.Time.now()

        for i in range(len(sentmsg.markers)):
           sentmsg.markers[i].header.stamp = rospy.Time.now()
           sentmsg.markers[i].pose.header.stamp = rospy.Time.now()
           
           sentmsg.markers[i].pose.pose.position.x = sentmsg.markers[i].pose.pose.position.x + random.gauss(0.0, LOCATION_NOISE)
           sentmsg.markers[i].pose.pose.position.y = sentmsg.markers[i].pose.pose.position.y + random.gauss(0.0, LOCATION_NOISE)
           sentmsg.markers[i].pose.pose.position.z = sentmsg.markers[i].pose.pose.position.z + random.gauss(0.0, LOCATION_NOISE)
           sentmsg.markers[i].pose.pose.orientation.x = sentmsg.markers[i].pose.pose.orientation.x + random.gauss(0.0, LOCATION_NOISE)
           sentmsg.markers[i].pose.pose.orientation.y = sentmsg.markers[i].pose.pose.orientation.y + random.gauss(0.0, LOCATION_NOISE)
           sentmsg.markers[i].pose.pose.orientation.z = sentmsg.markers[i].pose.pose.orientation.z + random.gauss(0.0, LOCATION_NOISE)
           sentmsg.markers[i].pose.pose.orientation.w = sentmsg.markers[i].pose.pose.orientation.w + random.gauss(0.0, LOCATION_NOISE)
           
        

        # broadcast tf
        for m in sentmsg.markers:
            br.sendTransform( (m.pose.pose.position.x,m.pose.pose.position.y,m.pose.pose.position.z), 
                              (m.pose.pose.orientation.x,m.pose.pose.orientation.y,m.pose.pose.orientation.z,m.pose.pose.orientation.w), 
                              rospy.Time.now(), 
                              'ar_marker_'+str(m.id), 
                               REF_FRAME)

        pub.publish(sentmsg)

        r.sleep()


if __name__ == '__main__' :
    main()























































































