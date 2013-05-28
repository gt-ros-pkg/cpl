#! /usr/bin/python

import rospy
import roslib
roslib.load_manifest('utilities_aa0413')
import tf


from ar_track_alvar.msg import *

import utilities_aa0413.ar_tag_tracker
import utilities_aa0413.hand_tracker
import utilities_aa0413.locations


####################################################
# test
####################################################

def main():
    rospy.init_node('f64326346nf')

    utilities_aa0413.hand_tracker.init()
    utilities_aa0413.ar_tag_tracker.init()


    rospy.sleep(2)

    while not rospy.is_shutdown():

       #msg = utilities_aa0413.ar_tag_tracker.get_latest_msg()
       #msg.markers[1].pose.pose.position.x = msg.markers[1].pose.pose.position.x + 0.1
       #utilities_aa0413.ar_tag_tracker.sim_update(msg)
       m = utilities_aa0413.ar_tag_tracker.get_markers()
       mname = []
       for i in m:
          mname.append(i.id)
       print mname
       rospy.sleep(0.5)
    

if __name__ == '__main__' :
    main()























































































