#!/usr/bin/python

import numpy as np

import roslib
roslib.load_manifest('tf')
roslib.load_manifest('hrl_geom')

import rospy
import tf
from geometry_msgs.msg import PoseStamped

from hrl_geom.pose_converter import PoseConv

def main():
    rospy.init_node('openni_tracker_repub')

    repub_rate = rospy.get_param('~repub_rate', 30)

    tf_list = tf.TransformListener()
    sides = ['right', 'left']
    pubs = {}
    for side in sides:
        pubs[side] = rospy.Publisher('/%s_hand' % side, PoseStamped)
    r = rospy.Rate(repub_rate)
    last_id = 1
    while not rospy.is_shutdown():
        potential_ids = np.concatenate(([last_id], np.setdiff1d(range(1,6), [last_id])))
        #potential_ids = [1]
        for user_id in potential_ids:
            found_hands = 0
            for side in sides:
                try:
                    #now = rospy.Time.now()
                    from_frame = '/kinect0_rgb_optical_frame'
                    to_frame = '/%s_hand_%d' % (side, user_id)
                    #tf_list.waitForTransform(from_frame, to_frame, rospy, rospy.Duration(1.))
                    tf_pose = tf_list.lookupTransform(from_frame, to_frame, rospy.Time())
                    ps_msg = PoseConv.to_pose_stamped_msg(from_frame, tf_pose)
                    pubs[side].publish(ps_msg)
                    found_hands += 1
                except Exception as e:
                    #print e
                    break
            if found_hands == 2:
                last_id = user_id
                break
        r.sleep()

if __name__ == '__main__':
    main()
