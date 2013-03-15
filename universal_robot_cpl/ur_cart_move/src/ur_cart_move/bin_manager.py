#! /usr/bin/python

import numpy as np

import roslib
roslib.load_manifest("ur_cart_move")
roslib.load_manifest("ar_track_alvar")
roslib.load_manifest('tf')
import rospy

import tf
from ar_track_alvar.msg import AlvarMarkers
from ur_cart_move.srv import MoveBin, MoveBinResponse
from hrl_geom.pose_converter import PoseConv

class BinManager(object):
    def __init__(self):
        self.ar_sub = rospy.Subscriber("/ar_pose_marker", AlvarMarkers, 
                                       self.ar_cb)
        self.ar_poses = {}
        self.move_bin = rospy.ServiceProxy('/move_bin', MoveBin)
        tf_list = tf.TransformListener()
        r = rospy.Rate(100)
        while not rospy.is_shutdown():
            try:
                pose = tf_list.lookupTransform('/base_link', '/lifecam1_optical_frame', 
                                               rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
            r.sleep()
        self.camera_pose = PoseConv.to_homo_mat(pose)
        self.move_bin.wait_for_service()

    def ar_cb(self, msg):
        cur_time = rospy.get_time()
        for marker in msg.markers:
            marker.pose.header = marker.header
            self.ar_poses[marker.id] = [cur_time, marker.pose]

    def do_thing(self):
        r = rospy.Rate(100)
        while not rospy.is_shutdown():
            raw_input()
            grasp_tag = self.camera_pose * PoseConv.to_homo_mat(self.ar_poses[5][1])
            place_tag = self.camera_pose * PoseConv.to_homo_mat(self.ar_poses[15][1])
            self.move_bin(PoseConv.to_pose_stamped_msg("lifecam1_optical_frame", grasp_tag), 
                          PoseConv.to_pose_stamped_msg("lifecam1_optical_frame", place_tag))
            r.sleep()

def main():
    rospy.init_node("bin_manager")
    bm = BinManager()
    bm.do_thing()
    rospy.spin()

if __name__ == "__main__":
    main()
