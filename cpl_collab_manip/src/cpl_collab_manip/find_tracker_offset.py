#!/usr/bin/python
import numpy as np

import roslib
roslib.load_manifest("hrl_geom")

import rospy
from geometry_msgs.msg import PoseStamped
from hrl_geom.pose_converter import PoseConv



class FindTrackerOffset(object):
    def __init__(self):
        rospy.Subscriber("/left_hand", PoseStamped, self.left_cb)
        rospy.Subscriber("/right_hand", PoseStamped, self.right_cb)
        rospy.Subscriber("/left_truth", PoseStamped, self.left_truth_cb)
        rospy.Subscriber("/right_truth", PoseStamped, self.right_truth_cb)
        self.hist_size = 10
        self.l_hist = np.zeros((self.hist_size,3))
        self.r_hist = np.zeros((self.hist_size,3))
        self.l_hist_ind = 0
        self.r_hist_ind = 0
        self.cur_left_hand = np.ones(3)
        self.cur_right_hand = np.ones(3)
        self.cur_left_truth = np.ones(3)
        self.cur_right_truth = np.ones(3)

    def left_cb(self, msg):
        self.cur_left_hand = np.array(PoseConv.to_pos_quat(msg)[0])

    def right_cb(self, msg):
        self.cur_right_hand = np.array(PoseConv.to_pos_quat(msg)[0])

    def left_truth_cb(self, msg):
        self.cur_left_truth = np.array(PoseConv.to_pos_quat(msg)[0])

    def right_truth_cb(self, msg):
        self.cur_right_truth = np.array(PoseConv.to_pos_quat(msg)[0])
        self.update()

    def update(self):
        l = self.cur_left_hand
        r = self.cur_right_hand
        lt = self.cur_left_truth
        rt = self.cur_right_truth
        dist_l_lt = np.linalg.norm(l-lt)
        dist_l_rt = np.linalg.norm(l-rt)
        dist_r_lt = np.linalg.norm(r-lt)
        dist_r_rt = np.linalg.norm(r-rt)
        min_dist = 0.15
        offset_val = [0.025, 0.016, -0.03]
        if dist_l_lt < dist_l_rt and dist_l_lt < min_dist:
            self.l_hist[self.l_hist_ind,:] = l-lt - offset_val
        elif dist_l_rt < min_dist:
            self.l_hist[self.l_hist_ind,:] = l-rt - offset_val
        if dist_r_lt < dist_r_rt and dist_r_lt < min_dist:
            self.r_hist[self.r_hist_ind,:] = r-lt - offset_val
        elif dist_r_rt < min_dist:
            self.r_hist[self.r_hist_ind,:] = r-rt - offset_val

        self.l_hist_ind += 1
        self.r_hist_ind += 1
        if self.l_hist_ind >= self.hist_size:
            self.l_hist_ind = 0
            self.r_hist_ind = 0
        print np.mean(self.l_hist,0), np.mean(self.r_hist,0)

def main():
    rospy.init_node("find_tracker_offset")

    fto = FindTrackerOffset()
    rospy.spin()

if __name__ == "__main__":
    main()
