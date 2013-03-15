#!/usr/bin/python

import numpy as np
from collections import defaultdict

import roslib
roslib.load_manifest('ur_cart_move')

import rospy

from ur_cart_move.ur_cart_move import load_ur_robot
from hrl_geom.pose_converter import PoseConv

from openravepy import *

def main():
    np.set_printoptions(precision=4, suppress=True)
    rospy.init_node("test_collision")
    test_prefix = rospy.get_param("~test_prefix", "")
    arm, kin, arm_behav = load_ur_robot(timeout=1., topic_prefix=test_prefix)

    env, robot, manip = kin.env, kin.robot, kin.manip
    last_time = rospy.get_time()
    n = 0
    period = 1.
    r = rospy.Rate(10)
    kin.env.SetViewer('qtcoin')
    while not rospy.is_shutdown():
        q = arm.get_q()
        kin.robot.SetDOFValues(q, kin.manip.GetArmIndices())
        kin.env.UpdatePublishedBodies()
        #q = np.random.rand(6)*np.pi*2 - np.pi
        if kin.is_self_colliding(q):
            print "SELF COLLISION", q
        #print result, q
        #if raw_input() == 'q':
        #    break
        #if rospy.get_time() - last_time > period:
        #    print n / period
        #    n = 0
        #    last_time = rospy.get_time()
        #n += 1
        r.sleep()


if __name__ == "__main__":
    main()
