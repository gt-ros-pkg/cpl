#! /usr/bin/python

import numpy as np
import yaml
import sys

import roslib
roslib.load_manifest("ur_cart_move")

import rospy
import roslaunch.substitution_args

from ur_cart_move.ur_cart_move import RAVEKinematics, ArmInterface, ArmBehaviors

def main():
    if len(sys.argv) < 2:
        print 'Need filename'
        return
    rospy.init_node("save_joint_configs")
    robot_descr = roslaunch.substitution_args.resolve_args('$(find ur10_description)/ur10_robot.dae')
    arm = ArmInterface(timeout=0.)
    kin = RAVEKinematics(robot_descr)
    if not arm.wait_for_states(timeout=5.):
        print 'arm not connected!'
        return
    arm_behav = ArmBehaviors(arm, kin)
    f = file(sys.argv[1], 'r')
    qs = yaml.safe_load(f.read())
    print qs
    f.close()
    for q in qs:
        if rospy.is_shutdown():
            break
        arm_behav.move_to_q(q, velocity=0.1)

if __name__ == "__main__":
    main()
