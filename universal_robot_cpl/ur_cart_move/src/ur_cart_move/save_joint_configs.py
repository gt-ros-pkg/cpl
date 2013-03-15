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
    f = file(sys.argv[1], 'w')
    qs = []
    while not rospy.is_shutdown():
        if raw_input() == 'q':
            break
        q = arm.get_q()
        print q
        qs.append(q.tolist())
    f.write(yaml.safe_dump(qs))
    f.close()

if __name__ == "__main__":
    main()
