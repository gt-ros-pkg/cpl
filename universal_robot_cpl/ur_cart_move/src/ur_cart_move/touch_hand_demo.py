#! /usr/bin/python

import numpy as np
import sys

import roslib
roslib.load_manifest("ur_cart_move")

import rospy
import roslaunch.substitution_args
import tf

from hrl_geom.pose_converter import PoseConv
from hrl_geom.transformations import rotation_from_matrix as mat_to_ang_axis_point
from hrl_geom.transformations import rotation_matrix as ang_axis_point_to_mat
from hrl_geom.transformations import euler_matrix
from ur_controller_manager.msg import URJointCommand, URModeStates, URJointStates

from ur_cart_move.ur_cart_move import RAVEKinematics, ArmInterface, ArmBehaviors
def load_ur_robot():
    robot_descr = roslaunch.substitution_args.resolve_args('$(find ur10_description)/ur10_robot.dae')
    arm = ArmInterface(timeout=0.)
    kin = RAVEKinematics(robot_descr)
    if not arm.wait_for_states(timeout=5.):
        print 'arm not connected!'
        return
    #print arm.get_q()
    arm_behav = ArmBehaviors(arm, kin)
    return arm, kin, arm_behav

class PersonMonitor(object):
    def __init__(self, body_part):
        self.tf_list = tf.TransformListener()
        self.body_part = body_part

    def get_transforms(self):
        person_trans, person_ids = [], []
        for person_id in range(5):
            body_frame = '/%s_%d' % (self.body_part, person_id)
            try:
                person_trans.append(
                    PoseConv.to_homo_mat(
                        self.tf_list.lookupTransform('/base_link', body_frame, rospy.Time(0))))
                person_ids.append(person_id)
            except:
                pass
        return person_trans, person_ids

def main():
    np.set_printoptions(precision=4, suppress=True)

    rospy.init_node("ur_cart_move")
    arm, kin, arm_behav = load_ur_robot()
    p = np.mat([0.5, 0., 0.])
    print kin.inverse_pos(p)
    return
    #arm_behav.move_to_x(x_final, velocity=0.1, blend_delta=0.2)
    person_mon = PersonMonitor('right_hand')
    r = rospy.Rate(1)
    while not rospy.is_shutdown():
        person_trans, person_ids = person_mon.get_transforms()
        if len(person_trans) > 0:
            x_person = person_trans[0]
            q_cur = arm.get_q()
            x_cur = kin.forward(q_cur)
            x_goal = x_cur.copy()
            p_person = x_person[:3,3]
            u_person = p_person / np.linalg.norm(p_person)
            x_goal[:3,3] = 0.8 * u_person
            q_goal = kin.inverse(x_goal, q_cur)
            if q_goal is not None and np.max(np.fabs(q_goal-q_cur)) < 1000.0:
                print q_goal
                print x_goal
                print q_goal - q_cur
                arm_behav.move_to_q(q_goal, velocity=0.05)
                rospy.sleep(2.)
            else:
                print 'No solution'
        r.sleep()

if __name__ == "__main__":
    main()
