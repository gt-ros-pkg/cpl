#! /usr/bin/python

import numpy as np
import sys

import roslib
roslib.load_manifest("ur_cart_move")
roslib.load_manifest("openni_tracker_msgs")

import rospy
import roslaunch.substitution_args
import tf

from hrl_geom.pose_converter import PoseConv
from hrl_geom.transformations import rotation_from_matrix as mat_to_ang_axis_point
from hrl_geom.transformations import rotation_matrix as ang_axis_point_to_mat
from hrl_geom.transformations import euler_matrix
from ur_controller_manager.msg import URJointCommand, URModeStates, URJointStates

from ur_cart_move.ur_cart_move import load_ur_robot
from openni_tracker_msgs.msg import skeletonData

class PersonMonitor(object):
    def __init__(self, joint_ids, timeout=1.0):
        self.tf_list = tf.TransformListener()
        self.skel_sub = rospy.Subscriber('/skeleton_data', skeletonData, self.skel_cb)
        self.joint_ids = joint_ids
        self.homo_poses = [[None, None], [None, None]]
        self.timeout = timeout
        self.last_times = [[-100, -100], [-100, -100]]

    def skel_cb(self, msg):
        for joint in msg.joints:
            for i, joint_id in enumerate(self.joint_ids):
                if joint.jointID == joint_id:
                    try:
                        joint_pose = PoseConv.to_pose_stamped_msg(msg.header.frame_id, joint.pose)
                        joint_pose.header.stamp = rospy.Time()
                        tf_pose = self.tf_list.transformPose('/base_link', joint_pose)
                        self.homo_poses[msg.kinectID][i] = PoseConv.to_homo_mat(tf_pose)
                        self.last_times[msg.kinectID][i] = rospy.get_time()
                    except Exception as e:
                        #print e
                        pass

    def get_transform(self):
        #print self.last_times
        now_time = rospy.get_time()
        active_poses = []
        x_vals = []
        for i in range(2):
            for j in range(2):
                if now_time - self.last_times[i][j] < self.timeout:
                    active_poses.append(self.homo_poses[i][j])
                    x_vals.append(self.homo_poses[i][j][0,3])
        if len(active_poses) == 0:
            return None
        return active_poses[np.argmax(x_vals)]

def main():
    np.set_printoptions(precision=4, suppress=True)

    rospy.init_node("ur_cart_move")
    arm, kin, arm_behav = load_ur_robot()
    #arm_behav.move_to_x(x_final, velocity=0.1, blend_delta=0.2)
    q_lwr_delta = np.pi/180.0 * np.array([30., 50., 30., 70., 90., 150.])
    q_upr_delta = q_lwr_delta
    person_mon = PersonMonitor([9, 15]) # head 0, left 9, right 15
    r = rospy.Rate(1)
    while not rospy.is_shutdown():
        person_trans = person_mon.get_transform()
        print person_trans
        #r.sleep()
        #continue
        if person_trans is not None:
            x_person = person_trans
            q_cur = arm.get_q()
            x_cur = kin.forward(q_cur)
            x_goal = x_cur.copy()
            p_person = x_person[:3,3]
            u_person = p_person / np.linalg.norm(p_person)

            # the position goal is the person, minus 10 cm towards the base
            x_goal[:3,3] = p_person - 0.1 * u_person
            
            # rotate a frame so the x-axis is pointing towards the person
            z_ang = np.arctan2(p_person[1,0], p_person[0,0])
            x_goal[:3,:3] = np.mat(euler_matrix(0., 0., z_ang))[:3,:3]

            print 'x_goal'
            print x_goal
            print 'x_cur'
            print x_cur
            #q_goal = kin.inverse(x_goal, q_cur,
            #                     q_min=q_cur-q_lwr_delta, q_max=q_cur+q_upr_delta)
            arm.unlock_security_stop()
            q_goal = kin.inverse_rand_search(x_goal, q_cur, 
                                             pos_tol=0.02, 
                                             rot_tol=np.array([10,10,10])*np.pi/180.0,
                                             q_min=q_cur-q_lwr_delta, q_max=q_cur+q_upr_delta)
            if q_goal is not None:
                print q_goal
                print x_goal
                print q_goal - q_cur
                q_goal[5] = q_cur[5]
                arm_behav.move_to_q(q_goal, velocity=0.05*4)
                rospy.sleep(0.1)
            else:
                pass
                #print 'No solution'
        r.sleep()

if __name__ == "__main__":
    main()
