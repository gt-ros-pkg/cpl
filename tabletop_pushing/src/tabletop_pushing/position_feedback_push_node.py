#!/usr/bin/env python
# Software License Agreement (BSD License)
#
#  Copyright (c) 2011, Georgia Institute of Technology
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#  * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#  * Neither the name of the Georgia Institute of Technology nor the names of
#     its contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  'AS IS' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import roslib; roslib.load_manifest('tabletop_pushing')
import rospy
import hrl_pr2_lib.pr2 as pr2
import hrl_lib.tf_utils as tfu
from hrl_pr2_arms.pr2_controller_switcher import ControllerSwitcher
from geometry_msgs.msg import PoseStamped
from pr2_controllers_msgs.msg import *
import tf
import numpy as np
from tabletop_pushing.srv import *
from math import sin, cos, pi, fabs
import sys

# Setup joints stolen from Kelsey's code.
LEFT_ARM_SETUP_JOINTS = np.matrix([[1.32734204881265387,
                                    -0.34601608409943324,
                                    1.4620635485239604,
                                    -1.2729772622637399,
                                    7.5123303230158518,
                                    -1.5570651396529178,
                                    -5.5929916630672727]]).T
RIGHT_ARM_SETUP_JOINTS = np.matrix([[-1.32734204881265387,
                                      -0.34601608409943324,
                                      -1.4620635485239604,
                                      -1.2729772622637399,
                                      -7.5123303230158518,
                                      -1.5570651396529178,
                                      -7.163787989862169]]).T
LEFT_ARM_READY_JOINTS = np.matrix([[0.42427649, 0.0656137,
                                    1.43411927, -2.11931035,
                                    -15.78839978, -1.64163257,
                                    -17.2947453]]).T
RIGHT_ARM_READY_JOINTS = np.matrix([[-0.42427649, 0.0656137,
                                     -1.43411927, -2.11931035,
                                     15.78839978, -1.64163257,
                                     8.64421842e+01]]).T
LEFT_ARM_PULL_READY_JOINTS = np.matrix([[0.42427649,
                                         -0.34601608409943324,
                                         1.43411927,
                                         -2.11931035,
                                         82.9984037,
                                         -1.64163257,
                                         -36.16]]).T
RIGHT_ARM_PULL_READY_JOINTS = np.matrix([[-0.42427649,
                                         -0.34601608409943324,
                                         -1.43411927,
                                         -2.11931035,
                                         -82.9984037,
                                         -1.64163257,
                                         54.8]]).T
LEFT_ARM_HIGH_PUSH_READY_JOINTS = np.matrix([[0.42427649,
                                              -0.34601608409943324,
                                              1.43411927,
                                              -2.11931035,
                                              -15.78839978,
                                              -1.64163257,
                                              -17.2947453]]).T
RIGHT_ARM_HIGH_PUSH_READY_JOINTS = np.matrix([[-0.42427649,
                                                -0.34601608409943324,
                                                -1.43411927,
                                                -2.11931035,
                                                15.78839978,
                                                -1.64163257,
                                                8.64421842e+01]]).T
LEFT_ARM_HIGH_SWEEP_READY_JOINTS = LEFT_ARM_HIGH_PUSH_READY_JOINTS
RIGHT_ARM_HIGH_SWEEP_READY_JOINTS = RIGHT_ARM_HIGH_PUSH_READY_JOINTS

class PositionFeedbackPushNode:

    def __init__(self):
        rospy.init_node('position_feedback_push_node', log_level=rospy.DEBUG)
        self.torso_z_offset = rospy.get_param('~torso_z_offset', 0.15)
        self.look_pt_x = rospy.get_param('~look_point_x', 0.45)
        self.head_pose_cam_frame = rospy.get_param('~head_pose_cam_frame',
                                                   'openni_rgb_frame')
        self.default_torso_height = rospy.get_param('~default_torso_height',
                                                    0.2)
        self.high_arm_init_z = rospy.get_param('~high_arm_start_z', 0.05)
        self.tf_listener = tf.TransformListener()

        # Set joint gains
        self.cs = ControllerSwitcher()
        self.arm_mode = None
        self.init_joint_controllers()
        self.init_cart_controllers()

        # Setup arms
        rospy.loginfo('Creating pr2 object')
        self.robot = pr2.PR2(self.tf_listener, arms=True, base=False,
                             use_kinematics=False)
        self.l_arm_cart_pub = rospy.Publisher(
            '/l_cart_posture_push/command_pose', PoseStamped)
        self.r_arm_cart_pub = rospy.Publisher(
            '/r_cart_posture_push/command_pose', PoseStamped)

        # Open callback services
        self.gripper_pre_push_srv = rospy.Service('gripper_pre_push',
                                                  GripperPush,
                                                  self.gripper_pre_push)
        # self.gripper_pre_sweep_srv = rospy.Service('gripper_pre_sweep',
        #                                            GripperPush,
        #                                            self.gripper_pre_sweep)
        # self.overhead_pre_push_srv = rospy.Service('overhead_pre_push',
        #                                            GripperPush,
        #                                            self.overhead_pre_push)
        self.raise_and_look_serice = rospy.Service('raise_and_look',
                                                   RaiseAndLook,
                                                   self.raise_and_look_action)


    #
    # Initialization functions
    #

    #
    # Arm pose initialization functions
    #
    def init_arm_pose(self, force_ready=False, which_arm='l'):
        '''
        Move the arm to the initial pose to be out of the way for viewing the
        tabletop
        '''
        if which_arm == 'l':
            robot_arm = self.robot.left
            robot_gripper = self.robot.left_gripper
            ready_joints = LEFT_ARM_READY_JOINTS
            setup_joints = LEFT_ARM_SETUP_JOINTS
        else:
            robot_arm = self.robot.right
            robot_gripper = self.robot.right_gripper
            ready_joints = RIGHT_ARM_READY_JOINTS
            setup_joints = RIGHT_ARM_SETUP_JOINTS

        self.switch_to_joint_controllers()
        rospy.loginfo('Closing %s_gripper' % which_arm)
        res = robot_gripper.close(block=True)
        rospy.loginfo('Closed %s_gripper' % which_arm)

        rospy.loginfo('Moving %s_arm to setup pose' % which_arm)
        robot_arm.set_pose(setup_joints, nsecs=2.0, block=True)
        rospy.loginfo('Moved %s_arm to setup pose' % which_arm)

    def reset_arm_pose(self, force_ready=False, which_arm='l',
                       high_arm_joints=False):
        '''
        Move the arm to the initial pose to be out of the way for viewing the
        tabletop
        '''
        if which_arm == 'l':
            robot_arm = self.robot.left
            robot_gripper = self.robot.left_gripper
            if high_arm_joints:
                ready_joints = LEFT_ARM_HIGH_PUSH_READY_JOINTS
            else:
                ready_joints = LEFT_ARM_READY_JOINTS
            setup_joints = LEFT_ARM_SETUP_JOINTS
        else:
            robot_arm = self.robot.right
            robot_gripper = self.robot.right_gripper
            if high_arm_joints:
                ready_joints = RIGHT_ARM_HIGH_PUSH_READY_JOINTS
            else:
                ready_joints = RIGHT_ARM_READY_JOINTS
            setup_joints = RIGHT_ARM_SETUP_JOINTS
        ready_diff = np.linalg.norm(pr2.diff_arm_pose(robot_arm.pose(),
                                                      ready_joints))

        # Choose to move to ready first, if it is closer, then move to init
        moved_ready = False
        if force_ready or ready_diff > READY_POSE_MOVE_THRESH or force_ready:
            rospy.loginfo('Moving %s_arm to ready pose' % which_arm)
            robot_arm.set_pose(ready_joints, nsecs=1.5, block=True)
            rospy.loginfo('Moved %s_arm to ready pose' % which_arm)
            moved_ready = True
        else:
            rospy.loginfo('Arm in ready pose')


        rospy.loginfo('Moving %s_arm to setup pose' % which_arm)
        robot_arm.set_pose(setup_joints, nsecs=1.5, block=True)
        rospy.loginfo('Moved %s_arm to setup pose' % which_arm)

    #
    # Behavior functions
    #
    def gripper_pre_push(self, request):
        response = GripperPushResponse()
        push_frame = request.start_point.header.frame_id
        rospy.loginfo('Have a request frame of: ' + str(push_frame));
        start_point = request.start_point.point
        wrist_yaw = request.wrist_yaw
        push_dist = request.desired_push_dist
        if request.left_arm:
            ready_joints = LEFT_ARM_READY_JOINTS
            if request.high_arm_init:
                ready_joints = LEFT_ARM_HIGH_PUSH_READY_JOINTS
            which_arm = 'l'
            push_arm_pub = self.l_arm_cart_pub
        else:
            ready_joints = RIGHT_ARM_READY_JOINTS
            if request.high_arm_init:
                ready_joints = RIGHT_ARM_HIGH_PUSH_READY_JOINTS
            which_arm = 'r'
            push_arm_pub = self.r_arm_cart_pub

        if request.arm_init:
            # TODO: Figure out if we want this
            pass
            # rospy.loginfo('Moving %s_arm to ready pose' % which_arm)

        start_pose = PoseStamped()
        start_pose.header = request.start_point.header
        start_pose.pose.position = request.start_point.point
        q = tf.transformations.quaternion_from_euler(0.0, 0.0, wrist_yaw)
        start_pose.pose.orientation.x = q[0]
        start_pose.pose.orientation.y = q[1]
        start_pose.pose.orientation.z = q[2]
        start_pose.pose.orientation.w = q[3]

        self.switch_to_cart_controllers()
        if request.high_arm_init:
            # TODO: Move to offset pose above the table
            start_pose.pose.position.z = self.high_arm_init_z
            push_arm_pub.publish(start_pose)
            # wait
            rospy.sleep(5.0)
            rospy.loginfo('Done moving to overhead start point')
            # Change z to lower arm to table
            start_pose.pose.position.z = request.start_point.point.z
        # Move to start pose
        push_arm_pub.publish(start_pose)
        # wait
        rospy.sleep(2.0)
        rospy.loginfo('Done moving to start point')
        return response


    def raise_and_look_action(self, request):
        '''
        Service callback to raise the spine to a specific height relative to the
        table height and tilt the head so that the Kinect views the table
        '''
        if request.init_arms:
            # TODO: Figure this out
            self.init_arms()

        if request.point_head_only:
            response = RaiseAndLookResponse()
            response.head_succeeded = self.init_head_pose(request.camera_frame)
            return response

        if not request.have_table_centroid:
            response = RaiseAndLookResponse()
            response.head_succeeded = self.init_head_pose(request.camera_frame)
            self.init_spine_pose()
            return response

        # Get torso_lift_link position in base_link frame
        (trans, rot) = self.tf_listener.lookupTransform('/base_link',
                                                        '/torso_lift_link',
                                                        rospy.Time(0))
        lift_link_z = trans[2]

        # tabletop position in base_link frame
        table_base = self.tf_listener.transformPose('/base_link',
                                                    request.table_centroid)
        table_z = table_base.pose.position.z
        goal_lift_link_z = table_z + self.torso_z_offset
        lift_link_delta_z = goal_lift_link_z - lift_link_z
        rospy.loginfo('Torso height (m): ' + str(lift_link_z))
        rospy.loginfo('Table height (m): ' + str(table_z))
        rospy.loginfo('Torso goal height (m): ' + str(goal_lift_link_z))
        rospy.loginfo('Torso delta (m): ' + str(lift_link_delta_z))

        # Set goal height based on passed on table height
        # TODO: Set these better
        torso_max = 0.3
        torso_min = 0.01
        current_torso_position = np.asarray(self.robot.torso.pose()).ravel()[0]
        torso_goal_position = current_torso_position + lift_link_delta_z
        torso_goal_position = (max(min(torso_max, torso_goal_position),
                                   torso_min))
        rospy.loginfo('Moving torso to ' + str(torso_goal_position))
        # Multiply by 2.0, because of units of spine
        self.robot.torso.set_pose(torso_goal_position)

        rospy.loginfo('Got torso client result')
        new_torso_position = np.asarray(self.robot.torso.pose()).ravel()[0]
        rospy.loginfo('New torso position is: ' + str(new_torso_position))

        # Get torso_lift_link position in base_link frame
        (new_trans, rot) = self.tf_listener.lookupTransform('/base_link',
                                                            '/torso_lift_link',
                                                            rospy.Time(0))
        new_lift_link_z = new_trans[2]
        rospy.loginfo('New Torso height (m): ' + str(new_lift_link_z))
        # tabletop position in base_link frame
        new_table_base = self.tf_listener.transformPose('/base_link',
                                                        request.table_centroid)
        new_table_z = new_table_base.pose.position.z
        rospy.loginfo('New Table height: ' + str(new_table_z))

        # Point the head at the table centroid
        # NOTE: Should we fix the tilt angle instead for consistency?
        look_pt = np.asmatrix([self.look_pt_x,
                               0.0,
                               -self.torso_z_offset])
        rospy.loginfo('Point head at ' + str(look_pt))
        head_res = self.robot.head.look_at(look_pt,
                                           request.table_centroid.header.frame_id,
                                           request.camera_frame)
        response = RaiseAndLookResponse()
        if head_res:
            rospy.loginfo('Succeeded in pointing head')
            response.head_succeeded = True
        else:
            rospy.loginfo('Failed to point head')
            response.head_succeeded = False
        return response

    def init_head_pose(self, camera_frame):
        look_pt = np.asmatrix([self.look_pt_x, 0.0, -self.torso_z_offset])
        rospy.loginfo('Point head at ' + str(look_pt))
        head_res = self.robot.head.look_at(look_pt,
                                           'torso_lift_link',
                                           camera_frame)
        if head_res:
            rospy.loginfo('Succeeded in pointing head')
            return True
        else:
            rospy.loginfo('Failed to point head')
            return False

    def init_spine_pose(self):
        rospy.loginfo('Setting spine height to '+str(self.default_torso_height))
        self.robot.torso.set_pose(self.default_torso_height)
        new_torso_position = np.asarray(self.robot.torso.pose()).ravel()[0]
        rospy.loginfo('New spine height is ' + str(new_torso_position))

    def init_arms(self):
        self.init_arm_pose(True, which_arm='r')
        self.init_arm_pose(True, which_arm='l')
        rospy.loginfo('Done initializing arms')

    #
    # Controller setup methods
    #
    def init_joint_controllers(self):
        self.arm_mode = 'joint_mode'
        prefix = roslib.packages.get_pkg_dir('hrl_pr2_arms')+'/params/'
        rospy.loginfo('Setting right arm controller gains')
        rospy.loginfo(self.cs.switch("r_arm_controller", "r_arm_controller",
                                     prefix + "pr2_arm_controllers_push.yaml"))
        rospy.loginfo('Setting left arm controller gains')
        rospy.loginfo(self.cs.switch("l_arm_controller", "l_arm_controller",
                                     prefix + "pr2_arm_controllers_push.yaml"))

    def init_cart_controllers(self):
        self.arm_mode = 'cart_mode'
        self.cs.carefree_switch('r', '%s_cart_posture_push',
                                '$(find tabletop_pushing)/params/j_transpose_task_params_pos_feedback_push.yaml')
        self.cs.carefree_switch('l', '%s_cart_posture_push',
                                '$(find tabletop_pushing)/params/j_transpose_task_params_pos_feedback_push.yaml')

    def switch_to_cart_controllers(self):
        if self.arm_mode != 'cart_mode':
            self.cs.carefree_switch('r', '%s_cart_posture_push')
            self.cs.carefree_switch('l', '%s_cart_posture_push')
            self.arm_mode = 'cart_mode'
            rospy.sleep(1.0)

    def switch_to_joint_controllers(self):
        if self.arm_mode != 'joint_mode':
            self.cs.carefree_switch('r', '%s_arm_controller')
            self.cs.carefree_switch('l', '%s_arm_controller')
            self.arm_mode = 'joint_mode'
            rospy.sleep(1.0)
    #
    # Main Control Loop
    #
    def run(self):
        '''
        Main control loop for the node
        '''
        self.init_spine_pose()
        self.init_head_pose(self.head_pose_cam_frame)
        self.init_arms()
        rospy.loginfo('Done initializing feedback pushing node.')
        rospy.spin()

if __name__ == '__main__':
    node = PositionFeedbackPushNode()
    node.run()
