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
import actionlib
import hrl_pr2_lib.pr2 as pr2
import hrl_pr2_lib.pressure_listener as pl
import hrl_lib.tf_utils as tfu
from hrl_pr2_arms.pr2_controller_switcher import ControllerSwitcher
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float64MultiArray
from pr2_controllers_msgs.msg import *
from pr2_manipulation_controllers.msg import *
import tf
import numpy as np
from tabletop_pushing.srv import *
from tabletop_pushing.msg import *
from math import sin, cos, pi, fabs, sqrt
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

_POSTURES = {
    'off': [],
    'mantis': [0, 1, 0,  -1, 3.14, -1, 3.14],
    'elbowupr': [-0.79,0,-1.6,  9999, 9999, 9999, 9999],
    'elbowupl': [0.79,0,1.6 , 9999, 9999, 9999, 9999],
    'old_elbowupr': [-0.79,0,-1.6, -0.79,3.14, -0.79,5.49],
    'old_elbowupl': [0.79,0,1.6, -0.79,3.14, -0.79,5.49],
    'elbowdownr': [-0.028262077316910873, 1.2946342642324222, -0.25785640577652386, -1.5498884526859626, -31.278913849571776, -1.0527644894829107, -1.8127318367654268],
    'elbowdownl': [-0.0088195719039858515, 1.2834828245284853, 0.20338442004843196, -1.5565279256852611, -0.096340012666916802, -1.0235018652439782, 1.7990893054129216]
}

class PositionFeedbackPushNode:

    def __init__(self):
        rospy.init_node('position_feedback_push_node', log_level=rospy.DEBUG)
        # Setup parameters
        self.torso_z_offset = rospy.get_param('~torso_z_offset', 0.30)
        self.look_pt_x = rospy.get_param('~look_point_x', 0.7)
        self.head_pose_cam_frame = rospy.get_param('~head_pose_cam_frame',
                                                   'head_mount_kinect_rgb_link')
        self.default_torso_height = rospy.get_param('~default_torso_height',
                                                    0.28)
        self.gripper_raise_dist = rospy.get_param('~gripper_raise_dist',
                                                  0.05)
        self.high_arm_init_z = rospy.get_param('~high_arm_start_z', 0.15)
        self.post_controller_switch_sleep = rospy.get_param(
            '~arm_switch_sleep_time', 0.5)
        self.move_cart_check_hz = rospy.get_param('~move_cart_check_hz', 100)
        self.arm_done_moving_count_thresh = rospy.get_param(
            '~not_moving_count_thresh', 30)
        self.arm_done_moving_epc_count_thresh = rospy.get_param(
            '~not_moving_epc_count_thresh', 60)
        self.post_move_count_thresh = rospy.get_param('~post_move_count_thresh',
                                                      10)
        self.pre_push_count_thresh = rospy.get_param('~pre_push_count_thresh',
                                                      50)
        self.still_moving_velocity = rospy.get_param('~moving_vel_thresh', 0.01)
        self.still_moving_angular_velocity = rospy.get_param('~angular_vel_thresh',
                                                             0.005)
        self.pressure_safety_limit = rospy.get_param('~pressure_limit',
                                                     2000)
        self.use_jinv = rospy.get_param('~use_jinv', True)
        self.use_cur_joint_posture = rospy.get_param('~use_joint_posture', True)
        # Setup cartesian controller parameters
        if self.use_jinv:
            self.base_cart_controller_name = '_cart_jinv_push'
            self.controller_state_msg = JinvTeleopControllerState
        else:
            self.base_cart_controller_name = '_cart_transpose_push'
            self.controller_state_msg = JTTaskControllerState
        self.base_vel_controller_name = '_cart_jinv_push'
        self.vel_controller_state_msg = JinvTeleopControllerState

        # Set joint gains
        rospy.loginfo('Craeting cs object')
        self.arm_mode = None
        self.cs = ControllerSwitcher()
        self.init_joint_controllers()
        self.init_cart_controllers()
        self.init_vel_controllers()

        # Setup arms
        self.tf_listener = tf.TransformListener()
        rospy.loginfo('Creating pr2 object')
        self.robot = pr2.PR2(self.tf_listener, arms=True, base=False, 
                             use_kinematics=False)#, use_projector=False)

        self.l_arm_cart_pub = rospy.Publisher(
            '/l'+self.base_cart_controller_name+'/command_pose', PoseStamped)
        self.r_arm_cart_pub = rospy.Publisher(
            '/r'+self.base_cart_controller_name+'/command_pose', PoseStamped)
        self.l_arm_cart_posture_pub = rospy.Publisher(
            '/l'+self.base_cart_controller_name+'/command_posture',
            Float64MultiArray)
        self.r_arm_cart_posture_pub = rospy.Publisher(
            '/r'+self.base_cart_controller_name+'/command_posture',
            Float64MultiArray)
        self.l_arm_cart_vel_pub = rospy.Publisher(
            '/l'+self.base_vel_controller_name+'/command_twist', TwistStamped)
        self.r_arm_cart_vel_pub = rospy.Publisher(
            '/r'+self.base_vel_controller_name+'/command_twist', TwistStamped)
        self.l_arm_vel_posture_pub = rospy.Publisher(
            '/l'+self.base_vel_controller_name+'/command_posture',
            Float64MultiArray)
        self.r_arm_vel_posture_pub = rospy.Publisher(
            '/r'+self.base_vel_controller_name+'/command_posture',
            Float64MultiArray)

        rospy.Subscriber('/l'+self.base_cart_controller_name+'/state',
                         self.controller_state_msg,
                         self.l_arm_cart_state_callback)
        rospy.Subscriber('/r'+self.base_cart_controller_name+'/state',
                         self.controller_state_msg,
                         self.r_arm_cart_state_callback)

        # TODO: setup velocity controller callback methods
        # rospy.Subscriber('/l'+self.base_vel_controller_name+'/state',
        #                  self.vel_controller_state_msg,
        #                  self.l_arm_vel_state_callback)
        # rospy.Subscriber('/r'+self.base_vel_controller_name+'/state',
        #                  self.vel_controller_state_msg,
        #                  self.r_arm_vel_state_callback)

        self.l_pressure_listener = pl.PressureListener(
            '/pressure/l_gripper_motor', self.pressure_safety_limit)
        self.r_pressure_listener = pl.PressureListener(
            '/pressure/r_gripper_motor', self.pressure_safety_limit)


        # State Info
        self.l_arm_pose = None
        self.l_arm_x_err = None
        self.l_arm_x_d = None
        self.l_arm_F = None

        self.r_arm_pose = None
        self.r_arm_x_err = None
        self.r_arm_x_d = None
        self.r_arm_F = None

        # Open callback services
        self.gripper_feedback_push_srv = rospy.Service(
            'gripper_feedback_push', GripperPush, self.gripper_feedback_push)
        self.gripper_pre_push_srv = rospy.Service('gripper_pre_push',
                                                  GripperPush,
                                                  self.gripper_pre_push)
        self.gripper_push_srv = rospy.Service('gripper_push', GripperPush,
                                              self.gripper_push)
        self.gripper_post_push_srv = rospy.Service('gripper_post_push',
                                                   GripperPush,
                                                   self.gripper_post_push)

        self.gripper_pre_sweep_srv = rospy.Service('gripper_pre_sweep',
                                                   GripperPush,
                                                   self.gripper_pre_sweep)
        self.gripper_sweep_srv = rospy.Service('gripper_sweep',
                                               GripperPush,
                                               self.gripper_sweep)
        self.gripper_post_sweep_srv = rospy.Service('gripper_post_sweep',
                                                    GripperPush,
                                                    self.gripper_post_sweep)

        self.overhead_pre_push_srv = rospy.Service('overhead_pre_push',
                                                   GripperPush,
                                                   self.overhead_pre_push)
        self.overhead_push_srv = rospy.Service('overhead_push',
                                               GripperPush,
                                               self.overhead_push)
        self.overhead_post_push_srv = rospy.Service('overhead_post_push',
                                                   GripperPush,
                                                   self.overhead_post_push)

        self.raise_and_look_serice = rospy.Service('raise_and_look',
                                                   RaiseAndLook,
                                                   self.raise_and_look)
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
            robot_gripper = self.robot.left_gripper
            ready_joints = LEFT_ARM_READY_JOINTS
            setup_joints = LEFT_ARM_SETUP_JOINTS
        else:
            robot_gripper = self.robot.right_gripper
            ready_joints = RIGHT_ARM_READY_JOINTS
            setup_joints = RIGHT_ARM_SETUP_JOINTS

        rospy.loginfo('Moving %s_arm to setup pose' % which_arm)
        self.set_arm_joint_pose(setup_joints, which_arm)
        rospy.loginfo('Moved %s_arm to setup pose' % which_arm)

        #rospy.loginfo('Closing %s_gripper' % which_arm)
        #res = robot_gripper.close(block=True)
        #rospy.loginfo('Closed %s_gripper' % which_arm)


    def reset_arm_pose(self, force_ready=False, which_arm='l',
                       high_arm_joints=False):
        '''
        Move the arm to the initial pose to be out of the way for viewing the
        tabletop
        '''
        if which_arm == 'l':
            robot_gripper = self.robot.left_gripper
            if high_arm_joints:
                ready_joints = LEFT_ARM_HIGH_PUSH_READY_JOINTS
            else:
                ready_joints = LEFT_ARM_READY_JOINTS
            setup_joints = LEFT_ARM_SETUP_JOINTS
        else:
            robot_gripper = self.robot.right_gripper
            if high_arm_joints:
                ready_joints = RIGHT_ARM_HIGH_PUSH_READY_JOINTS
            else:
                ready_joints = RIGHT_ARM_READY_JOINTS
            setup_joints = RIGHT_ARM_SETUP_JOINTS
        ready_diff = np.linalg.norm(pr2.diff_arm_pose(self.get_arm_joint_pose(which_arm),
                                                      ready_joints))

        # Choose to move to ready first, if it is closer, then move to init
        if force_ready or ready_diff > READY_POSE_MOVE_THRESH:
            rospy.logdebug('Moving %s_arm to ready pose' % which_arm)
            self.set_arm_joint_pose(ready_joints, which_arm, nsecs=1.5)
            rospy.logdebug('Moved %s_arm to ready pose' % which_arm)
        else:
            rospy.logdebug('Arm in ready pose')

        rospy.logdebug('Moving %s_arm to setup pose' % which_arm)
        self.set_arm_joint_pose(setup_joints, which_arm, nsecs=1.5)
        rospy.logdebug('Moved %s_arm to setup pose' % which_arm)

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
            rospy.logwarn('Failed to point head')
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
    # Behavior functions
    #
    def gripper_feedback_push(self, request):
        response = GripperPushResponse()
        start_point = request.start_point.point
        wrist_yaw = request.wrist_yaw
        push_dist = request.desired_push_dist

        if request.left_arm:
            ready_joints = LEFT_ARM_READY_JOINTS
            which_arm = 'l'
        else:
            ready_joints = RIGHT_ARM_READY_JOINTS
            which_arm = 'r'

        self.active_arm = which_arm
        rospy.loginfo('Creating ac')
        ac = actionlib.SimpleActionClient('push_tracker',
                                          VisFeedbackPushTrackingAction)
        rospy.loginfo('waiting for server')
        if not ac.wait_for_server(rospy.Duration(5.0)):
            rospy.loginfo('Failed to connect to push tracker server')
            return response
        ac.cancel_all_goals()
        self.feedback_count = 0

        # Start pushing forward
        # self.vel_push_forward(which_arm)
        done_cb = None
        active_cb = None
        feedback_cb = self.tracker_feedback_gripper_push
        goal = VisFeedbackPushTrackingGoal()
        goal.which_arm = which_arm
        goal.header.frame_id = request.start_point.header.frame_id
        goal.desired_pose = request.goal_pose
        self.desired_pose = request.goal_pose
        rospy.loginfo('Sending goal of: ' + str(goal.desired_pose))
        ac.send_goal(goal, done_cb, active_cb, feedback_cb)
        # Block until done
        rospy.loginfo('Waiting for result')
        ac.wait_for_result(rospy.Duration(0))
        rospy.loginfo('Result received')
        self.stop_moving_vel(which_arm)

        # TODO: send back info
        return response

    def tracker_feedback_gripper_push(self, feedback):
        which_arm = self.active_arm
        update_twist = TwistStamped()
        update_twist.header.frame_id = 'torso_lift_link'
        update_twist.header.stamp = rospy.Time(0)
        update_twist.twist.linear.z = 0.0
        update_twist.twist.angular.x = 0.0
        update_twist.twist.angular.y = 0.0
        update_twist.twist.angular.z = 0.0

        # Push centroid towards the desired goal
        goal_x_dot = self.desired_pose.x - feedback.x.x
        goal_y_dot = self.desired_pose.y - feedback.x.y

        # Add in direction to corect for spinning
        # TODO: Correct component based on transform in object pose
        spin_x_dot = sin(feedback.x.theta)*feedback.x_dot.theta
        spin_y_dot = cos(feedback.x.theta)*feedback.x_dot.theta

        k_g = 0.1
        k_s = 0.03

        update_twist.twist.linear.x = k_g*goal_x_dot + k_s*spin_x_dot
        update_twist.twist.linear.y = k_g*goal_y_dot + k_s*spin_y_dot
        if self.feedback_count % 15 == 0:
            rospy.loginfo('Desired pose: ' + str(self.desired_pose))
            rospy.loginfo('Current pose: ' + str(feedback.x))
            rospy.loginfo('q_goal_dot: (' + str(k_g*goal_x_dot) + ', ' + str(k_g*goal_y_dot) + ')')
            rospy.loginfo('q_spin_dot: (' + str(spin_x_dot) + ', ' + str(spin_y_dot) + ')')
            rospy.loginfo('q_dot: (' + str(update_twist.twist.linear.x) + ',' + str(update_twist.twist.linear.y) + ')\n')
        update_twist.twist.linear.x = k_g*goal_x_dot
        update_twist.twist.linear.y = k_g*goal_y_dot
        self.update_vel(update_twist, which_arm)
        self.feedback_count += 1

    def gripper_push(self, request):
        response = GripperPushResponse()
        start_point = request.start_point.point
        wrist_yaw = request.wrist_yaw
        push_dist = request.desired_push_dist

        if request.left_arm:
            ready_joints = LEFT_ARM_READY_JOINTS
            which_arm = 'l'
        else:
            ready_joints = RIGHT_ARM_READY_JOINTS
            which_arm = 'r'

        rospy.loginfo('Pushing forward ' + str(push_dist) + 'm')
        # pose_err, err_dist = self.move_relative_gripper(
        #     np.matrix([push_dist, 0.0, 0.0]).T, which_arm)
        # pose_err, err_dist = self.move_relative_torso(
        #     np.matrix([cos(wrist_yaw)*push_dist,
        #                sin(wrist_yaw)*push_dist, 0.0]).T, which_arm)
        pose_err, err_dist = self.move_relative_torso_epc(wrist_yaw, push_dist,
                                                          which_arm)
        rospy.loginfo('Done pushing forward')

        response.dist_pushed = push_dist - err_dist
        return response

    def gripper_pre_push(self, request):
        response = GripperPushResponse()
        start_point = request.start_point.point
        wrist_yaw = request.wrist_yaw
        push_dist = request.desired_push_dist
        if request.left_arm:
            ready_joints = LEFT_ARM_READY_JOINTS
            if request.high_arm_init:
                ready_joints = LEFT_ARM_HIGH_PUSH_READY_JOINTS
            which_arm = 'l'
            robot_gripper = self.robot.left_gripper
        else:
            ready_joints = RIGHT_ARM_READY_JOINTS
            if request.high_arm_init:
                ready_joints = RIGHT_ARM_HIGH_PUSH_READY_JOINTS
            which_arm = 'r'
            robot_gripper = self.robot.right_gripper

        if request.arm_init:
            self.set_arm_joint_pose(ready_joints, which_arm, nsecs=1.5)
            rospy.logdebug('Moving %s_arm to ready pose' % which_arm)

        start_pose = PoseStamped()
        start_pose.header = request.start_point.header
        start_pose.pose.position.x = start_point.x
        start_pose.pose.position.y = start_point.y
        start_pose.pose.position.z = start_point.z
        q = tf.transformations.quaternion_from_euler(0.0, -0.1, wrist_yaw)
        start_pose.pose.orientation.x = q[0]
        start_pose.pose.orientation.y = q[1]
        start_pose.pose.orientation.z = q[2]
        start_pose.pose.orientation.w = q[3]

        if request.open_gripper:
            # TODO: open gripper
            res = robot_gripper.open(block=True, position=0.05)

        if request.high_arm_init:
            start_pose.pose.position.z = self.high_arm_init_z
            self.move_to_cart_pose(start_pose, which_arm)
            rospy.logdebug('Done moving to overhead start point')
            # Change z to lower arm to table
            start_pose.pose.position.z = start_point.z
            # self.move_down_until_contact(which_arm)

        # Move to start pose
        self.move_to_cart_pose(start_pose, which_arm,
                               self.pre_push_count_thresh)
        rospy.loginfo('Done moving to start point')
        return response

    def gripper_post_push(self, request):
        response = GripperPushResponse()
        start_point = request.start_point.point
        wrist_yaw = request.wrist_yaw
        push_dist = request.desired_push_dist

        if request.left_arm:
            ready_joints = LEFT_ARM_READY_JOINTS
            which_arm = 'l'
            robot_gripper = self.robot.left_gripper
        else:
            ready_joints = RIGHT_ARM_READY_JOINTS
            which_arm = 'r'
            robot_gripper = self.robot.right_gripper

        # Retract in a straight line
        rospy.logdebug('Moving gripper up')
        pose_err, err_dist = self.move_relative_gripper(
            np.matrix([0.0, 0.0, self.gripper_raise_dist]).T, which_arm,
            move_cart_count_thresh=self.post_move_count_thresh)
        rospy.logdebug('Moving gripper backwards')
        pose_err, err_dist = self.move_relative_gripper(
            np.matrix([-push_dist, 0.0, 0.0]).T, which_arm,
            move_cart_count_thresh=self.post_move_count_thresh)
        rospy.loginfo('Done moving backwards')
        start_pose = PoseStamped()
        start_pose.header = request.start_point.header
        start_pose.pose.position.x = start_point.x
        start_pose.pose.position.y = start_point.y
        start_pose.pose.position.z = start_point.z
        q = tf.transformations.quaternion_from_euler(0.0, 0.0, wrist_yaw)
        start_pose.pose.orientation.x = q[0]
        start_pose.pose.orientation.y = q[1]
        start_pose.pose.orientation.z = q[2]
        start_pose.pose.orientation.w = q[3]

        if request.open_gripper:
            # TODO: Close gripper
            res = robot_gripper.close(block=True)

        if request.high_arm_init:
            start_pose.pose.position.z = self.high_arm_init_z
            rospy.logdebug('Moving up to initial point')
            self.move_to_cart_pose(start_pose, which_arm,
                                   self.post_move_count_thresh)
            rospy.loginfo('Done moving up to initial point')

        if request.arm_reset:
            self.reset_arm_pose(True, which_arm, request.high_arm_init)
        return response

    def gripper_sweep(self, request):
        response = GripperPushResponse()
        start_point = request.start_point.point
        wrist_yaw = request.wrist_yaw
        push_dist = request.desired_push_dist

        if request.left_arm:
            ready_joints = LEFT_ARM_READY_JOINTS
            which_arm = 'l'
            wrist_roll = -pi
        else:
            ready_joints = RIGHT_ARM_READY_JOINTS
            which_arm = 'r'
            wrist_roll = 0.0

        # NOTE: because of the wrist roll orientation, +Z at the gripper
        # equates to negative Y in the torso_lift_link at 0.0 yaw
        # So we flip the push_dist to make things look like one would expect
        rospy.loginfo('Sweeping gripper in ' + str(push_dist) + 'm')
        # pose_err, err_dist = self.move_relative_gripper(
        #     np.matrix([0.0, 0.0, -push_dist]).T, which_arm)
        if wrist_yaw > -pi*0.5:
            push_angle = wrist_yaw + pi*0.5
        else:
            push_angle = wrist_yaw - pi*0.5
        # pose_err, err_dist = self.move_relative_torso(
        #     np.matrix([cos(push_angle)*push_dist,
        #                sin(push_angle)*push_dist, 0.0]).T, which_arm)
        pose_err, err_dist = self.move_relative_torso_epc(push_angle, push_dist,
                                                          which_arm)

        rospy.logdebug('Done sweeping in')

        # response.dist_pushed = push_dist - err_dist
        return response

    def gripper_pre_sweep(self, request):
        response = GripperPushResponse()
        start_point = request.start_point.point
        wrist_yaw = request.wrist_yaw
        push_dist = request.desired_push_dist

        if request.left_arm:
            ready_joints = LEFT_ARM_READY_JOINTS
            if request.high_arm_init:
                ready_joints = LEFT_ARM_HIGH_SWEEP_READY_JOINTS
            which_arm = 'l'
            wrist_roll = -pi
        else:
            ready_joints = RIGHT_ARM_READY_JOINTS
            if request.high_arm_init:
                ready_joints = RIGHT_ARM_HIGH_SWEEP_READY_JOINTS
            which_arm = 'r'
            wrist_roll = 0.0

        start_pose = PoseStamped()
        start_pose.header = request.start_point.header
        start_pose.pose.position.x = start_point.x
        start_pose.pose.position.y = start_point.y
        start_pose.pose.position.z = start_point.z
        q = tf.transformations.quaternion_from_euler(0.5*pi, 0.0, wrist_yaw)
        start_pose.pose.orientation.x = q[0]
        start_pose.pose.orientation.y = q[1]
        start_pose.pose.orientation.z = q[2]
        start_pose.pose.orientation.w = q[3]

        if request.arm_init:
            rospy.logdebug('Moving %s_arm to ready pose' % which_arm)
            self.set_arm_joint_pose(ready_joints, which_arm, nsecs=1.5)
            # Rotate wrist before moving to position
            rospy.logdebug('Rotating wrist for sweep')
            arm_pose = self.get_arm_joint_pose(which_arm)
            arm_pose[-1] =  wrist_roll
            self.set_arm_joint_pose(arm_pose, which_arm, nsecs=1.0)
        if request.high_arm_init:
            # Move to offset pose above the table
            start_pose.pose.position.z = self.high_arm_init_z
            self.move_to_cart_pose(start_pose, which_arm)
            rospy.logdebug('Done moving to overhead start point')
            # Lower arm to table
            start_pose.pose.position.z = start_point.z
            # self.move_down_until_contact(which_arm)

        self.move_to_cart_pose(start_pose, which_arm,
                               self.pre_push_count_thresh)
        rospy.loginfo('Done moving to start point')

        return response

    def gripper_post_sweep(self, request):
        response = GripperPushResponse()
        start_point = request.start_point.point
        wrist_yaw = request.wrist_yaw
        push_dist = request.desired_push_dist

        if request.left_arm:
            ready_joints = LEFT_ARM_READY_JOINTS
            which_arm = 'l'
        else:
            ready_joints = RIGHT_ARM_READY_JOINTS
            which_arm = 'r'

        rospy.logdebug('Moving gripper up')
        pose_err, err_dist = self.move_relative_gripper(
            np.matrix([0.0, self.gripper_raise_dist, 0.0]).T, which_arm,
            move_cart_count_thresh=self.post_move_count_thresh)
        rospy.logdebug('Sweeping gripper outward')
        pose_err, err_dist = self.move_relative_gripper(
            np.matrix([0.0, 0.0, push_dist]).T, which_arm,
            move_cart_count_thresh=self.post_move_count_thresh)
        rospy.loginfo('Done sweeping outward')

        if request.high_arm_init:
            rospy.logdebug('Moving up to end point')
            wrist_yaw = request.wrist_yaw
            end_pose = PoseStamped()
            end_pose.header = request.start_point.header
            end_pose.pose.position.x = start_point.x
            end_pose.pose.position.y = start_point.y
            end_pose.pose.position.z = self.high_arm_init_z
            q = tf.transformations.quaternion_from_euler(0.5*pi, 0.0, wrist_yaw)
            end_pose.pose.orientation.x = q[0]
            end_pose.pose.orientation.y = q[1]
            end_pose.pose.orientation.z = q[2]
            end_pose.pose.orientation.w = q[3]
            self.move_to_cart_pose(end_pose, which_arm,
                                   self.post_move_count_thresh)
            rospy.loginfo('Done moving up to end point')

        if request.arm_reset:
            self.reset_arm_pose(True, which_arm, request.high_arm_init)
        return response

    def overhead_push(self, request):
        response = GripperPushResponse()
        start_point = request.start_point.point
        wrist_yaw = request.wrist_yaw
        push_dist = request.desired_push_dist

        if request.left_arm:
            which_arm = 'l'
            wrist_pitch = 0.5*pi
        else:
            which_arm = 'r'
            wrist_pitch = -0.5*pi

        rospy.loginfo('Pushing forward ' + str(push_dist) + 'm')
        # pose_err, err_dist = self.move_relative_gripper(
        #     np.matrix([0.0, 0.0, push_dist]).T, which_arm)
        # pose_err, err_dist = self.move_relative_torso(
        #     np.matrix([cos(wrist_yaw)*push_dist,
        #                sin(wrist_yaw)*push_dist, 0.0]).T, which_arm)
        pose_err, err_dist = self.move_relative_torso_epc(wrist_yaw, push_dist,
                                                          which_arm)

        rospy.logdebug('Done pushing forward')

        # TODO: Add this back in
        response.dist_pushed = push_dist - err_dist
        return response

    def overhead_pre_push(self, request):
        response = GripperPushResponse()
        start_point = request.start_point.point
        wrist_yaw = request.wrist_yaw
        push_dist = request.desired_push_dist

        if request.left_arm:
            ready_joints = LEFT_ARM_READY_JOINTS
            if request.high_arm_init:
                ready_joints = LEFT_ARM_PULL_READY_JOINTS
            which_arm = 'l'
            wrist_pitch = 0.5*pi
        else:
            ready_joints = RIGHT_ARM_READY_JOINTS
            if request.high_arm_init:
                ready_joints = RIGHT_ARM_PULL_READY_JOINTS
            which_arm = 'r'
            wrist_pitch = -0.5*pi

        start_pose = PoseStamped()
        start_pose.header = request.start_point.header
        start_pose.pose.position.x = start_point.x
        start_pose.pose.position.y = start_point.y
        start_pose.pose.position.z = start_point.z
        q = tf.transformations.quaternion_from_euler(0.0, fabs(wrist_pitch),
                                                     wrist_yaw)
        start_pose.pose.orientation.x = q[0]
        start_pose.pose.orientation.y = q[1]
        start_pose.pose.orientation.z = q[2]
        start_pose.pose.orientation.w = q[3]

        if request.arm_init:
            rospy.logdebug('Moving %s_arm to ready pose' % which_arm)
            self.set_arm_joint_pose(ready_joints, which_arm, nsecs=1.5)

            if not request.high_arm_init:
                # Rotate wrist before moving to position
                rospy.logdebug('Rotating elbow for overhead push')
                arm_pose = self.get_arm_joint_pose(which_arm)
                arm_pose[-3] =  wrist_pitch
                self.set_arm_joint_pose(arm_pose, which_arm, nsecs=1.0)

        if request.high_arm_init:
            # Move to offset pose above the table
            start_pose.pose.position.z = self.high_arm_init_z
            self.move_to_cart_pose(start_pose, which_arm)
            rospy.logdebug('Done moving to overhead start point')
            # Lower arm to table
            start_pose.pose.position.z = start_point.z
            # self.move_down_until_contact(which_arm)

        # Move to offset pose
        self.move_to_cart_pose(start_pose, which_arm,
                               self.pre_push_count_thresh)
        rospy.loginfo('Done moving to start point')

        return response

    def overhead_post_push(self, request):
        response = GripperPushResponse()
        start_point = request.start_point.point
        wrist_yaw = request.wrist_yaw
        push_dist = request.desired_push_dist

        if request.left_arm:
            ready_joints = LEFT_ARM_READY_JOINTS
            if request.high_arm_init:
                ready_joints = LEFT_ARM_PULL_READY_JOINTS
            which_arm = 'l'
            wrist_pitch = 0.5*pi
        else:
            ready_joints = RIGHT_ARM_READY_JOINTS
            if request.high_arm_init:
                ready_joints = RIGHT_ARM_PULL_READY_JOINTS
            which_arm = 'r'
            wrist_pitch = -0.5*pi

        rospy.logdebug('Moving gripper up')
        pose_err, err_dist = self.move_relative_gripper(
            np.matrix([-self.gripper_raise_dist, 0.0, 0.0]).T, which_arm,
            move_cart_count_thresh=self.post_move_count_thresh)
        rospy.logdebug('Done moving up')
        rospy.logdebug('Pushing reverse')
        pose_err, err_dist = self.move_relative_gripper(
            np.matrix([0.0, 0.0, -push_dist]).T, which_arm,
            move_cart_count_thresh=self.post_move_count_thresh)
        rospy.loginfo('Done pushing reverse')

        if request.high_arm_init:
            rospy.logdebug('Moving up to end point')
            wrist_yaw = request.wrist_yaw
            end_pose = PoseStamped()
            end_pose.header = request.start_point.header
            end_pose.pose.position.x = start_point.x
            end_pose.pose.position.y = start_point.y
            end_pose.pose.position.z = self.high_arm_init_z
            q = tf.transformations.quaternion_from_euler(0.0, 0.5*pi, wrist_yaw)
            end_pose.pose.orientation.x = q[0]
            end_pose.pose.orientation.y = q[1]
            end_pose.pose.orientation.z = q[2]
            end_pose.pose.orientation.w = q[3]
            self.move_to_cart_pose(end_pose, which_arm,
                                   self.post_move_count_thresh)
            rospy.loginfo('Done moving up to end point')

        if request.arm_reset:
            self.reset_arm_pose(True, which_arm, request.high_arm_init)
        return response

    def raise_and_look(self, request):
        '''
        Service callback to raise the spine to a specific height relative to the
        table height and tilt the head so that the Kinect views the table
        '''
        if request.init_arms:
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
        (trans, rot) = self.tf_listener.lookupTransform('base_link',
                                                        'torso_lift_link',
                                                        rospy.Time(0))
        lift_link_z = trans[2]

        # tabletop position in base_link frame
        request.table_centroid.header.stamp = rospy.Time(0)
        table_base = self.tf_listener.transformPose('base_link',
                                                    request.table_centroid)
        table_z = table_base.pose.position.z
        goal_lift_link_z = table_z + self.torso_z_offset
        lift_link_delta_z = goal_lift_link_z - lift_link_z
        # rospy.logdebug('Torso height (m): ' + str(lift_link_z))
        rospy.logdebug('Table height (m): ' + str(table_z))
        rospy.logdebug('Torso goal height (m): ' + str(goal_lift_link_z))
        # rospy.logdebug('Torso delta (m): ' + str(lift_link_delta_z))

        # Set goal height based on passed on table height
        # TODO: Set these better
        torso_max = 0.3
        torso_min = 0.01
        current_torso_position = np.asarray(self.robot.torso.pose()).ravel()[0]
        torso_goal_position = current_torso_position + lift_link_delta_z
        torso_goal_position = (max(min(torso_max, torso_goal_position),
                                   torso_min))
        # rospy.logdebug('Moving torso to ' + str(torso_goal_position))
        # Multiply by 2.0, because of units of spine
        self.robot.torso.set_pose(torso_goal_position)

        # rospy.logdebug('Got torso client result')
        new_torso_position = np.asarray(self.robot.torso.pose()).ravel()[0]
        # rospy.logdebug('New torso position is: ' + str(new_torso_position))

        # Get torso_lift_link position in base_link frame

        (new_trans, rot) = self.tf_listener.lookupTransform('base_link',
                                                            'torso_lift_link',
                                                            rospy.Time(0))
        new_lift_link_z = new_trans[2]
        # rospy.logdebug('New Torso height (m): ' + str(new_lift_link_z))
        # tabletop position in base_link frame
        new_table_base = self.tf_listener.transformPose('base_link',
                                                        request.table_centroid)
        new_table_z = new_table_base.pose.position.z
        rospy.loginfo('New Table height: ' + str(new_table_z))

        # Point the head at the table centroid
        # NOTE: Should we fix the tilt angle instead for consistency?
        look_pt = np.asmatrix([self.look_pt_x,
                               0.0,
                               -self.torso_z_offset])
        rospy.logdebug('Point head at ' + str(look_pt))
        head_res = self.robot.head.look_at(look_pt,
                                           request.table_centroid.header.frame_id,
                                           request.camera_frame)
        response = RaiseAndLookResponse()
        if head_res:
            rospy.loginfo('Succeeded in pointing head')
            response.head_succeeded = True
        else:
            rospy.logwarn('Failed to point head')
            response.head_succeeded = False
        return response

    #
    # Controller wrapper methods
    #
    def get_arm_joint_pose(self, which_arm):
        if which_arm == 'l':
            return self.robot.left.pose()
        else:
            return self.robot.right.pose()

    def set_arm_joint_pose(self, joint_pose, which_arm, nsecs=2.0):
        self.switch_to_joint_controllers()
        if which_arm == 'l':
            self.robot.left.set_pose(joint_pose, nsecs, block=True)
        else:
            self.robot.right.set_pose(joint_pose, nsecs, block=True)

    def move_to_cart_pose(self, pose, which_arm,
                          done_moving_count_thresh=None, pressure=1000):
        if done_moving_count_thresh is None:
            done_moving_count_thresh = self.arm_done_moving_count_thresh
        self.switch_to_cart_controllers()
        if which_arm == 'l':
            self.l_arm_cart_pub.publish(pose)
            posture_pub = self.l_arm_cart_posture_pub
            pl = self.l_pressure_listener
        else:
            self.r_arm_cart_pub.publish(pose)
            posture_pub = self.r_arm_cart_posture_pub
            pl = self.r_pressure_listener

        arm_not_moving_count = 0
        r = rospy.Rate(self.move_cart_check_hz)
        pl.rezero()
        pl.set_threshold(pressure)
        while arm_not_moving_count < done_moving_count_thresh:
            if not self.arm_moving_cart(which_arm):
                arm_not_moving_count += 1
            else:
                arm_not_moving_count = 0
            # Command posture
            m = self.get_desired_posture(which_arm)
            posture_pub.publish(m)

            if pl.check_safety_threshold():
                rospy.loginfo('Exceeded pressure safety thresh!\n')
                break
            if pl.check_threshold():
                rospy.loginfo('Exceeded pressure contact thresh...')
                # TODO: Let something know?
            r.sleep()

        # Return pose error
        if which_arm == 'l':
            arm_error = self.l_arm_x_err
        else:
            arm_error = self.r_arm_x_err
        error_dist = sqrt(arm_error.linear.x**2 + arm_error.linear.y**2 +
                         arm_error.linear.z**2)
        rospy.logdebug('Move cart gripper error dist: ' + str(error_dist)+'\n')
        return (arm_error, error_dist)


    def arm_moving_cart(self, which_arm):
        if which_arm == 'l':
            x_err = self.l_arm_x_err
            x_d = self.l_arm_x_d
        else:
            x_err = self.r_arm_x_err
            x_d = self.r_arm_x_d

        moving = (fabs(x_d.linear.x) > self.still_moving_velocity or
                  fabs(x_d.linear.y) > self.still_moving_velocity or
                  fabs(x_d.linear.z) > self.still_moving_velocity or
                  fabs(x_d.angular.x) > self.still_moving_angular_velocity or
                  fabs(x_d.angular.y) > self.still_moving_angular_velocity or
                  fabs(x_d.angular.z) > self.still_moving_angular_velocity)

        return moving

    def move_relative_gripper(self, rel_push_vector, which_arm,
                              move_cart_count_thresh=None, pressure=1000):
        rel_pose = PoseStamped()
        rel_pose.header.stamp = rospy.Time(0)
        rel_pose.header.frame_id = '/'+which_arm + '_gripper_tool_frame'
        rel_pose.pose.position.x = float(rel_push_vector[0])
        rel_pose.pose.position.y = float(rel_push_vector[1])
        rel_pose.pose.position.z = float(rel_push_vector[2])
        rel_pose.pose.orientation.x = 0
        rel_pose.pose.orientation.y = 0
        rel_pose.pose.orientation.z = 0
        rel_pose.pose.orientation.w = 1.0
        return self.move_to_cart_pose(rel_pose, which_arm,
                                      move_cart_count_thresh, pressure)

    def move_relative_torso(self, rel_push_vector, which_arm,
                            move_cart_count_thresh=None, pressure=1000):
        if which_arm == 'l':
            cur_pose = self.l_arm_pose
        else:
            cur_pose = self.r_arm_pose
        rel_pose = PoseStamped()
        rel_pose.header.stamp = rospy.Time(0)
        rel_pose.header.frame_id = 'torso_lift_link'
        rel_pose.pose.position.x = cur_pose.pose.position.x + \
            float(rel_push_vector[0])
        rel_pose.pose.position.y = cur_pose.pose.position.y + \
            float(rel_push_vector[1])
        rel_pose.pose.position.z = cur_pose.pose.position.z + \
            float(rel_push_vector[2])
        rel_pose.pose.orientation = cur_pose.pose.orientation
        return self.move_to_cart_pose(rel_pose, which_arm,
                                      move_cart_count_thresh, pressure)

    def move_relative_torso_epc(self, push_angle, push_dist, which_arm,
                                move_cart_count_thresh=None, pressure=1000):
        delta_x = cos(push_angle)*push_dist
        delta_y = sin(push_angle)*push_dist
        move_x = cos(push_angle)
        move_y = cos(push_angle)
        if which_arm == 'l':
            start_pose = self.l_arm_pose
        else:
            start_pose = self.r_arm_pose
        desired_pose = PoseStamped()
        desired_pose.header.stamp = rospy.Time(0)
        desired_pose.header.frame_id = 'torso_lift_link'
        desired_pose.pose.position.x = start_pose.pose.position.x + delta_x
        desired_pose.pose.position.y = start_pose.pose.position.y + delta_y
        desired_pose.pose.position.z = start_pose.pose.position.z
        desired_pose.pose.orientation = start_pose.pose.orientation

        desired_x = desired_pose.pose.position.x
        desired_y = desired_pose.pose.position.y

        start_x = start_pose.pose.position.x
        start_y = start_pose.pose.position.y

        def move_epc_generator(cur_ep, which_arm, converged_epsilon=0.01):
            ep = cur_ep
            ep.header.stamp = rospy.Time(0)
            step_size = 0.001
            ep.pose.position.x += step_size*move_x
            ep.pose.position.y += step_size*move_y
            if which_arm == 'l':
                cur_pose = self.l_arm_pose
            else:
                cur_pose = self.r_arm_pose
            cur_x = cur_pose.pose.position.x
            cur_y = cur_pose.pose.position.y
            arm_error_x = desired_x - cur_x
            arm_error_y = desired_y - cur_y
            error_dist = sqrt(arm_error_x**2 + arm_error_y**2)

            # TODO: Check moved passed point
            if error_dist < converged_epsilon:
                stop = 'converged'
            elif (((start_x > desired_x and cur_x < desired_x) or
                   (start_x < desired_x and cur_x > desired_x)) and
                  ((start_y > desired_y and cur_y < desired_y) or
                   (start_y < desired_y and cur_y > desired_y))):
                stop = 'moved_passed'
            else:
                stop = ''
            return (stop, ep)

        return self.move_to_cart_pose_epc(desired_pose, which_arm,
                                          move_epc_generator,
                                          move_cart_count_thresh, pressure)

    def move_to_cart_pose_epc(self, desired_pose, which_arm, ep_gen,
                              done_moving_count_thresh=None, pressure=1000,
                              exit_on_contact=False):
        if done_moving_count_thresh is None:
            done_moving_count_thresh = self.arm_done_moving_epc_count_thresh

        if which_arm == 'l':
            pose_pub = self.l_arm_cart_pub
            posture_pub = self.l_arm_cart_posture_pub
            pl = self.l_pressure_listener
        else:
            pose_pub = self.r_arm_cart_pub
            posture_pub = self.r_arm_cart_posture_pub
            pl = self.r_pressure_listener

        self.switch_to_cart_controllers()

        arm_not_moving_count = 0
        r = rospy.Rate(self.move_cart_check_hz)
        pl.rezero()
        pl.set_threshold(pressure)
        rospy.Time()
        timeout = 5
        timeout_at = rospy.get_time() + timeout
        ep = desired_pose
        while True:
            if not self.arm_moving_cart(which_arm):
                arm_not_moving_count += 1
            else:
                arm_not_moving_count = 0

            if arm_not_moving_count > done_moving_count_thresh:
                rospy.loginfo('Exiting do to no movement!')
                break
            if pl.check_safety_threshold():
                rospy.loginfo('Exceeded pressure safety thresh!')
                break
            if pl.check_threshold():
                rospy.loginfo('Exceeded pressure contact thresh...')
                if exit_on_contact:
                    break
            if timeout_at < rospy.get_time():
                rospy.loginfo('Exceeded time to move EPC!')
                stop = 'timed out'
                break
            # Command posture
            m = self.get_desired_posture(which_arm)
            posture_pub.publish(m)
            # Command pose
            pose_pub.publish(ep)
            r.sleep()

            # Determine new equilibrium point
            stop, ep = ep_gen(ep, which_arm)
            if stop != '':
                rospy.loginfo('Reached goal pose: ' + stop + '\n')
                break
        self.stop_moving_cart(which_arm)
        # Return pose error
        if which_arm == 'l':
            arm_error = self.l_arm_x_err
        else:
            arm_error = self.r_arm_x_err
        error_dist = sqrt(arm_error.linear.x**2 + arm_error.linear.y**2 +
                          arm_error.linear.z**2)
        rospy.loginfo('Move cart gripper error dist: ' + str(error_dist)+'\n')
        return (arm_error, error_dist)

    def stop_moving_cart(self, which_arm):
        if which_arm == 'l':
            self.l_arm_cart_pub.publish(self.l_arm_pose)
        else:
            self.r_arm_cart_pub.publish(self.r_arm_pose)

    def move_down_until_contact(self, which_arm, pressure=1000):
        rospy.loginfo('Moving down!')
        down_twist = TwistStamped()
        down_twist.header.stamp = rospy.Time(0)
        down_twist.header.frame_id = 'torso_lift_link'
        down_twist.twist.linear.x = 0.0
        down_twist.twist.linear.y = 0.0
        down_twist.twist.linear.z = -0.1
        down_twist.twist.angular.x = 0.0
        down_twist.twist.angular.y = 0.0
        down_twist.twist.angular.z = 0.0
        return self.move_until_contact(down_twist, which_arm)


    def move_until_contact(self, twist, which_arm,
                          done_moving_count_thresh=None, pressure=1000):
        self.switch_to_vel_controllers()
        if which_arm == 'l':
            vel_pub = self.l_arm_cart_vel_pub
            posture_pub = self.l_arm_vel_posture_pub
            pl = self.l_pressure_listener
        else:
            vel_pub = self.r_arm_cart_vel_pub
            posture_pub = self.r_arm_vel_posture_pub
            pl = self.r_pressure_listener

        arm_not_moving_count = 0
        r = rospy.Rate(self.move_cart_check_hz)
        pl.rezero()
        pl.set_threshold(pressure)
        rospy.Time()
        timeout = 5
        timeout_at = rospy.get_time() + timeout

        while timeout_at > rospy.get_time():
            twist.header.stamp = rospy.Time(0)
            vel_pub.publish(twist)
            if not self.arm_moving_cart(which_arm):
                arm_not_moving_count += 1
            else:
                arm_not_moving_count = 0
            # Command posture
            m = self.get_desired_posture(which_arm)
            posture_pub.publish(m)

            if pl.check_safety_threshold():
                rospy.loginfo('Exceeded pressure safety thresh!\n')
                break
            if pl.check_threshold():
                rospy.loginfo('Exceeded pressure contact thresh...')
                break
            r.sleep()

        self.stop_moving_vel(which_arm)

        # Return pose error
        return

    def vel_push_forward(self, which_arm, speed=0.3):
        self.switch_to_vel_controllers()

        forward_twist = TwistStamped()
        if which_arm == 'l':
            vel_pub = self.l_arm_cart_vel_pub
            posture_pub = self.l_arm_vel_posture_pub
        else:
            vel_pub = self.r_arm_cart_vel_pub
            posture_pub = self.r_arm_vel_posture_pub

        forward_twist.header.frame_id = '/'+which_arm + '_gripper_tool_frame'
        forward_twist.header.stamp = rospy.Time(0)
        forward_twist.twist.linear.x = speed
        forward_twist.twist.linear.y = 0.0
        forward_twist.twist.linear.z = 0.0
        forward_twist.twist.angular.x = 0.0
        forward_twist.twist.angular.y = 0.0
        forward_twist.twist.angular.z = 0.0
        m = self.get_desired_posture(which_arm)
        posture_pub.publish(m)
        vel_pub.publish(forward_twist)

    def update_vel(self, update_twist, which_arm):
        # Note: assumes velocity controller already running
        if which_arm == 'l':
            vel_pub = self.l_arm_cart_vel_pub
            posture_pub = self.l_arm_vel_posture_pub
        else:
            vel_pub = self.r_arm_cart_vel_pub
            posture_pub = self.r_arm_vel_posture_pub

        m = self.get_desired_posture(which_arm)
        posture_pub.publish(m)
        vel_pub.publish(update_twist)

    def stop_moving_vel(self, which_arm):
        self.switch_to_vel_controllers()
        if which_arm == 'l':
            vel_pub = self.l_arm_cart_vel_pub
            posture_pub = self.l_arm_vel_posture_pub
        else:
            vel_pub = self.r_arm_cart_vel_pub
            posture_pub = self.r_arm_vel_posture_pub

        stop_twist = TwistStamped()
        stop_twist.header.stamp = rospy.Time(0)
        stop_twist.header.frame_id = 'torso_lift_link'
        stop_twist.twist.linear.x = 0.0
        stop_twist.twist.linear.y = 0.0
        stop_twist.twist.linear.z = 0.0
        stop_twist.twist.angular.x = 0.0
        stop_twist.twist.angular.y = 0.0
        stop_twist.twist.angular.z = 0.0
        vel_pub.publish(stop_twist)

    def get_desired_posture(self, which_arm):
        if which_arm == 'l':
            posture = 'elbowupl'
        else:
            posture = 'elbowupr'

        joints = self.get_arm_joint_pose(which_arm)
        joints = joints.tolist()
        joints = [j[0] for j in joints]
        if self.use_cur_joint_posture:
            m = Float64MultiArray(data=joints)
        else:
            m = Float64MultiArray(data=_POSTURES[posture])
        return m

    def l_arm_cart_state_callback(self, state_msg):
        x_err = state_msg.x_err
        x_d = state_msg.xd
        self.l_arm_pose = state_msg.x
        self.l_arm_x_err = x_err
        self.l_arm_x_d = x_d
        self.l_arm_F = state_msg.F

    def r_arm_cart_state_callback(self, state_msg):
        x_err = state_msg.x_err
        x_d = state_msg.xd
        self.r_arm_pose = state_msg.x
        self.r_arm_x_err = state_msg.x_err
        self.r_arm_x_d = state_msg.xd
        self.r_arm_F = state_msg.F

    def gripper_pose(self, which_arm="l"):
        if which_arm == 'l':
          robot_gripper = self.robot.left_gripper
          gripper_joint = 'l_gripper_joint'
        else:
          robot_gripper = self.robot.right_gripper
          gripper_joint = 'r_gripper_joint'
        joint_state = robot_gripper.pose(gripper_joint);
        print joint_state

    def gripper_open(self, which_arm='l'):
        '''
        Move the arm to the initial pose to be out of the way for viewing the
        tabletop
        '''
        if which_arm == 'l':
          robot_gripper = self.robot.left_gripper
        else:
          robot_gripper = self.robot.right_gripper
        rospy.loginfo('Opening %s_gripper' % which_arm)
        res = robot_gripper.open(block=True)
        rospy.loginfo('Opening %s_gripper' % which_arm)


    #
    # Controller setup methods
    #
    def init_joint_controllers(self):
        self.arm_mode = 'joint_mode'
        prefix = roslib.packages.get_pkg_dir('hrl_pr2_arms')+'/params/'
        rospy.loginfo('Setting right arm controller gains')
        self.cs.switch("r_arm_controller", "r_arm_controller",
                       prefix + "pr2_arm_controllers_push.yaml")
        rospy.loginfo('Setting left arm controller gains')
        self.cs.switch("l_arm_controller", "l_arm_controller",
                       prefix + "pr2_arm_controllers_push.yaml")
        rospy.sleep(self.post_controller_switch_sleep)

    def init_cart_controllers(self):
        self.arm_mode = 'cart_mode'
        if self.use_jinv:
            self.cs.carefree_switch('r', '%s'+self.base_cart_controller_name,
                                    '$(find tabletop_pushing)/params/j_inverse_params_low.yaml')
            self.cs.carefree_switch('l', '%s'+self.base_cart_controller_name,
                                    '$(find tabletop_pushing)/params/j_inverse_params_low.yaml')
        else:
            self.cs.carefree_switch('r', '%s'+self.base_cart_controller_name,
                                    '$(find tabletop_pushing)/params/j_transpose_task_params_pos_feedback_push.yaml')
            self.cs.carefree_switch('l', '%s'+self.base_cart_controller_name,
                                    '$(find tabletop_pushing)/params/j_transpose_task_params_pos_feedback_push.yaml')

        rospy.sleep(self.post_controller_switch_sleep)

    def init_vel_controllers(self):
        self.arm_mode = 'vel_mode'
        self.cs.carefree_switch('r', '%s'+self.base_vel_controller_name,
                                '$(find tabletop_pushing)/params/j_inverse_params_custom.yaml')
        self.cs.carefree_switch('l', '%s'+self.base_vel_controller_name,
                                '$(find tabletop_pushing)/params/j_inverse_params_custom.yaml')

    def switch_to_cart_controllers(self):
        if self.arm_mode != 'cart_mode':
            self.cs.carefree_switch('r', '%s'+self.base_cart_controller_name)
            self.cs.carefree_switch('l', '%s'+self.base_cart_controller_name)
            self.arm_mode = 'cart_mode'
            rospy.sleep(self.post_controller_switch_sleep)

    def switch_to_joint_controllers(self):
        if self.arm_mode != 'joint_mode':
            self.cs.carefree_switch('r', '%s_arm_controller')
            self.cs.carefree_switch('l', '%s_arm_controller')
            self.arm_mode = 'joint_mode'
            rospy.sleep(self.post_controller_switch_sleep)

    def switch_to_vel_controllers(self):
        if self.arm_mode != 'vel_mode':
            self.cs.carefree_switch('r', '%s'+self.base_vel_controller_name)
            self.cs.carefree_switch('l', '%s'+self.base_vel_controller_name)
            self.arm_mode = 'vel_mode'
            rospy.sleep(self.post_controller_switch_sleep)

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
        self.switch_to_cart_controllers()
        rospy.loginfo('Done initializing feedback pushing node.')
        rospy.spin()

if __name__ == '__main__':
    node = PositionFeedbackPushNode()
    node.run()
