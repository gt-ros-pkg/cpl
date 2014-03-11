#!/usr/bin/env python
# Software License Agreement (BSD License)
#
#  Copyright (c) 2014, Georgia Institute of Technology
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
from kinematics_msgs.srv import *
from std_msgs.msg import Float64MultiArray
from pr2_controllers_msgs.msg import *
from pr2_manipulation_controllers.msg import *
import tf
import numpy as np
from tabletop_pushing.srv import *
from tabletop_pushing.msg import *
from math import sin, cos, pi, fabs, sqrt, atan2
from push_learning import ControlAnalysisIO
import rbf_control
import sys
import os
from push_primitives import *
from model_based_pushing_control import *
from analyze_mpc import MPCSolutionIO, PushTrajectoryIO
from pushing_dynamics_models import *
from push_trajectory_generator import *
from util import sign, subPIAngle, trigAugState

_OFFLINE = False
_USE_LEARN_IO = True
_BUFFER_DATA = True
_SAVE_MPC_DATA = True
_USE_SHAPE_INFO_IN_SVM = False
_FORCE_SINGLE_SQP_SOLVE = False

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
    'elbowdownl': [-0.0088195719039858515, 1.2834828245284853, 0.20338442004843196, -1.5565279256852611, -0.096340012666916802, -1.0235018652439782, 1.7990893054129216],
    'gripper_place_r': [-0.58376927, 0.20531188, -1.98435142, -1.35661954, -10.97764169, -0.08100618, -6.48535644],
    'gripper_place_l': [0.9424233, 0.24058796, 2.04239987, -1.4576695 , -1.58940656, -0.5444458 , -6.23912942]
}

_ARM_ERROR_CODES = {}
_ARM_ERROR_CODES[-1] = 'PLANNING_FAILED'
_ARM_ERROR_CODES[1]='SUCCESS'
_ARM_ERROR_CODES[-2]='TIMED_OUT'
_ARM_ERROR_CODES[-3]='START_STATE_IN_COLLISION'
_ARM_ERROR_CODES[-4]='START_STATE_VIOLATES_PATH_CONSTRAINTS'
_ARM_ERROR_CODES[-5]='GOAL_IN_COLLISION'
_ARM_ERROR_CODES[-6]='GOAL_VIOLATES_PATH_CONSTRAINTS'
_ARM_ERROR_CODES[-7]='INVALID_ROBOT_STATE'
_ARM_ERROR_CODES[-8]='INCOMPLETE_ROBOT_STATE'
_ARM_ERROR_CODES[-9]='INVALID_PLANNER_ID'
_ARM_ERROR_CODES[-10]='INVALID_NUM_PLANNING_ATTEMPTS'
_ARM_ERROR_CODES[-11]='INVALID_ALLOWED_PLANNING_TIME'
_ARM_ERROR_CODES[-12]='INVALID_GROUP_NAME'
_ARM_ERROR_CODES[-13]='INVALID_GOAL_JOINT_CONSTRAINTS'
_ARM_ERROR_CODES[-14]='INVALID_GOAL_POSITION_CONSTRAINTS'
_ARM_ERROR_CODES[-15]='INVALID_GOAL_ORIENTATION_CONSTRAINTS'
_ARM_ERROR_CODES[-16]='INVALID_PATH_JOINT_CONSTRAINTS'
_ARM_ERROR_CODES[-17]='INVALID_PATH_POSITION_CONSTRAINTS'
_ARM_ERROR_CODES[-18]='INVALID_PATH_ORIENTATION_CONSTRAINTS'
_ARM_ERROR_CODES[-19]='INVALID_TRAJECTORY'
_ARM_ERROR_CODES[-20]='INVALID_INDEX'
_ARM_ERROR_CODES[-21]='JOINT_LIMITS_VIOLATED'
_ARM_ERROR_CODES[-22]='PATH_CONSTRAINTS_VIOLATED'
_ARM_ERROR_CODES[-23]='COLLISION_CONSTRAINTS_VIOLATED'
_ARM_ERROR_CODES[-24]='GOAL_CONSTRAINTS_VIOLATED'
_ARM_ERROR_CODES[-25]='JOINTS_NOT_MOVING'
_ARM_ERROR_CODES[-26]='TRAJECTORY_CONTROLLER_FAILED'
_ARM_ERROR_CODES[-27]='FRAME_TRANSFORM_FAILURE'
_ARM_ERROR_CODES[-28]='COLLISION_CHECKING_UNAVAILABLE'
_ARM_ERROR_CODES[-29]='ROBOT_STATE_STALE'
_ARM_ERROR_CODES[-30]='SENSOR_INFO_STALE'
_ARM_ERROR_CODES[-31]='NO_IK_SOLUTION'
_ARM_ERROR_CODES[-32]='INVALID_LINK_NAME'
_ARM_ERROR_CODES[-33]='IK_LINK_IN_COLLISION'
_ARM_ERROR_CODES[-34]='NO_FK_SOLUTION'
_ARM_ERROR_CODES[-35]='KINEMATICS_STATE_IN_COLLISION'
_ARM_ERROR_CODES[-36]='INVALID_TIMEOUT'

class PositionFeedbackPushNode:

    def __init__(self):
        rospy.init_node('position_feedback_push_node')
        self.controller_io = ControlAnalysisIO()
        self.use_learn_io = False
        self.use_gripper_place_joint_posture = False
        out_file_name = '/u/thermans/data/new/control_out_'+str(rospy.get_time())+'.txt'
        rospy.loginfo('Opening controller output file: '+out_file_name)
        if _USE_LEARN_IO:
            self.learn_io = None
        if _SAVE_MPC_DATA:
            self.mpc_q_star_io = None
            self.target_trajectory_io = None

        # Setup parameters
        self.learned_controller_base_path = rospy.get_param('~learned_controller_base_path',
                                                            '/u/thermans/cfg/controllers/')
        self.torso_z_offset = rospy.get_param('~torso_z_offset', 0.30)
        self.look_pt_x = rospy.get_param('~look_point_x', 0.7)
        self.head_pose_cam_frame = rospy.get_param('~head_pose_cam_frame',
                                                   'head_mount_kinect_rgb_link')
        self.default_torso_height = rospy.get_param('~default_torso_height',
                                                    0.30)
        self.gripper_raise_dist = rospy.get_param('~gripper_raise_dist',
                                                  0.05)
        self.gripper_pull_reverse_dist = rospy.get_param('~gripper_pull_reverse_dist',
                                                         0.1)
        self.gripper_pull_forward_dist = rospy.get_param('~gripper_pull_forward_dist',
                                                         0.15)
        self.gripper_push_reverse_dist = rospy.get_param('~gripper_push_reverse_dist',
                                                         0.03)
        self.high_arm_init_z = rospy.get_param('~high_arm_start_z', 0.15)
        self.lower_arm_init_z = rospy.get_param('~lower_arm_start_z', -0.10)
        self.post_controller_switch_sleep = rospy.get_param(
            '~arm_switch_sleep_time', 0.5)
        self.move_cart_check_hz = rospy.get_param('~move_cart_check_hz', 100)
        self.arm_done_moving_count_thresh = rospy.get_param(
            '~not_moving_count_thresh', 30)
        self.arm_done_moving_epc_count_thresh = rospy.get_param(
            '~not_moving_epc_count_thresh', 60)
        self.post_move_count_thresh = rospy.get_param('~post_move_count_thresh',
                                                      10)
        self.post_pull_count_thresh = rospy.get_param('~post_move_count_thresh',
                                                      20)
        self.pre_push_count_thresh = rospy.get_param('~pre_push_count_thresh',
                                                      50)
        self.still_moving_velocity = rospy.get_param('~moving_vel_thresh', 0.01)
        self.still_moving_angular_velocity = rospy.get_param('~angular_vel_thresh',
                                                             0.005)
        self.pressure_safety_limit = rospy.get_param('~pressure_limit',
                                                     2000)
        self.max_close_effort = rospy.get_param('~max_close_effort', 50)

        self.k_g = rospy.get_param('~push_control_goal_gain', 0.1)
        self.k_g_direct = rospy.get_param('~push_control_direct_goal_gain', 0.1);
        self.k_s_d = rospy.get_param('~push_control_spin_gain', 0.05)
        self.k_s_p = rospy.get_param('~push_control_position_spin_gain', 0.05)

        self.k_contact_g = rospy.get_param('~push_control_contact_goal_gain', 0.05)
        self.k_contact_d = rospy.get_param('~push_control_contact_gain', 0.05)
        self.k_alignment_spin_x = rospy.get_param('~push_control_contact_spin_gain', 0.25)

        self.k_h_f = rospy.get_param('~push_control_forward_heading_gain', 0.1)
        self.k_rotate_spin_x = rospy.get_param('~rotate_to_heading_hand_spin_gain', 0.0)
        self.max_heading_u_x = rospy.get_param('~max_heading_push_u_x', 0.2)
        self.max_heading_u_y = rospy.get_param('~max_heading_push_u_y', 0.01)
        self.max_goal_vel = rospy.get_param('~max_goal_vel', 0.015)
        self.straight_line_u_max = rospy.get_param('~straight_line_vel', 0.015)

        self.straight_v = rospy.get_param('~straight_line_goal_vel', 0.03)

        self.use_jinv = rospy.get_param('~use_jinv', True)
        self.use_cur_joint_posture = rospy.get_param('~use_joint_posture', True)

        self.servo_head_during_pushing = rospy.get_param('servo_head_during_pushing', False)
        self.RBF = None

        # MPC Parameters
        self.MPC = None
        self.trajectory_generator = None
        self.mpc_delta_t = rospy.get_param('~mpc_delta_t', 1.0/9.)
        self.mpc_state_space_dim = rospy.get_param('~mpc_n', 6)
        self.mpc_input_space_dim = rospy.get_param('~mpc_m', 3)
        self.mpc_lookahead_horizon = rospy.get_param('~mpc_H', 10)
        self.mpc_u_max = rospy.get_param('~mpc_max_u', 0.03)
        self.mpc_u_max_angular = rospy.get_param('~mpc_max_u_angular', 0.1)
        self.min_num_mpc_trajectory_steps = rospy.get_param('~num_mpc_trajectory_steps', 2)
        self.mpc_max_step_size = rospy.get_param('~mpc_max_step_size', 0.01)
        self.open_loop_segment_length = rospy.get_param('~sqp_open_loop_segment_length', 50)
        self.svr_base_path = rospy.get_param('~svr_base_path', '/cfg/SVR_DYN/')
        # Model performance analysis
        self.models_to_check_db = rospy.get_param('~model_checker_db_file_path', 'cfg/shape_db/shapes.txt')
        self.model_checker_output_path = rospy.get_param('~model_checker_output_path', 'cfg')
        self.trajectory_to_check = []
        self.check_model_performance = False
        self.use_err_dynamics = rospy.get_param('~use_error_dynamics', False)
        self.use_gp_dynamics = rospy.get_param('~use_gp_dynamics', False)
        self.sqp_max_iter = rospy.get_param('~sqp_max_iter', 50)
        self.mpc_max_iter = rospy.get_param('~mpc_max_iter', 10)

        # Set joint gains
        self.arm_mode = None
        # state Info
        self.l_arm_pose = None
        self.l_arm_x_err = None
        self.l_arm_x_d = None
        self.l_arm_F = None

        self.r_arm_pose = None
        self.r_arm_x_err = None
        self.r_arm_x_d = None
        self.r_arm_F = None

        self.goal_cb_count = 0

        # Setup cartesian controller parameters
        if self.use_jinv:
            self.base_cart_controller_name = '_cart_jinv_push'
            self.controller_state_msg = JinvTeleopControllerState
        else:
            self.base_cart_controller_name = '_cart_transpose_push'
            self.controller_state_msg = JTTaskControllerState
        self.base_vel_controller_name = '_cart_jinv_push'
        self.vel_controller_state_msg = JinvTeleopControllerState
        self.tf_listener = tf.TransformListener()

        if not _OFFLINE:
            self.cs = ControllerSwitcher()
            self.init_joint_controllers()
            self.init_cart_controllers()
            self.init_vel_controllers()

            # Setup arms
            rospy.loginfo('Creating pr2 object')
            self.robot = pr2.PR2(self.tf_listener, arms=True, base=False, use_kinematics=False)#, use_projector=False)
        # Arm Inverse Kinematics
        self.l_arm_ik_proxy = rospy.ServiceProxy('/pr2_left_arm_kinematics/get_ik', GetPositionIK)
        self.r_arm_ik_proxy = rospy.ServiceProxy('/pr2_right_arm_kinematics/get_ik', GetPositionIK)
        self.l_arm_ik_solver_proxy = rospy.ServiceProxy('/pr2_left_arm_kinematics/get_ik_solver_info',
                                                        GetKinematicSolverInfo)
        self.r_arm_ik_solver_proxy = rospy.ServiceProxy('/pr2_right_arm_kinematics/get_ik_solver_info',
                                                        GetKinematicSolverInfo)
        # Cartesian position and velocity controllers
        self.l_arm_cart_pub = rospy.Publisher('/l'+self.base_cart_controller_name+'/command_pose', PoseStamped)
        self.r_arm_cart_pub = rospy.Publisher('/r'+self.base_cart_controller_name+'/command_pose', PoseStamped)
        self.l_arm_cart_posture_pub = rospy.Publisher('/l'+self.base_cart_controller_name+'/command_posture',
                                                      Float64MultiArray)
        self.r_arm_cart_posture_pub = rospy.Publisher('/r'+self.base_cart_controller_name+'/command_posture',
                                                      Float64MultiArray)
        self.l_arm_cart_vel_pub = rospy.Publisher('/l'+self.base_vel_controller_name+'/command_twist', TwistStamped)
        self.r_arm_cart_vel_pub = rospy.Publisher('/r'+self.base_vel_controller_name+'/command_twist', TwistStamped)
        self.l_arm_vel_posture_pub = rospy.Publisher('/l'+self.base_vel_controller_name+'/command_posture',
                                                     Float64MultiArray)
        self.r_arm_vel_posture_pub = rospy.Publisher('/r'+self.base_vel_controller_name+'/command_posture',
                                                     Float64MultiArray)

        rospy.Subscriber('/l'+self.base_cart_controller_name+'/state', self.controller_state_msg,
                         self.l_arm_cart_state_callback)
        rospy.Subscriber('/r'+self.base_cart_controller_name+'/state', self.controller_state_msg,
                         self.r_arm_cart_state_callback)

        rospy.Subscriber('/l'+self.base_vel_controller_name+'/state', self.vel_controller_state_msg,
                         self.l_arm_vel_state_callback)
        rospy.Subscriber('/r'+self.base_vel_controller_name+'/state', self.vel_controller_state_msg,
                         self.r_arm_vel_state_callback)

        self.l_pressure_listener = pl.PressureListener('/pressure/l_gripper_motor', self.pressure_safety_limit)
        self.r_pressure_listener = pl.PressureListener('/pressure/r_gripper_motor', self.pressure_safety_limit)

        # Open callback services
        self.overhead_feedback_push_srv = rospy.Service(
            'overhead_feedback_push', FeedbackPush, self.overhead_feedback_push)
        self.overhead_feedback_post_push_srv = rospy.Service(
            'overhead_feedback_post_push', FeedbackPush,
            self.overhead_feedback_post_push)

        self.gripper_feedback_push_srv = rospy.Service(
            'gripper_feedback_push', FeedbackPush, self.gripper_feedback_push)
        self.gripper_feedback_post_push_srv = rospy.Service(
            'gripper_feedback_post_push', FeedbackPush,
            self.gripper_feedback_post_push)

        self.gripper_feedback_sweep_srv = rospy.Service(
            'gripper_feedback_sweep', FeedbackPush, self.gripper_feedback_sweep)
        self.gripper_feedback_post_sweep_srv = rospy.Service(
            'gripper_feedback_post_sweep', FeedbackPush,
            self.gripper_feedback_post_sweep)

        self.gripper_pre_push_srv = rospy.Service(
            'gripper_pre_push', FeedbackPush, self.gripper_pre_push)
        self.gripper_pre_sweep_srv = rospy.Service(
            'gripper_pre_sweep', FeedbackPush, self.gripper_pre_sweep)
        self.overhead_pre_push_srv = rospy.Service(
            'overhead_pre_push', FeedbackPush, self.overhead_pre_push)

        self.raise_and_look_serice = rospy.Service(
            'raise_and_look', RaiseAndLook, self.raise_and_look)
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

        rospy.loginfo('Closing %s_gripper' % which_arm)
        res = robot_gripper.close(block=True, effort=self.max_close_effort)
        rospy.loginfo('Closed %s_gripper' % which_arm)


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
        # (trans, rot) = self.tf_listener.lookupTransform('torso_lift_link', camera_frame, rospy.Time(0))
        # rospy.loginfo('Transform from torso_lift_link to ' + camera_frame + ' is ' + str(trans) + '\t' + str(rot))

        look_pt = np.asmatrix([self.look_pt_x, 0.0, -self.torso_z_offset])
        rospy.loginfo('Point head at ' + str(look_pt)+ ' in frame ' + camera_frame)
        head_res = self.robot.head.look_at(look_pt, 'torso_lift_link', camera_frame, wait=True)
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
    # Feedback Behavior functions
    #
    def gripper_feedback_push(self, request):
        feedback_cb = self.tracker_feedback_push
        return self.feedback_push_behavior(request, feedback_cb)

    def gripper_feedback_sweep(self, request):
        feedback_cb = self.tracker_feedback_push
        return self.feedback_push_behavior(request, feedback_cb)

    def overhead_feedback_push(self, request):
        feedback_cb = self.tracker_feedback_push
        return self.feedback_push_behavior(request, feedback_cb)

    def feedback_push_behavior(self, request, feedback_cb):
        response = FeedbackPushResponse()
        start_point = request.start_point.point
        wrist_yaw = request.wrist_yaw

        self.use_learn_io = (_USE_LEARN_IO and request.learn_out_file_name != '')
        if self.use_learn_io:
            self.learn_io = ControlAnalysisIO()
            self.learn_io.open_out_file(request.learn_out_file_name)
        if _SAVE_MPC_DATA:
            trajectory_file_name = request.learn_out_file_name[:-4]+'-trajectory.txt'
            mpc_file_name = request.learn_out_file_name[:-4]+'-q_star.txt'
            self.target_trajectory_io = PushTrajectoryIO()
            self.target_trajectory_io.open_out_file(trajectory_file_name)
            self.mpc_q_star_io = MPCSolutionIO()
            self.mpc_q_star_io.open_out_file(mpc_file_name)

        if request.check_model_performance:
            self.check_model_performance = True
            self.used_model_name = None
            self.trajectory_to_check = []

        if request.left_arm:
            which_arm = 'l'
        else:
            which_arm = 'r'

        self.active_arm = which_arm
        rospy.loginfo('Creating ac')
        ac = actionlib.SimpleActionClient('push_tracker',
                                          VisFeedbackPushTrackingAction)
        rospy.loginfo('waiting for server')
        if not ac.wait_for_server(rospy.Duration(5.0)):
            rospy.loginfo('Failed to connect to push tracker server')
            if self.use_learn_io:
                self.learn_io.close_out_file()
            if _SAVE_MPC_DATA:
                self.mpc_q_star_io.close_out_file()
                self.target_trajectory_io.close_out_file()
            return response
        ac.cancel_all_goals()
        self.feedback_count = 0
        self.goal_cb_count += 1

        if not _OFFLINE:
            self.switch_to_vel_controllers()
            self.stop_moving_vel(which_arm)

        done_cb = None
        active_cb = None
        goal = VisFeedbackPushTrackingGoal()
        goal.which_arm = which_arm
        goal.header.frame_id = request.start_point.header.frame_id
        goal.desired_pose = request.goal_pose
        self.desired_pose = request.goal_pose
        goal.controller_name = request.controller_name
        goal.proxy_name = request.proxy_name
        goal.behavior_primitive = request.behavior_primitive
        goal.open_loop_push = False

        # Load learned controller information if necessary
        if request.controller_name.startswith(RBF_CONTROLLER_PREFIX):
            self.setupRBFController(goal.controller_name)
        elif request.controller_name.startswith(AFFINE_CONTROLLER_PREFIX):
            self.AFFINE_A, self.AFFINE_B = self.loadAffineController(goal.controller_name)
        elif request.controller_name.startswith(MPC_CONTROLLER_PREFIX):
            # Load dynamics model being used here based on the controller name
            if request.controller_name == MPC_NAIVE_LINEAR_DYN:
                dyn_model = NaiveInputDynamics(self.mpc_delta_t, self.mpc_state_space_dim,
                                               self.mpc_input_space_dim)
            else:
                mpc_suffix = goal.controller_name[len(MPC_CONTROLLER_PREFIX):]
                self.used_model_name = mpc_suffix
                # NOTE: Add switch here if we get other non svm dynamics models
                base_path = roslib.packages.get_pkg_dir('tabletop_pushing') + self.svr_base_path
                param_file_string = base_path + mpc_suffix + '_params.txt'
                if self.use_err_dynamics:
                    dyn_model = SVRWithNaiveLinearPushDynamics(self.mpc_delta_t,
                                                               self.mpc_state_space_dim,
                                                               self.mpc_input_space_dim,
                                                               param_file_name = param_file_string)
                elif self.use_gp_dynamics:
                    dyn_model = GPPushDynamics(self.mpc_delta_t,
                                               self.mpc_state_space_dim,
                                               self.mpc_input_space_dim,
                                               param_file_name = param_file_string)
                else:
                    dyn_model = SVRPushDynamics(param_file_name = param_file_string)

            U_max = [self.mpc_u_max, self.mpc_u_max, self.mpc_u_max_angular]
            self.MPC =  ModelPredictiveController(dyn_model, self.mpc_lookahead_horizon, U_max,
                                                  max_iter = self.mpc_max_iter)

            # Create target trajectory to goal pose
            self.trajectory_generator = PiecewiseLinearTrajectoryGenerator(self.mpc_max_step_size,
                                                                           self.min_num_mpc_trajectory_steps)

        # TODO: Setup open loop generic stuff here
        open_loop = (request.controller_name.startswith(OPEN_LOOP_SQP_PREFIX) or
                     request.controller_name == OPEN_LOOP_STRAIGHT_LINE_CONTROLLER)
        if open_loop:
            # TODO: Send goal but don't block so we can record tracking data, have open_loop feedback_cb
            if request.controller_name.startswith(OPEN_LOOP_SQP_PREFIX):
                rospy.loginfo('Setting up open loop SQP controller')
                if request.controller_name == SQP_NAIVE_LINEAR_DYN:
                    rospy.loginfo('Using naive dynamics model')
                    dyn_model = NaiveInputDynamics(self.mpc_delta_t, self.mpc_state_space_dim,
                                                   self.mpc_input_space_dim)
                else:
                    sqp_suffix = goal.controller_name[len(OPEN_LOOP_SQP_PREFIX):]
                    # NOTE: Add switch here if we get other non svm dynamics models
                    base_path = roslib.packages.get_pkg_dir('tabletop_pushing') + self.svr_base_path
                    param_file_string = base_path + sqp_suffix + '_params.txt'
                    self.used_model_name = sqp_suffix
                    rospy.loginfo('Loading dynamics model: ' + param_file_string)
                    if self.use_err_dynamics:
                        dyn_model = SVRWithNaiveLinearPushDynamics(self.mpc_delta_t,
                                                                   self.mpc_state_space_dim,
                                                                   self.mpc_input_space_dim,
                                                                   param_file_name = param_file_string)
                    elif self.use_gp_dynamics:
                        dyn_model = GPPushDynamics(self.mpc_delta_t,
                                                   self.mpc_state_space_dim,
                                                   self.mpc_input_space_dim,
                                                   param_file_name = param_file_string)
                    else:
                        dyn_model = SVRPushDynamics(param_file_name = param_file_string)

                U_max = [self.mpc_u_max, self.mpc_u_max, self.mpc_u_max_angular]
                self.SQPOpt =  ModelPredictiveController(dyn_model, self.mpc_lookahead_horizon,
                                                         U_max, iprint_level=2, ftol=1.0E-3,
                                                         max_iter = self.sqp_max_iter)
                rospy.loginfo('Calling controller')
                response.action_aborted = not self.open_loop_sqp_controller(goal, request, which_arm, ac)
            elif request.controller_name == OPEN_LOOP_STRAIGHT_LINE_CONTROLLER:
                rospy.loginfo('Sending goal of: ' + str(goal.desired_pose))
                goal.open_loop_push = True
                ac.send_goal(goal)
                rospy.loginfo('Sent goal')

                rospy.loginfo('Setting up open loop straight line controller')
                response.action_aborted = not self.open_loop_straight_line_controller(goal, request, which_arm)

            # TODO: Force abort, but check the final goal_error...
            rospy.loginfo('Done with open loop push. Getting goal.')
            ac.cancel_goal()
            rospy.loginfo('Waiting for result')
            ac.wait_for_result(rospy.Duration(5))
            rospy.loginfo('Result received')
            result = ac.get_result()
            if result is not None:
                response.action_aborted = result.aborted
                if result.aborted:
                    rospy.loginfo('Result aborted!')
            else:
                rospy.logwarn('No result received!\n')
                response.action_aborted = True

        else:
            rospy.loginfo('Sending goal of: ' + str(goal.desired_pose))
            ac.send_goal(goal, done_cb, active_cb, feedback_cb)
            # Block until done
            rospy.loginfo('Waiting for result')
            ac.wait_for_result(rospy.Duration(0))
            rospy.loginfo('Result received')
            result = ac.get_result()
            response.action_aborted = result.aborted

        if request.check_model_performance:
            base_path = roslib.packages.get_pkg_dir('tabletop_pushing') + self.svr_base_path
            rospy.loginfo('base_path: ' + base_path)
            model_db_path = roslib.packages.get_pkg_dir('tabletop_pushing') + self.models_to_check_db
            model_checker = ModelPerformanceChecker(model_db_path, base_path)

            (best_model, ranked_models) = model_checker.choose_best_model(self.trajectory_to_check)
            self.check_model_performance = False
            # return the best model
            response.best_model = str(best_model)
            response.best_model_score = min(ranked_models.keys())
            response.used_model_score = model_checker.check_model_score(self.MPC.dyn_model, self.trajectory_to_check)
            # save ranked to disk
            if not os.path.exists(self.model_checker_output_path):
                os.mkdir(self.model_checker_output_path)
            output_file_name = self.model_checker_output_path + self.models_to_check_db.split('/')[-1]
            if self.used_model_name is None:
                self.used_model_name = request.controller_name
            model_checker.write_to_disk(ranked_models, output_file_name, response.used_model_score,
                                        self.used_model_name)

        # Cleanup and save data
        if not _OFFLINE:
            self.stop_moving_vel(which_arm)

        if self.use_learn_io:
            if _BUFFER_DATA:
                self.learn_io.write_buffer_to_disk()
                if _SAVE_MPC_DATA:
                    self.mpc_q_star_io.write_buffer_to_disk()
                    self.target_trajectory_io.write_buffer_to_disk()
            self.learn_io.close_out_file()
            if _SAVE_MPC_DATA:
                self.mpc_q_star_io.close_out_file()
                self.target_trajectory_io.close_out_file()

        return response

    def tracker_feedback_push(self, feedback):
        if self.feedback_count == 0:
            self.theta0 = feedback.x.theta
            self.x0 = feedback.x.x
            self.y0 = feedback.x.y
        which_arm = self.active_arm
        if which_arm == 'l':
            cur_ee_pose = self.l_arm_pose
        else:
            cur_ee_pose = self.r_arm_pose
        if _OFFLINE:
            cur_pose_tool_frame = PoseStamped()
            cur_pose_tool_frame.pose.position.x = 0.0
            cur_pose_tool_frame.pose.position.y = 0.0
            cur_pose_tool_frame.pose.position.z = 0.0
            cur_pose_tool_frame.pose.orientation.x = 0.0
            cur_pose_tool_frame.pose.orientation.y = 0.0
            cur_pose_tool_frame.pose.orientation.z = 0.0
            cur_pose_tool_frame.pose.orientation.w = 1.0
            if which_arm == 'l':
                cur_pose_tool_frame.header.frame_id = 'l_gripper_tool_frame'
            else:
                cur_pose_tool_frame.header.frame_id = 'r_gripper_tool_frame'
            cur_ee_pose = self.tf_listener.transformPose('torso_lift_link', cur_pose_tool_frame)

        if self.feedback_count % 5 == 0:
            rospy.loginfo('X_goal: (' + str(self.desired_pose.x) + ', ' +
                          str(self.desired_pose.y) + ', ' +
                          str(self.desired_pose.theta) + ')')
            rospy.loginfo('X: (' + str(feedback.x.x) + ', ' + str(feedback.x.y)
                          + ', ' + str(feedback.x.theta) + ')')
            rospy.loginfo('X_dot: (' + str(feedback.x_dot.x) + ', ' +
                          str(feedback.x_dot.y) + ', ' +
                          str(feedback.x_dot.theta) + ')')
            rospy.loginfo('X_error: (' + str(self.desired_pose.x - feedback.x.x) + ', ' +
                          str(self.desired_pose.y - feedback.x.y) + ', ' +
                          str(self.desired_pose.theta - feedback.x.theta) + ')')

        # TODO: Create options for non-velocity control updates, separate things more
        # NOTE: Add new pushing visual feedback controllers here
        if feedback.controller_name == ROTATE_TO_HEADING:
            update_twist = self.rotateHeadingControllerPalm(feedback, self.desired_pose, which_arm, cur_ee_pose)
        elif feedback.controller_name == CENTROID_CONTROLLER:
            update_twist = self.centroidAlignmentController(feedback, self.desired_pose, cur_ee_pose)
        elif feedback.controller_name == DIRECT_GOAL_CONTROLLER:
            update_twist = self.directGoalController(feedback, self.desired_pose)
        elif feedback.controller_name == DIRECT_GOAL_GRIPPER_CONTROLLER:
            update_twist = self.directGoalGripperController(feedback, self.desired_pose, cur_ee_pose)
        elif feedback.controller_name == STRAIGHT_LINE_CONTROLLER:
            update_twist = self.straightLineController(feedback, self.desired_pose)
        elif feedback.controller_name == SPIN_COMPENSATION:
            update_twist = self.spinCompensationController(feedback, self.desired_pose)
        elif feedback.controller_name.startswith(RBF_CONTROLLER_PREFIX):
            update_twist = self.RBFFeedbackController(feedback)
        elif feedback.controller_name.startswith(AFFINE_CONTROLLER_PREFIX):
            update_twist = self.affineFeedbackController(feedback, self.AFFINE_A, self.AFFINE_B)
        elif feedback.controller_name.startswith(MPC_CONTROLLER_PREFIX):
            update_twist = self.MPCFeedbackController(feedback, cur_ee_pose, self.desired_pose)

        if self.feedback_count % 5 == 0:
            rospy.loginfo('q_dot: (' + str(update_twist.twist.linear.x) + ', ' +
                          str(update_twist.twist.linear.y) + ', ' +
                          str(update_twist.twist.linear.z) + ')')
            rospy.loginfo('omega: (' + str(update_twist.twist.angular.x) + ', ' +
                          str(update_twist.twist.angular.y) + ', ' +
                          str(update_twist.twist.angular.z) + ')\n')

        if self.use_learn_io:
            if _BUFFER_DATA:
                log_data = self.learn_io.buffer_line
            else:
                log_data = self.learn_io.write_line
            log_data(feedback.x, feedback.x_dot, self.desired_pose, self.theta0,
                                      update_twist.twist, update_twist.header.stamp.to_sec(),
                                      cur_ee_pose.pose, feedback.header.seq, feedback.z, feedback.shape_descriptor)

        if self.servo_head_during_pushing and not _OFFLINE:
            look_pt = np.asmatrix([feedback.x.x,
                                   feedback.x.y,
                                   feedback.z])
            head_res = self.robot.head.look_at(look_pt, feedback.header.frame_id,
                                               self.head_pose_cam_frame)
        if not _OFFLINE:
            self.update_vel(update_twist, which_arm)
        self.feedback_count += 1

    def open_loop_sqp_controller(self, goal, push_request, which_arm, ac):
        if _FORCE_SINGLE_SQP_SOLVE:
            return self.open_loop_sqp_controller_batch(goal, push_request, which_arm, ac)
        else:
            return self.open_loop_sqp_controller_segments(goal, push_request, which_arm, ac)

    def open_loop_sqp_controller_batch(self, goal, push_request, which_arm, ac):
        '''
        Returns True if the open loop control was correctly applied, False if fails or aborted
        '''
        # Generate push trajectory
        trajectory_generator = PiecewiseLinearTrajectoryGenerator(self.mpc_max_step_size,
                                                                  self.min_num_mpc_trajectory_steps)
        pose_list = [goal.desired_pose]
        x_d = trajectory_generator.generate_trajectory(push_request.obj_start_pose, pose_list)

        if which_arm == 'l':
            ee_pose = self.l_arm_pose
        else:
            ee_pose = self.r_arm_pose
        if _OFFLINE:
            cur_pose_tool_frame = PoseStamped()
            cur_pose_tool_frame.pose.position.x = 0.0
            cur_pose_tool_frame.pose.position.y = 0.0
            cur_pose_tool_frame.pose.position.z = 0.0
            cur_pose_tool_frame.pose.orientation.x = 0.0
            cur_pose_tool_frame.pose.orientation.y = 0.0
            cur_pose_tool_frame.pose.orientation.z = 0.0
            cur_pose_tool_frame.pose.orientation.w = 1.0
            if which_arm == 'l':
                cur_pose_tool_frame.header.frame_id = 'l_gripper_tool_frame'
            else:
                cur_pose_tool_frame.header.frame_id = 'r_gripper_tool_frame'
            ee_pose = self.tf_listener.transformPose('torso_lift_link', cur_pose_tool_frame)


        [_, _, ee_phi] = tf.transformations.euler_from_quaternion(np.array([ee_pose.pose.orientation.x,
                                                                            ee_pose.pose.orientation.y,
                                                                            ee_pose.pose.orientation.z,
                                                                            ee_pose.pose.orientation.w]))
        x0 = np.asarray([push_request.obj_start_pose.x, push_request.obj_start_pose.y,
                         push_request.obj_start_pose.theta,
                         ee_pose.pose.position.x, ee_pose.pose.position.y, ee_phi])

        # Get push plan using SQP via MPC class
        control_tape = []
        self.SQPOpt.H = len(x_d)-1
        self.SQPOpt.regenerate_bounds()
        rospy.loginfo('Solving for trajectory of length ' + str(len(x_d)))
        q_star = self.SQPOpt.feedbackControl(x0, x_d)

        # Loop through the plan converting the SQP results into twist messages
        [x_star, u_star] = get_x_u_from_q(q_star, x0, self.SQPOpt.H, self.SQPOpt.n, self.SQPOpt.m)
        for U_k in u_star:
            u_x = U_k[0]
            u_y = U_k[1]
            u_phi = U_k[2]
            u = TwistStamped()
            u.header.frame_id = 'torso_lift_link'
            u.twist.linear.x = max( min(u_x, self.mpc_u_max), -self.mpc_u_max)
            u.twist.linear.y = max( min(u_y, self.mpc_u_max), -self.mpc_u_max)
            u.twist.linear.z = 0.0
            u.twist.angular.x = 0.0
            u.twist.angular.y = 0.0
            if push_request.controller_name == SQP_NAIVE_LINEAR_DYN:
                u.twist.angular.z = 0.0
            else:
                u.twist.angular.z = max( min(u_phi, self.mpc_u_max_angular), -self.mpc_u_max_angular)
            control_tape.append(u)

        # TODO: Setup data logging here
        if _SAVE_MPC_DATA:
            if _BUFFER_DATA:
                self.target_trajectory_io.buffer_line(0, x_d)
                self.mpc_q_star_io.buffer_line(0, q_star)
            else:
                self.target_trajectory_io.write_line(0, x_d)
                self.mpc_q_star_io.write_line(0, q_star)

        # Send goal information to save visuale data
        rospy.loginfo('Sending goal of: ' + str(goal.desired_pose))
        goal.open_loop_push = True
        ac.send_goal(goal)
        rospy.loginfo('Sent goal')

        # Run the control tape
        r_out = rospy.Rate(100)
        for play_count, u in enumerate(control_tape):
            if play_count % 5 == 0:
                rospy.loginfo('q_dot: (' +
                              str(u.twist.linear.x) + ', ' +
                              str(u.twist.linear.y) + ', ' +
                              str(u.twist.linear.z) + ')')
                rospy.loginfo('omega: (' +
                              str(u.twist.angular.x) + ', ' +
                              str(u.twist.angular.y) + ', ' +
                              str(u.twist.angular.z) + ')\n')
            timeout_at = rospy.get_time() + self.mpc_delta_t

            # Apply control
            while timeout_at > rospy.get_time():
                u.header.stamp = rospy.Time.now()
                self.update_vel(u, which_arm)
                r_out.sleep()
        self.stop_moving_vel(which_arm)

        return True

    def open_loop_sqp_controller_segments(self, goal, push_request, which_arm, ac):
        '''
        Returns True if the open loop control was correctly applied, False if fails or aborted
        '''
        # Generate push trajectory
        trajectory_generator = PiecewiseLinearTrajectoryGenerator(self.mpc_max_step_size,
                                                                  self.min_num_mpc_trajectory_steps)
        pose_list = [goal.desired_pose]
        x_d = trajectory_generator.generate_trajectory(push_request.obj_start_pose, pose_list)

        if which_arm == 'l':
            ee_pose = self.l_arm_pose
        else:
            ee_pose = self.r_arm_pose
        if _OFFLINE:
            cur_pose_tool_frame = PoseStamped()
            cur_pose_tool_frame.pose.position.x = 0.0
            cur_pose_tool_frame.pose.position.y = 0.0
            cur_pose_tool_frame.pose.position.z = 0.0
            cur_pose_tool_frame.pose.orientation.x = 0.0
            cur_pose_tool_frame.pose.orientation.y = 0.0
            cur_pose_tool_frame.pose.orientation.z = 0.0
            cur_pose_tool_frame.pose.orientation.w = 1.0
            if which_arm == 'l':
                cur_pose_tool_frame.header.frame_id = 'l_gripper_tool_frame'
            else:
                cur_pose_tool_frame.header.frame_id = 'r_gripper_tool_frame'
            ee_pose = self.tf_listener.transformPose('torso_lift_link', cur_pose_tool_frame)


        [_, _, ee_phi] = tf.transformations.euler_from_quaternion(np.array([ee_pose.pose.orientation.x,
                                                                            ee_pose.pose.orientation.y,
                                                                            ee_pose.pose.orientation.z,
                                                                            ee_pose.pose.orientation.w]))
        x0 = np.asarray([push_request.obj_start_pose.x, push_request.obj_start_pose.y,
                         push_request.obj_start_pose.theta,
                         ee_pose.pose.position.x, ee_pose.pose.position.y, ee_phi])

        # Get push plan using SQP via MPC class
        # HACK: Turned off segments and just solve for the full trajectory again, now that things go faster...
        H = self.open_loop_segment_length
        num_segments = int(ceil(float(len(x_d))/H))

        control_tape = []
        q_super_star = np.asarray([])
        # Solve for each segment
        for i in xrange(num_segments):
            # Get current segment of the desired chain x_d_i
            x_d_i = x_d[i*H:(i+1)*H]
            # set H to full sequence length x_d_i
            if len(x_d_i) < 2:
                rospy.logwarn('x_d_i is too short! Breaking!')
                break
            self.SQPOpt.H = len(x_d_i)-1
            self.SQPOpt.regenerate_bounds()
            rospy.loginfo('Solving optimization for push plan segment ' + str(i+1) + ' of ' + str(num_segments))
            rospy.loginfo('Current segment has ' + str(len(x_d_i)) + ' steps out of a total ' + str(len(x_d)) )
            q_star = self.SQPOpt.feedbackControl(x0, x_d_i)
            q_super_star = np.concatenate((q_super_star, q_star))

            # Loop through the plan converting the SQP results into twist messages
            [x_star, u_star] = get_x_u_from_q(q_star, x0, self.SQPOpt.H, self.SQPOpt.n, self.SQPOpt.m)
            for U_k in u_star:
                u_x = U_k[0]
                u_y = U_k[1]
                u_phi = U_k[2]
                u = TwistStamped()
                u.header.frame_id = 'torso_lift_link'
                u.twist.linear.x = max( min(u_x, self.mpc_u_max), -self.mpc_u_max)
                u.twist.linear.y = max( min(u_y, self.mpc_u_max), -self.mpc_u_max)
                u.twist.linear.z = 0.0
                u.twist.angular.x = 0.0
                u.twist.angular.y = 0.0
                u.twist.angular.z = max( min(u_phi, self.mpc_u_max_angular), -self.mpc_u_max_angular)
                control_tape.append(u)

            # update x0 based on final poase in q_star
            x0 = np.asarray(x_star[-1])

        # TODO: Setup data logging here
        if _SAVE_MPC_DATA:
            if _BUFFER_DATA:
                self.target_trajectory_io.buffer_line(0, x_d)
                self.mpc_q_star_io.buffer_line(0, q_super_star)
            else:
                self.target_trajectory_io.write_line(0, x_d)
                self.mpc_q_star_io.write_line(0, q_super_star)

        # Send goal information to save visuale data
        rospy.loginfo('Sending goal of: ' + str(goal.desired_pose))
        goal.open_loop_push = True
        ac.send_goal(goal)
        rospy.loginfo('Sent goal')

        # Run the control tape
        r_out = rospy.Rate(100)
        for play_count, u in enumerate(control_tape):
            if play_count % H == 0:
                rospy.loginfo('Executing segment: ' + str(play_count/H+1))
            # if play_count % 5 == 0:
            #     rospy.loginfo('q_dot: (' +
            #                   str(u.twist.linear.x) + ', ' +
            #                   str(u.twist.linear.y) + ', ' +
            #                   str(u.twist.linear.z) + ')')
            #     rospy.loginfo('omega: (' +
            #                   str(u.twist.angular.x) + ', ' +
            #                   str(u.twist.angular.y) + ', ' +
            #                   str(u.twist.angular.z) + ')\n')
            timeout_at = rospy.get_time() + self.mpc_delta_t

            # Apply control
            while timeout_at > rospy.get_time():
                u.header.stamp = rospy.Time.now()
                self.update_vel(u, which_arm)
                r_out.sleep()
        self.stop_moving_vel(which_arm)

        return True

    def open_loop_straight_line_controller(self, goal, push_request, which_arm):
        # Get goal vector
        x_error = goal.desired_pose.x - push_request.obj_start_pose.x
        y_error = goal.desired_pose.y - push_request.obj_start_pose.y

        # Get pushing velocity
        u = TwistStamped()
        u.header.frame_id = 'torso_lift_link'
        u.header.stamp = rospy.Time.now()
        u.twist.linear.z = 0.0
        u.twist.angular.x = 0.0
        u.twist.angular.y = 0.0
        u.twist.angular.z = 0.0
        if fabs(x_error) > fabs(y_error):
            u.twist.linear.x = sign(x_error)*self.straight_line_u_max
            u.twist.linear.y = y_error/fabs(x_error)*self.straight_line_u_max
            push_time = fabs(x_error)/fabs(u.twist.linear.x)
        else:
            u.twist.linear.y = sign(y_error)*self.straight_line_u_max
            u.twist.linear.x = x_error/fabs(y_error)*self.straight_line_u_max
            push_time = fabs(y_error)/fabs(u.twist.linear.y)

        rospy.loginfo('Pushing for ' + str(push_time) + ' seconds with velocity: ['+ str(u.twist.linear.x) + ', ' +
                      str(u.twist.linear.y) + ']')
        # TODO: Log something?

        # Push for required time
        r = rospy.Rate(100)
        timeout_at = rospy.get_time() + push_time
        while timeout_at > rospy.get_time():
            u.header.stamp = rospy.Time.now()
            self.update_vel(u, which_arm)
            r.sleep()

        self.stop_moving_vel(which_arm)

        return True
    #
    # Feedback Controller functions
    #

    def spinCompensationController(self, cur_state, desired_state):
        u = TwistStamped()
        u.header.frame_id = 'torso_lift_link'
        u.header.stamp = rospy.Time.now()
        u.twist.linear.z = 0.0
        u.twist.angular.x = 0.0
        u.twist.angular.y = 0.0
        u.twist.angular.z = 0.0
        # Push centroid towards the desired goal
        x_error = desired_state.x - cur_state.x.x
        y_error = desired_state.y - cur_state.x.y
        t_error = subPIAngle(desired_state.theta - cur_state.x.theta)
        t0_error = subPIAngle(self.theta0 - cur_state.x.theta)
        goal_x_dot = self.k_g*x_error
        goal_y_dot = self.k_g*y_error
        # Add in direction to corect for spinning
        goal_angle = atan2(goal_y_dot, goal_x_dot)
        transform_angle = goal_angle
        # transform_angle = t0_error
        spin_x_dot = -sin(transform_angle)*(self.k_s_d*cur_state.x_dot.theta +
                                            -self.k_s_p*t0_error)
        spin_y_dot =  cos(transform_angle)*(self.k_s_d*cur_state.x_dot.theta +
                                            -self.k_s_p*t0_error)
        # TODO: Clip values that get too big
        u.twist.linear.x = goal_x_dot + spin_x_dot
        u.twist.linear.y = goal_y_dot + spin_y_dot
        if self.feedback_count % 5 == 0:
            rospy.loginfo('q_goal_dot: (' + str(goal_x_dot) + ', ' +
                          str(goal_y_dot) + ')')
            rospy.loginfo('q_spin_dot: (' + str(spin_x_dot) + ', ' +
                          str(spin_y_dot) + ')')
        return u

    def rotateHeadingControllerPalm(self, cur_state, desired_state, which_arm, ee_pose):
        u = TwistStamped()
        u.header.frame_id = which_arm+'_gripper_palm_link'
        u.header.stamp = rospy.Time.now()
        u.twist.linear.x = 0.0
        u.twist.linear.y = 0.0
        # TODO: Track object rotation with gripper angle
        u.twist.angular.x = -self.k_rotate_spin_x*cur_state.x_dot.theta
        u.twist.angular.y = 0.0
        u.twist.angular.z = 0.0
        t_error = subPIAngle(desired_state.theta - cur_state.x.theta)
        s_theta = sign(t_error)
        t_dist = fabs(t_error)
        heading_x_dot = self.k_h_f*t_dist
        u.twist.linear.z = min(heading_x_dot, self.max_heading_u_x)
        if self.feedback_count % 5 == 0:
            rospy.loginfo('heading_x_dot: (' + str(heading_x_dot) + ')')
            rospy.loginfo('heading_rotate: (' + str(u.twist.angular.x) + ')')
        return u

    def spinCircleStuff(self, cur_state, desired_state, which_arm):
        r = 1.0 # Spin radius
        T = 32
        t = 0 # TODO: Increment / send in
        theta_dot = 2*pi/T # Constant rotational speed
        theta = theta_dot*t # Angle around spin circle
        x_circle = r*sin(theta)
        y_circle = r*cos(theta)
        x_dot_circle = r*cos(theta)*theta_dot
        y_dot_circle = -r*sin(theta)*theta_dot

    def centroidAlignmentController(self, cur_state, desired_state, ee_pose):
        u = TwistStamped()
        u.header.frame_id = 'torso_lift_link'
        u.header.stamp = rospy.Time.now()
        u.twist.linear.z = 0.0
        u.twist.angular.x = 0.0
        u.twist.angular.y = 0.0
        u.twist.angular.z = 0.0

        # Push centroid towards the desired goal
        centroid = cur_state.x
        ee = ee_pose.pose.position
        x_error = desired_state.x - centroid.x
        y_error = desired_state.y - centroid.y
        goal_x_dot = self.k_contact_g*x_error
        goal_y_dot = self.k_contact_g*y_error

        # Add in direction to corect for not pushing through the centroid
        goal_angle = atan2(goal_y_dot, goal_x_dot)
        transform_angle = goal_angle
        m = (((ee.x - centroid.x)*x_error + (ee.y - centroid.y)*y_error) /
             sqrt(x_error*x_error + y_error*y_error))
        tan_pt_x = centroid.x + m*x_error
        tan_pt_y = centroid.y + m*y_error
        contact_pt_x_dot = self.k_contact_d*(tan_pt_x - ee.x)
        contact_pt_y_dot = self.k_contact_d*(tan_pt_y - ee.y)
        # Compensate for spinning with wrist rotation
        if cur_state.behavior_primitive == OVERHEAD_PUSH:
            contact_pt_z_angular_dot = -self.k_alignment_spin_x*cur_state.x_dot.theta
        else:
            contact_pt_z_angular_dot = 0.0
        # Clip values that get too big
        u.twist.linear.x = max( min(goal_x_dot + contact_pt_x_dot, self.mpc_u_max), -self.mpc_u_max)
        u.twist.linear.y = max( min(goal_y_dot + contact_pt_y_dot, self.mpc_u_max), -self.mpc_u_max)
        u.twist.angular.z = max( min(contact_pt_z_angular_dot, self.mpc_u_max_angular), -self.mpc_u_max_angular)

        if self.feedback_count % 5 == 0:
            rospy.loginfo('tan_pt: (' + str(tan_pt_x) + ', ' + str(tan_pt_y) + ')')
            rospy.loginfo('ee: (' + str(ee.x) + ', ' + str(ee.y) + ')')
            rospy.loginfo('q_goal_dot: (' + str(goal_x_dot) + ', ' +
                          str(goal_y_dot) + ')')
            rospy.loginfo('contact_pt_x_dot: (' + str(contact_pt_x_dot) + ', ' +
                          str(contact_pt_y_dot) + ')')
        return u

    def directGoalController(self, cur_state, desired_state):
        u = TwistStamped()
        u.header.frame_id = 'torso_lift_link'
        u.header.stamp = rospy.Time.now()
        u.twist.linear.z = 0.0
        u.twist.angular.x = 0.0
        u.twist.angular.y = 0.0
        u.twist.angular.z = 0.0

        # Push centroid towards the desired goal
        centroid = cur_state.x
        x_error = desired_state.x - centroid.x
        y_error = desired_state.y - centroid.y
        goal_x_dot = max(min(self.k_g_direct*x_error, self.max_goal_vel), -self.max_goal_vel)
        goal_y_dot = max(min(self.k_g_direct*y_error, self.max_goal_vel), -self.max_goal_vel)

        u.twist.linear.x = goal_x_dot
        u.twist.linear.y = goal_y_dot
        if self.feedback_count % 5 == 0:
            rospy.loginfo('q_goal_dot: (' + str(goal_x_dot) + ', ' +
                          str(goal_y_dot) + ')')
        return u

    def straightLineController(self, cur_state, desired_state):
        u = TwistStamped()
        u.header.frame_id = 'torso_lift_link'
        u.header.stamp = rospy.Time.now()
        u.twist.linear.z = 0.0
        u.twist.angular.x = 0.0
        u.twist.angular.y = 0.0
        u.twist.angular.z = 0.0

        # Push centroid towards the desired goal
        centroid = cur_state.x
        x_error = desired_state.x - cur_state.init_x.x
        y_error = desired_state.y - cur_state.init_x.y
        goal_x_dot = x_error/(fabs(x_error)+fabs(y_error))*self.straight_v
        goal_y_dot = y_error/(fabs(x_error)+fabs(y_error))*self.straight_v

        u.twist.linear.x = goal_x_dot
        u.twist.linear.y = goal_y_dot
        if self.feedback_count % 5 == 0:
            rospy.loginfo('q_goal_dot: (' + str(goal_x_dot) + ', ' +
                          str(goal_y_dot) + ')')
        return u

    def directGoalGripperController(self, cur_state, desired_state, ee_pose):
        u = TwistStamped()
        u.header.frame_id = 'torso_lift_link'
        u.header.stamp = rospy.Time.now()
        u.twist.linear.z = 0.0
        u.twist.angular.x = 0.0
        u.twist.angular.y = 0.0
        u.twist.angular.z = 0.0

        ee = ee_pose.pose.position

        # Push centroid towards the desired goal
        centroid = cur_state.x
        x_error = desired_state.x - ee.x
        y_error = desired_state.y - ee.y
        goal_x_dot = max(min(self.k_g_direct*x_error, self.max_goal_vel), -self.max_goal_vel)
        goal_y_dot = max(min(self.k_g_direct*y_error, self.max_goal_vel), -self.max_goal_vel)

        u.twist.linear.x = goal_x_dot
        u.twist.linear.y = goal_y_dot
        if self.feedback_count % 5 == 0:
            rospy.loginfo('ee_x: (' + str(ee.x) + ', ' + str(ee.y) + ', ' + str(ee.z) + ')')
            rospy.loginfo('q_goal_dot: (' + str(goal_x_dot) + ', ' + str(goal_y_dot) + ')')
        return u

    def affineFeedbackController(self, cur_state, A, b):
        u = TwistStamped()
        u.header.frame_id = 'torso_lift_link'
        u.header.stamp = rospy.Time.now()
        u.twist.linear.z = 0.0
        u.twist.angular.x = 0.0
        u.twist.angular.y = 0.0
        u.twist.angular.z = 0.0

        X = np.asarray(cur_state.x)

        u_t = A*X+b
        # TODO: Make this dependent on the specified control state
        u.twist.linear.x = u_t[0]
        u.twist.linear.y = u_t[1]
        return u

    def RBFFeedbackController(self, cur_state):
        u = TwistStamped()
        u.header.frame_id = 'torso_lift_link'
        u.header.stamp = rospy.Time.now()
        u.twist.linear.z = 0.0
        u.twist.angular.x = 0.0
        u.twist.angular.y = 0.0
        u.twist.angular.z = 0.0

        # Replace angles with sin(theta), cos(theta)
        # TODO: Double check this after pilco change
        ndx = [4]
        X = trigAugState(np.asarray(cur_state.x), ndx, True)
        D = np.zeros((P.shape[1], 1))
        u_t = self.RBF.feedbackControl(X)
        # TODO: Make this mapping dependent on the specified control state, read from file
        u.twist.linear.x = u_t[0]
        u.twist.linear.y = u_t[1]
        return u

    def MPCFeedbackController(self, cur_state, ee_pose, desired_pose):
        # TODO: Add ability to switch model at each feedback call (using model predictive error, etc)

        k0 = cur_state.header.seq
        # Get updated list for forward trajectory
        pose_list = [desired_pose]
        x_d_i = self.trajectory_generator.generate_trajectory(cur_state.x, pose_list)

        self.MPC.H = min(self.mpc_lookahead_horizon, len(x_d_i)-1)
        self.MPC.regenerate_bounds()
        rospy.loginfo('MPC.H = ' + str(self.MPC.H))

        # TODO: Choose if XTRA is used from SVR dynamics model flags
        # Feature vector for the dynamics model
        if _USE_SHAPE_INFO_IN_SVM:
            xtra = cur_state.shape_descriptor
        else:
            xtra = []
        [_, _, ee_phi] = tf.transformations.euler_from_quaternion(np.array([ee_pose.pose.orientation.x,
                                                                            ee_pose.pose.orientation.y,
                                                                            ee_pose.pose.orientation.z,
                                                                            ee_pose.pose.orientation.w]))

        x0 = [cur_state.x.x, cur_state.x.y, cur_state.x.theta,
              ee_pose.pose.position.x, ee_pose.pose.position.y, ee_phi]
        q_star = self.MPC.feedbackControl(x0, x_d_i, xtra)

        if _SAVE_MPC_DATA:
            if _BUFFER_DATA:
                self.target_trajectory_io.buffer_line(k0, x_d_i)
                self.mpc_q_star_io.buffer_line(k0, q_star)
            else:
                self.target_trajectory_io.write_line(k0, x_d_i)
                self.mpc_q_star_io.write_line(k0, q_star)

        if self.check_model_performance:
            u_k = [max( min(q_star[0], self.mpc_u_max), -self.mpc_u_max),
                   max( min(q_star[1], self.mpc_u_max), -self.mpc_u_max),
                   max( min(q_star[2], self.mpc_u_max_angular), -self.mpc_u_max_angular)]
            self.trajectory_to_check.append((x0, u_k))

        # Use previous to initialize at next time step
        self.MPC.init_from_previous = True

        u = TwistStamped()
        u.header.frame_id = 'torso_lift_link'
        u.header.stamp = rospy.Time.now()
        u.twist.linear.x = max( min(q_star[0], self.mpc_u_max), -self.mpc_u_max)
        u.twist.linear.y = max( min(q_star[1], self.mpc_u_max), -self.mpc_u_max)
        if fabs(u.twist.linear.x) < 0.001:
            u.twist.linear.x = 0.0
        if fabs(u.twist.linear.y) < 0.001:
            u.twist.linear.y = 0.0
        u.twist.linear.z = 0.0
        u.twist.angular.x = 0.0
        u.twist.angular.y = 0.0
        if cur_state.controller_name == MPC_NAIVE_LINEAR_DYN:
            u.twist.angular.z = 0.0
        else:
            u.twist.angular.z = max( min(q_star[2], self.mpc_u_max_angular), -self.mpc_u_max_angular)
        return u
    #
    # Post behavior functions
    #
    def overhead_feedback_post_push(self, request):
        response = FeedbackPushResponse()
        start_point = request.start_point.point
        wrist_yaw = request.wrist_yaw

        if request.left_arm:
            ready_joints = LEFT_ARM_READY_JOINTS
            if request.high_arm_init:
                ready_joints = LEFT_ARM_PULL_READY_JOINTS
            which_arm = 'l'
            wrist_pitch = 0.5*pi
            robot_gripper = self.robot.left_gripper
        else:
            ready_joints = RIGHT_ARM_READY_JOINTS
            if request.high_arm_init:
                ready_joints = RIGHT_ARM_PULL_READY_JOINTS
            which_arm = 'r'
            wrist_pitch = -0.5*pi
            robot_gripper = self.robot.right_gripper

        rospy.logdebug('Moving gripper up')
        pose_err, err_dist = self.move_relative_gripper(
            np.matrix([-self.gripper_raise_dist, 0.0, 0.0]).T, which_arm,
            move_cart_count_thresh=self.post_move_count_thresh)
        rospy.logdebug('Done moving up')
        rospy.logdebug('Pushing reverse')
        pose_err, err_dist = self.move_relative_gripper(
            np.matrix([0.0, 0.0, -self.gripper_push_reverse_dist]).T, which_arm,
            move_cart_count_thresh=self.post_move_count_thresh)
        rospy.loginfo('Done pushing reverse')


        rospy.logdebug('Moving up to end point')
        wrist_yaw = request.wrist_yaw
        end_pose = PoseStamped()
        end_pose.header = request.start_point.header

        # Move straight up to point above the current EE pose
        if request.left_arm:
            cur_pose = self.l_arm_pose
        else:
            cur_pose = self.r_arm_pose

        end_pose.pose.position.x = cur_pose.pose.position.x
        end_pose.pose.position.y = cur_pose.pose.position.y
        end_pose.pose.position.z = self.high_arm_init_z
        q = tf.transformations.quaternion_from_euler(0.0, 0.5*pi, wrist_yaw)
        end_pose.pose.orientation.x = q[0]
        end_pose.pose.orientation.y = q[1]
        end_pose.pose.orientation.z = q[2]
        end_pose.pose.orientation.w = q[3]
        self.move_to_cart_pose(end_pose, which_arm,
                               self.post_move_count_thresh)
        rospy.loginfo('Done moving up to end point')

        if request.open_gripper:
            res = robot_gripper.close(block=True, effort=self.max_close_effort)
        self.reset_arm_pose(True, which_arm, request.high_arm_init)
        return response

    def gripper_feedback_post_push(self, request):
        response = FeedbackPushResponse()
        start_point = request.start_point.point
        wrist_yaw = request.wrist_yaw
        is_pull = request.behavior_primitive == GRIPPER_PULL

        if request.left_arm:
            ready_joints = LEFT_ARM_READY_JOINTS
            if request.high_arm_init:
                ready_joints = LEFT_ARM_PULL_READY_JOINTS
            which_arm = 'l'
            robot_gripper = self.robot.left_gripper
        else:
            ready_joints = RIGHT_ARM_READY_JOINTS
            if request.high_arm_init:
                ready_joints = RIGHT_ARM_PULL_READY_JOINTS
            which_arm = 'r'
            robot_gripper = self.robot.right_gripper

        if is_pull:
            self.stop_moving_vel(which_arm)
            rospy.loginfo('Opening gripper')
            res = robot_gripper.open(position=0.9,block=True)
            rospy.loginfo('Done opening gripper')
            rospy.logdebug('Pulling reverse')
            pose_err, err_dist = self.move_relative_gripper(
                np.matrix([-self.gripper_pull_reverse_dist, 0.0, 0.0]).T, which_arm,
                move_cart_count_thresh=self.post_pull_count_thresh)
            rospy.loginfo('Done pulling reverse')
        else:
            rospy.logdebug('Moving gripper up')
            pose_err, err_dist = self.move_relative_gripper(
                np.matrix([0.0, 0.0, -self.gripper_raise_dist]).T, which_arm,
                move_cart_count_thresh=self.post_move_count_thresh)
            rospy.logdebug('Done moving up')
            rospy.logdebug('Pushing reverse')
            pose_err, err_dist = self.move_relative_gripper(
                np.matrix([-self.gripper_push_reverse_dist, 0.0, 0.0]).T, which_arm,
                move_cart_count_thresh=self.post_move_count_thresh)
            rospy.loginfo('Done pushing reverse')

        rospy.logdebug('Moving up to end point')
        wrist_yaw = request.wrist_yaw
        end_pose = PoseStamped()
        end_pose.header = request.start_point.header

        # Move straight up to point above the current EE pose
        if request.left_arm:
            cur_pose = self.l_arm_pose
        else:
            cur_pose = self.r_arm_pose

        end_pose.pose.position.x = cur_pose.pose.position.x
        end_pose.pose.position.y = cur_pose.pose.position.y
        end_pose.pose.position.z = self.high_arm_init_z
        q = tf.transformations.quaternion_from_euler(0.0, 0.0, wrist_yaw)
        end_pose.pose.orientation.x = q[0]
        end_pose.pose.orientation.y = q[1]
        end_pose.pose.orientation.z = q[2]
        end_pose.pose.orientation.w = q[3]
        self.move_to_cart_pose(end_pose, which_arm,
                               self.post_move_count_thresh)
        rospy.loginfo('Done moving up to end point')

        if request.open_gripper or is_pull:
            rospy.loginfo('Closing gripper')
            res = robot_gripper.close(block=True, effort=self.max_close_effort)
            rospy.loginfo('Done closing gripper')

        self.reset_arm_pose(True, which_arm, request.high_arm_init)
        return response

    def gripper_feedback_post_sweep(self, request):
        response = FeedbackPushResponse()
        start_point = request.start_point.point
        wrist_yaw = request.wrist_yaw

        if request.left_arm:
            ready_joints = LEFT_ARM_READY_JOINTS
            if request.high_arm_init:
                ready_joints = LEFT_ARM_PULL_READY_JOINTS
            which_arm = 'l'
            robot_gripper = self.robot.left_gripper
        else:
            ready_joints = RIGHT_ARM_READY_JOINTS
            if request.high_arm_init:
                ready_joints = RIGHT_ARM_PULL_READY_JOINTS
            which_arm = 'r'
            robot_gripper = self.robot.right_gripper

        # TODO: Make this better
        rospy.logdebug('Moving gripper up')
        pose_err, err_dist = self.move_relative_gripper(
            np.matrix([0.0, -self.gripper_raise_dist, 0.0]).T, which_arm,
            move_cart_count_thresh=self.post_move_count_thresh)
        rospy.logdebug('Done moving up')
        rospy.logdebug('Sweeping reverse')
        pose_err, err_dist = self.move_relative_gripper(
            np.matrix([0.0, 0.0, -self.gripper_push_reverse_dist]).T, which_arm,
            move_cart_count_thresh=self.post_move_count_thresh)
        rospy.loginfo('Done sweeping reverse')

        rospy.logdebug('Moving up to end point')
        wrist_yaw = request.wrist_yaw
        end_pose = PoseStamped()
        end_pose.header = request.start_point.header

        # Move straight up to point above the current EE pose
        if request.left_arm:
            cur_pose = self.l_arm_pose
        else:
            cur_pose = self.r_arm_pose

        end_pose.pose.position.x = cur_pose.pose.position.x
        end_pose.pose.position.y = cur_pose.pose.position.y
        end_pose.pose.position.z = self.high_arm_init_z
        q = tf.transformations.quaternion_from_euler(0.5*pi, 0.0, wrist_yaw)
        end_pose.pose.orientation.x = q[0]
        end_pose.pose.orientation.y = q[1]
        end_pose.pose.orientation.z = q[2]
        end_pose.pose.orientation.w = q[3]
        self.move_to_cart_pose(end_pose, which_arm,
                               self.post_move_count_thresh)
        rospy.loginfo('Done moving up to end point')

        if request.open_gripper:
            rospy.loginfo('Closing gripper')
            res = robot_gripper.close(block=True, effort=self.max_close_effort)
            rospy.loginfo('Done closing gripper')

        self.reset_arm_pose(True, which_arm, request.high_arm_init)
        return response

    #
    # Pre behavior functions
    #
    def gripper_pre_push(self, request):
        response = FeedbackPushResponse()
        start_point = request.start_point.point
        wrist_yaw = request.wrist_yaw
        is_pull = request.behavior_primitive == GRIPPER_PULL

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

        self.set_arm_joint_pose(ready_joints, which_arm, nsecs=1.5)
        rospy.logdebug('Moving %s_arm to ready pose' % which_arm)

        start_pose = PoseStamped()
        start_pose.header = request.start_point.header
        start_pose.pose.position.x = start_point.x
        start_pose.pose.position.y = start_point.y
        start_pose.pose.position.z = start_point.z

        wrist_pitch = 0.0625*pi
        q = tf.transformations.quaternion_from_euler(0.0, wrist_pitch, wrist_yaw)
        start_pose.pose.orientation.x = q[0]
        start_pose.pose.orientation.y = q[1]
        start_pose.pose.orientation.z = q[2]
        start_pose.pose.orientation.w = q[3]

        # TODO: Make gripper open dist a parameter
        if request.open_gripper:
            res = robot_gripper.open(block=True, position=0.05)
        if is_pull:
            res = robot_gripper.open(block=True, position=0.9)

        self.use_gripper_place_joint_posture = True
        if request.high_arm_init:
            start_pose.pose.position.z = self.high_arm_init_z
            self.move_to_cart_pose(start_pose, which_arm)
            rospy.logdebug('Done moving to overhead start point')
            # Change z to lower arm to table
            start_pose.pose.position.z = self.lower_arm_init_z
            self.move_to_cart_pose(start_pose, which_arm)
            rospy.loginfo('Done moving to lower start point')
            start_pose.pose.position.z = start_point.z
            # self.move_down_until_contact(which_arm)

        # Move to start pose
        if not self.move_to_cart_pose_ik(start_pose, which_arm):
            rospy.logwarn('IK Failed, not at desired initial pose')
            response.failed_pre_position = True
            # self.move_to_cart_pose(start_pose, which_arm, self.pre_push_count_thresh)
            return response
        else:
            response.failed_pre_position = False

        rospy.loginfo('Done moving to start point')
        self.use_gripper_place_joint_posture = False
        if is_pull:
            rospy.loginfo('Moving forward to grasp pose')
            pose_err, err_dist = self.move_relative_gripper(
                np.matrix([self.gripper_pull_forward_dist, 0, 0]).T, which_arm,
                move_cart_count_thresh=self.post_move_count_thresh)

            rospy.loginfo('Closing grasp for pull')
            res = robot_gripper.close(block=True, effort=self.max_close_effort)
            rospy.loginfo('Done closing gripper')

        return response

    def gripper_pre_sweep(self, request):
        response = FeedbackPushResponse()
        start_point = request.start_point.point
        wrist_yaw = request.wrist_yaw

        if request.left_arm:
            ready_joints = LEFT_ARM_READY_JOINTS
            if request.high_arm_init:
                ready_joints = LEFT_ARM_HIGH_SWEEP_READY_JOINTS
            which_arm = 'l'
            wrist_roll = -pi
            robot_gripper = self.robot.left_gripper
        else:
            ready_joints = RIGHT_ARM_READY_JOINTS
            if request.high_arm_init:
                ready_joints = RIGHT_ARM_HIGH_SWEEP_READY_JOINTS
            which_arm = 'r'
            wrist_roll = 0.0
            robot_gripper = self.robot.right_gripper

        if request.open_gripper:
            res = robot_gripper.open(block=True, position=0.9)
            raw_input('Waiting for input to close gripper: ')
            print '\n'
            res = robot_gripper.close(block=True, effort=self.max_close_effort)

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
            start_pose.pose.position.z = self.lower_arm_init_z
            self.move_to_cart_pose(start_pose, which_arm)
            rospy.loginfo('Done moving to lower start point')

            # Lower arm to table
            start_pose.pose.position.z = start_point.z
            # self.move_down_until_contact(which_arm)

        self.move_to_cart_pose(start_pose, which_arm, self.pre_push_count_thresh)
        rospy.loginfo('Done moving to start point')

        return response

    def overhead_pre_push(self, request):
        response = FeedbackPushResponse()
        start_point = request.start_point.point
        wrist_yaw = request.wrist_yaw

        if request.left_arm:
            ready_joints = LEFT_ARM_READY_JOINTS
            if request.high_arm_init:
                ready_joints = LEFT_ARM_PULL_READY_JOINTS
            which_arm = 'l'
            wrist_pitch = 0.5*pi
            robot_gripper = self.robot.left_gripper
        else:
            ready_joints = RIGHT_ARM_READY_JOINTS
            if request.high_arm_init:
                ready_joints = RIGHT_ARM_PULL_READY_JOINTS
            which_arm = 'r'
            wrist_pitch = -0.5*pi
            robot_gripper = self.robot.right_gripper
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


        rospy.logdebug('Moving %s_arm to ready pose' % which_arm)
        self.set_arm_joint_pose(ready_joints, which_arm, nsecs=1.5)

        if not request.high_arm_init:
            # Rotate wrist before moving to position
            rospy.logdebug('Rotating elbow for overhead push')
            arm_pose = self.get_arm_joint_pose(which_arm)
            arm_pose[-3] =  wrist_pitch
            self.set_arm_joint_pose(arm_pose, which_arm, nsecs=1.0)

        if request.open_gripper:
            res = robot_gripper.open(block=True, position=0.05)

        if request.high_arm_init:
            # Move to offset pose above the table
            start_pose.pose.position.z = self.high_arm_init_z
            self.move_to_cart_pose(start_pose, which_arm, self.pre_push_count_thresh)
            rospy.logdebug('Done moving to overhead start point')
            start_pose.pose.position.z = self.lower_arm_init_z
            self.move_to_cart_pose(start_pose, which_arm, self.pre_push_count_thresh)
            rospy.loginfo('Done moving to lower start point')

            # Lower arm to table
            start_pose.pose.position.z = start_point.z
            # self.move_down_until_contact(which_arm)

        # Move to start pose
        if not self.move_to_cart_pose_ik(start_pose, which_arm):
            rospy.logwarn('IK Failed, not at desired initial pose')
            response.failed_pre_position = True
            # self.move_to_cart_pose(start_pose, which_arm, self.pre_push_count_thresh)
        else:
            response.failed_pre_position = False
            rospy.loginfo('Done moving to start point')

        return response

    #
    # Fixed length pushing behaviors
    #
    def gripper_push(self, request):
        response = FeedbackPushResponse()
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

    def gripper_post_push(self, request):
        response = FeedbackPushResponse()
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
            res = robot_gripper.close(block=True, position=0.9, effort=self.max_close_effort)

        if request.high_arm_init:
            start_pose.pose.position.z = self.high_arm_init_z
            rospy.logdebug('Moving up to initial point')
            self.move_to_cart_pose(start_pose, which_arm,
                                   self.post_move_count_thresh)
            rospy.loginfo('Done moving up to initial point')

        self.reset_arm_pose(True, which_arm, request.high_arm_init)
        return response

    def gripper_sweep(self, request):
        response = FeedbackPushResponse()
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

    def gripper_post_sweep(self, request):
        response = FeedbackPushResponse()
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

        self.reset_arm_pose(True, which_arm, request.high_arm_init)
        return response

    def overhead_push(self, request):
        response = FeedbackPushResponse()
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

    def overhead_post_push(self, request):
        response = FeedbackPushResponse()
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

        self.reset_arm_pose(True, which_arm, request.high_arm_init)
        return response

    #
    # Head and spine setup functions
    #
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
        (trans, rot) = self.tf_listener.lookupTransform('base_link', 'torso_lift_link', rospy.Time(0))
        lift_link_z = trans[2]

        # tabletop position in base_link frame
        request.table_centroid.header.stamp = rospy.Time(0)
        table_base = self.tf_listener.transformPose('base_link', request.table_centroid)
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
        rospy.loginfo('New spine height is ' + str(new_torso_position))

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
        # rospy.loginfo('Move cart gripper error dist: ' + str(error_dist)+'\n')
        # rospy.loginfo('Move cart gripper error: ' + str(arm_error.linear)+'\n'+str(arm_error.angular))
        return (arm_error, error_dist)

    def move_to_cart_pose_ik(self, pose, which_arm, pressure=1000, nsecs=2.0):
        if which_arm == 'l':
            ik_proxy = self.l_arm_ik_proxy
            solver_proxy = self.l_arm_ik_solver_proxy
        else:
            ik_proxy = self.r_arm_ik_proxy
            solver_proxy = self.r_arm_ik_solver_proxy
        wrist_name = which_arm + '_wrist_roll_link'
        tool_name  = which_arm + '_gripper_tool_frame'

        # rospy.loginfo('Getting solver info')
        solver_req = GetKinematicSolverInfoRequest()
        solver_res = solver_proxy(solver_req)

        # Convert tool tip pose to wrist pose
        rospy.loginfo('Requested pose for tool link is: ' + str(pose.pose))
        R_wrist_to_tool = np.asmatrix(tf.transformations.quaternion_matrix(np.asarray([pose.pose.orientation.x,
                                                                                       pose.pose.orientation.y,
                                                                                       pose.pose.orientation.z,
                                                                                       pose.pose.orientation.w])))
        # NOTE: Gripper is 18cm long
        tool_to_wrist_vec = R_wrist_to_tool * np.asmatrix([[-0.18, 0.0, 0.0, 1.0]]).T
        pose.pose.position.x += tool_to_wrist_vec[0]
        pose.pose.position.y += tool_to_wrist_vec[1]
        pose.pose.position.z += tool_to_wrist_vec[2]
        rospy.loginfo('Requested pose for wrist link is: ' + str(pose.pose))

        # Get IK of desired wrist pose
        ik_req = GetPositionIKRequest()
        ik_req.timeout = rospy.Duration(5) # 5 Secs
        ik_req.ik_request.pose_stamped = pose
        ik_req.ik_request.ik_link_name = wrist_name
        # Seed the solution with the current arm state
        arm_pose = self.get_arm_joint_pose(which_arm)
        # rospy.loginfo('Current arm pose is: ' + str(arm_pose))
        # current_state = self.arm_pose_to_robot_state(arm_pose, which_arm)
        # rospy.loginfo('Converting current arm pose to robot state')
        ik_req.ik_request.ik_seed_state.joint_state.position = []
        ik_req.ik_request.ik_seed_state.joint_state.name = []
        for i in xrange(len(solver_res.kinematic_solver_info.joint_names)):
            ik_req.ik_request.ik_seed_state.joint_state.position.append(arm_pose[i,0])
            ik_req.ik_request.ik_seed_state.joint_state.name.append(solver_res.kinematic_solver_info.joint_names[i])
            # rospy.loginfo('State ' + str(ik_req.ik_request.ik_seed_state.joint_state.name[i]) + ' has state ' +
            #               str(ik_req.ik_request.ik_seed_state.joint_state.position[i]))
        # rospy.loginfo('Requesting ik solution')
        ik_res = ik_proxy(ik_req)
        # Check that we got a solution
        if ik_res.error_code.val != ik_res.error_code.SUCCESS:
            try:
                rospy.logwarn('IK failed with error code: ' + _ARM_ERROR_CODES[ik_res.error_code.val])
            except KeyError:
                rospy.logwarn('IK failed with unknown error code: ' + str(ik_res.error_code.val))
            return False
        # Move to IK joint pose
        # rospy.loginfo('Converting IK result')
        joint_pose = self.ik_robot_state_to_arm_pose(ik_res.solution)
        # rospy.loginfo('Desired joint pose from IK is: ' + str(joint_pose))
        self.set_arm_joint_pose(joint_pose, which_arm, nsecs=nsecs)
        return True

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
        # rospy.loginfo('Move cart gripper error dist: ' + str(error_dist)+'\n')
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
        rospy.loginfo('Stopping to move velocity for ' + which_arm + '_arm')
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
            place_posture = 'gripper_place_l'
        else:
            posture = 'elbowupr'
            place_posture = 'gripper_place_r'

        joints = self.get_arm_joint_pose(which_arm)
        joints = joints.tolist()
        joints = [j[0] for j in joints]
        if self.use_gripper_place_joint_posture:
            m = Float64MultiArray(data=_POSTURES[place_posture])
        elif self.use_cur_joint_posture:
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

    def l_arm_vel_state_callback(self, state_msg):
        x_err = state_msg.x_err
        x_d = state_msg.xd
        self.l_arm_pose = state_msg.x
        self.l_arm_x_err = x_err
        self.l_arm_x_d = x_d
        self.l_arm_F = state_msg.F

    def r_arm_vel_state_callback(self, state_msg):
        x_err = state_msg.x_err
        x_d = state_msg.xd
        self.r_arm_pose = state_msg.x
        self.r_arm_x_err = state_msg.x_err
        self.r_arm_x_d = state_msg.xd
        self.r_arm_F = state_msg.F

    def ik_robot_state_to_arm_pose(self, robot_state):
        # HACK: Assumes the state is from the ik solver
        joint_pose = np.asmatrix(np.zeros((len(robot_state.joint_state.position), 1)))
        for i in xrange(len(robot_state.joint_state.position)):
            joint_pose[i,0] = robot_state.joint_state.position[i]

        return joint_pose
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

    def setupRBFController(self, controller_name):
        controller_file_path = self.learned_controller_base_path+controller_name+'.txt'
        self.RBF = rbf_control.RBFController()
        self.RBF.loadRBFController(controller_file_path)

    def loadAffineController(self, controller_name):
        controller_file = file(self.learned_controller_base_path+controller_name+'.txt','r')
        A = None
        B = None
        controller_file.close()
        return (A, B)

    def shutdown_hook(self):
        rospy.loginfo('Cleaning up node on shutdown')
        if _USE_LEARN_IO:
            if self.learn_io is not None:
                self.learn_io.close_out_file()
        if _SAVE_MPC_DATA:
            if self.mpc_q_star_io is not None:
                self.mpc_q_star_io.close_out_file()
            if self.target_trajectory_io is not None:
                self.target_trajectory_io.close_out_file()
        # TODO: stop moving the arms on shutdown

    #
    # Main Control Loop
    #
    def run(self):
        '''
        Main control loop for the node
        '''
        # getting_joints = True
        # while getting_joints:
        #     code_in = raw_input('Press <Enter> to get current arm joints: ')
        #     print 'Left arm: ' + str(self.robot.left.pose())
        #     print 'Right arm: ' + str(self.robot.right.pose())
        #     if code_in.startswith('q'):
        #         getting_joints = False
        if not _OFFLINE:
            self.init_spine_pose()
            self.init_head_pose(self.head_pose_cam_frame)
            self.init_arms()
            self.switch_to_cart_controllers()
        rospy.loginfo('Done initializing feedback pushing node.')
        rospy.on_shutdown(self.shutdown_hook)
        rospy.spin()

if __name__ == '__main__':
    node = PositionFeedbackPushNode()
    node.run()
