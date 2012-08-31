#!/usr/bin/env python
# Software License Agreement (BSD License)
#
#  Copyright (c) 2012, Georgia Institute of Technology
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
import hrl_pr2_lib.linear_move as lm
import hrl_pr2_lib.pr2 as pr2
import hrl_lib.tf_utils as tfu
import tf
import numpy as np
from tabletop_pushing.srv import *
from tabletop_pushing.msg import *
from math import sin, cos, pi, fabs, sqrt
import sys
from push_learning import PushLearningIO
from geometry_msgs.msg import Pose2D
import time
import random

CENTROID_CONTROLLER ='centroid_controller'
SPIN_COMPENSATION = 'spin_compensation'
SPIN_TO_HEADING = 'spin_to_heading'
CONTROLLERS = [CENTROID_CONTROLLER, SPIN_COMPENSATION]

GRIPPER_PUSH = 'gripper_push'
GRIPPER_SWEEP = 'gripper_sweep'
OVERHEAD_PUSH = 'overhead_push'
PUSH_PRIMITIVES = [OVERHEAD_PUSH, GRIPPER_PUSH, GRIPPER_SWEEP]
ACTION_PRIMITIVES = {CENTROID_CONTROLLER:PUSH_PRIMITIVES, SPIN_COMPENSATION:PUSH_PRIMITIVES}

ELLIPSE_PROXY = 'ellipse'
CENTROID_PROXY = 'centroid'
PERCEPTUAL_PROXIES = {CENTROID_CONTROLLER:[CENTROID_PROXY], SPIN_COMPENSATION:[ELLIPSE_PROXY]}

_OFFLINE = False
_USE_LEARN_IO = False
_TEST_START_POSE = False
_WAIT_BEFORE_STRAIGHT_PUSH = False
_SPIN_FIRST = False
_USE_CENTROID_CONTROLLER = True
_USE_FIXED_GOAL = False

class TabletopExecutive:

    def __init__(self, use_singulation, use_learning):
        rospy.init_node('tabletop_executive_node',log_level=rospy.DEBUG)
        self.min_push_dist = rospy.get_param('~min_push_dist', 0.07)
        self.max_push_dist = rospy.get_param('~mix_push_dist', 0.3)
        self.use_overhead_x_thresh = rospy.get_param('~use_overhead_x_thresh', 0.55)
        self.use_sweep_angle_thresh = rospy.get_param('~use_sweep_angle_thresh', pi*0.4)
        self.use_pull_angle_thresh = rospy.get_param('~use_sweep_angle_thresh', pi*0.525)
        self.use_same_side_y_thresh = rospy.get_param('~use_same_side_y_thresh', 0.3)
        self.use_same_side_x_thresh = rospy.get_param('~use_same_side_x_thresh', 0.8)

        self.gripper_offset_dist = rospy.get_param('~gripper_push_offset_dist', 0.05)
        self.gripper_start_z = rospy.get_param('~gripper_push_start_z', -0.25)

        self.sweep_offset_dist = rospy.get_param('~gripper_sweep_offset_dist', 0.04)
        self.sweep_start_z = rospy.get_param('~gripper_sweep_start_z', -0.27)

        self.overhead_offset_dist = rospy.get_param('~overhead_push_offset_dist', 0.05)
        self.overhead_start_z = rospy.get_param('~overhead_push_start_z', -0.275)
        self.pull_start_z = rospy.get_param('~overhead_push_start_z', -0.27)

        self.max_restart_limit = rospy.get_param('~max_restart_limit', 3)

        # Setup service proxies
        if not _OFFLINE:
            # New visual feedback proxies
            self.overhead_feedback_push_proxy = rospy.ServiceProxy(
                'overhead_feedback_push', FeedbackPush)
            self.overhead_feedback_post_push_proxy = rospy.ServiceProxy(
                'overhead_feedback_post_push', FeedbackPush)
            self.gripper_feedback_push_proxy = rospy.ServiceProxy(
                'gripper_feedback_push', FeedbackPush)
            self.gripper_feedback_post_push_proxy = rospy.ServiceProxy(
                'gripper_feedback_post_push', FeedbackPush)
            self.gripper_feedback_sweep_proxy = rospy.ServiceProxy(
                'gripper_feedback_sweep', FeedbackPush)
            self.gripper_feedback_post_sweep_proxy = rospy.ServiceProxy(
                'gripper_feedback_post_sweep', FeedbackPush)
            self.overhead_feedback_pre_push_proxy = rospy.ServiceProxy('overhead_pre_push',
                                                                       FeedbackPush)
            self.gripper_feedback_pre_push_proxy = rospy.ServiceProxy('gripper_pre_push',
                                                                      FeedbackPush)
            self.gripper_feedback_pre_sweep_proxy = rospy.ServiceProxy('gripper_pre_sweep',
                                                                       FeedbackPush)
            # Proxy to setup spine and head
            self.raise_and_look_proxy = rospy.ServiceProxy('raise_and_look',
                                                           RaiseAndLook)
        self.table_proxy = rospy.ServiceProxy('get_table_location', LocateTable)

        # TODO: Make this a string passed in somewhere

        if use_singulation:
            self.init_singulation()
        if use_learning:
            self.init_learning()

    def init_singulation(self):
        # Singulation Push proxy
        self.singulation_push_vector_proxy = rospy.ServiceProxy(
            'get_singulation_push_vector', SingulationPush)

    def init_learning(self):
        # Singulation Push proxy
        if _USE_LEARN_IO:
            self.learn_io = PushLearningIO()
            learn_file_name = '/u/thermans/data/new/goal_out.txt'
            rospy.loginfo('Opening learn file: '+learn_file_name)
            self.learn_io.open_out_file(learn_file_name)
        self.learning_push_vector_proxy = rospy.ServiceProxy(
            'get_learning_push_vector', LearnPush)
        # Get table height and raise to that before anything else
        if not _OFFLINE:
            self.raise_and_look()
        # Initialize push pose
        initialized = False
        r = rospy.Rate(0.5)
        while not initialized:
            initialized = self.initialize_learning_push()
            r.sleep()
        rospy.loginfo('Done initializing learning')

    def finish_learning(self):
        rospy.loginfo('Done with learning pushes and such.')
        if _USE_LEARN_IO:
            self.learn_io.close_out_file()

    def run_singulation(self, num_pushes=1, use_guided=True):
        # Get table height and raise to that before anything else
        self.raise_and_look()
        # Initialize push pose
        self.initialize_singulation_push_vector();

        # NOTE: Should exit before reaching num_pushes, this is just a backup
        for i in xrange(num_pushes):
            pose_res = self.request_singulation_push(use_guided)
            # raw_input('Hit any key to continue')
            # continue
            if pose_res is None:
                rospy.logwarn("pose_res is None. Exiting pushing");
                break
            if pose_res.no_push:
                rospy.loginfo("No push. Exiting pushing.");
                break
            rospy.loginfo('Performing push #' + str(i+1))
            # Decide push based on the orientation returned
            rospy.loginfo('Push start_point: (' + str(pose_res.start_point.x) +
                          ', ' + str(pose_res.start_point.y) +
                          ', ' + str(pose_res.start_point.z) + ')')
            rospy.loginfo('Push angle: ' + str(pose_res.push_angle))
            rospy.loginfo('Push dist: ' + str(pose_res.push_dist))

            # TODO: Make this a function
            # Choose push behavior
            if fabs(pose_res.push_angle) > self.use_pull_angle_thresh:
                action_primitive = OVERHEAD_PUSH
            elif pose_res.start_point.x < self.use_overhead_x_thresh:
                action_primitive = OVERHEAD_PUSH
            elif fabs(pose_res.push_angle) > self.use_sweep_angle_thresh:
                action_primitive = GRIPPER_SWEEP
            else:
                action_primitive = GRIPPER_PUSH

            # action_primitive = GRIPPER_PUSH
            # action_primitive = OVERHEAD_PUSH
            # action_primitive = GRIPPER_SWEEP
            # TODO: Make this a function
            # Choose arm
            if (fabs(pose_res.start_point.y) > self.use_same_side_y_thresh or
                pose_res.start_point.x > self.use_same_side_x_thresh):
                if (pose_res.start_point.y < 0):
                    which_arm = 'r'
                    rospy.loginfo('Setting arm to right because of limits')
                else:
                    which_arm = 'l'
                    rospy.loginfo('Setting arm to left because of limits')
            elif pose_res.push_angle > 0:
                which_arm = 'r'
                rospy.loginfo('Setting arm to right because of angle')
            else:
                which_arm = 'l'
                rospy.loginfo('Setting arm to left because of angle')

            push_dist = pose_res.push_dist
            push_dist = max(min(push_dist, self.max_push_dist),
                            self.min_push_dist)
            if action_primitive == GRIPPER_PUSH:
                self.gripper_push_object(push_dist, which_arm, pose_res, True)
            if action_primitive == GRIPPER_SWEEP:
                self.sweep_object(push_dist, which_arm, pose_res, True)
            if action_primitive == OVERHEAD_PUSH:
                self.overhead_push_object(push_dist, which_arm, pose_res, True)
            rospy.loginfo('Done performing push behavior.\n')

        if not (pose_res is None):
            rospy.loginfo('Singulated objects: ' + str(pose_res.singulated))
            rospy.loginfo('Final estimate of ' + str(pose_res.num_objects) +
                          ' objects')

    def run_feedback_testing(self, action_primitive):
        high_init = True
        use_spin_push = _SPIN_FIRST
        continuing = False
        while True:
            start_time = time.time()
            # Select controller to use
            if use_spin_push:
                controller_name = 'spin_to_heading'
            elif _USE_CENTROID_CONTROLLER:
                controller_name = 'centroid_controller'
            else:
                controller_name = 'spin_compensation'

            if use_spin_push:
                goal_pose = self.generate_random_table_pose()
                code_in = raw_input('Set object in start pose and press <Enter>: ')
                if code_in.startswith('q'):
                    return
            elif _WAIT_BEFORE_STRAIGHT_PUSH or not _SPIN_FIRST:
                if not _SPIN_FIRST:
                    # goal_pose = self.generate_random_table_pose()
                    goal_pose = Pose2D()
                    goal_pose.x = 0.7
                    goal_pose.y = 0.0
                    goal_pose.theta = 0.0
                    if continuing:
                        code_in = ''
                        continuing = False
                    else:
                        code_in = raw_input('Set object in start pose and press <Enter>: ')
                else:
                    code_in = raw_input('Spun object to orientation going to push now <Enter>: ')
                if code_in.startswith('q'):
                    return

            push_vec_res = self.get_feedback_push_start_pose(goal_pose, controller_name)
            code_in = raw_input('Got start pose to continue press <Enter>: ')
            if code_in.startswith('q'):
                return

            if push_vec_res is None:
                return
            which_arm = self.choose_arm(push_vec_res.push, controller_name)
            res, push_res = self.perform_push(which_arm, action_primitive, push_vec_res, goal_pose,
                                      controller_name, '', high_init)
            if res == 'aborted':
                rospy.loginfo('Continuing after abortion')
                continuing = True
                continue

            # NOTE: Alternate between spinning and pushing
            if not _TEST_START_POSE and _SPIN_FIRST:
                use_spin_push = (not use_spin_push)
            push_time = time.time() - start_time
            self.analyze_push(action_primitive, controller_name, proxy_name, which_arm, push_time,
                              push_vector_res, goal_pose)

            if not res or res == 'quit':
                return

    def run_push_exploration(self, object_id='test_object', ask_for_input=True):
        if ask_for_input:
            code_in = raw_input('Set object in start pose and press <Enter>: ')
            if code_in.startswith('q'):
                return

        for controller in CONTROLLERS:
            for action_primitive in ACTION_PRIMITIVES[controller]:
                for proxy in PERCEPTUAL_PROXIES[controller]:
                    res = self.explore_push(action_primitive, controller, proxy)
                    if res == 'quit':
                        rospy.loginfo('Quiting on user request')
                        return

    def explore_push(self, action_primitive, controller_name, proxy_name):
        rospy.loginfo('Exploring push triple: (' + action_primitive + ', '
                      + controller_name + ', ' + proxy_name + ')')
        continuing = False
        done_with_push = False
        goal_pose = Pose2D()
        if _USE_FIXED_GOAL:
            goal_pose.x = 0.7
            goal_pose.y = 0.0
            goal_pose.theta = 0.0
        else:
            goal_pose = self.generate_random_table_pose()

        restart_count = 0
        start_time = time.time()
        while not done_with_push:
            if continuing:
                continuing = False
            push_vec_res = self.get_feedback_push_start_pose(goal_pose, controller_name, proxy_name, action_primitive)

            if push_vec_res is None:
                return
            which_arm = self.choose_arm(push_vec_res.push, controller_name)
            res, push_res = self.perform_push(which_arm, action_primitive,
                                              push_vec_res, goal_pose,
                                              controller_name, proxy_name)
            if res == 'quit':
                return res
            elif res == 'aborted':
                rospy.loginfo('Continuing after push was aborted')
                continuing = True
                restart_count += 1
                if restart_count <= self.max_restart_limit:
                    continue
                else:
                    done_with_push = True
            else:
                rospy.loginfo('Stopping push attempt because of too many restarts')
                done_with_push = True
        push_time = time.time() - start_time
        # TODO: Figure out what needs to be sent in here,
        # make sure we have it all
        if _OFFLINE:
            code_in = raw_input('Press <Enter> to get analysis vector: ')
            if code_in.startswith('q'):
                return 'quit'
        self.analyze_push(action_primitive, controller_name, proxy_name, which_arm, push_time,
                          push_vec_res, goal_pose)
        return res

    def get_feedback_push_start_pose(self, goal_pose, controller_name, proxy_name, action_primitive):
        get_push = True
        while get_push:
            push_vec_res = self.request_feedback_push_start_pose(goal_pose, controller_name,
                                                                 proxy_name, action_primitive)

            if push_vec_res is None:
                return None
            if push_vec_res.no_objects:
                code_in = raw_input('No objects found. Place object and press <Enter>: ')
                if code_in.startswith('q'):
                    return None
            else:
                return push_vec_res

    def choose_arm(self, push_vec, controller_name):
        if controller_name == 'spin_to_heading':
            if (push_vec.start_point.y < 0):
                which_arm = 'r'
                rospy.loginfo('Setting arm to right because of spinning')
            else:
                which_arm = 'l'
                rospy.loginfo('Setting arm to left because of spinning')
            return which_arm

        if (fabs(push_vec.start_point.y) > self.use_same_side_y_thresh or
            push_vec.start_point.x > self.use_same_side_x_thresh):
            if (push_vec.start_point.y < 0):
                which_arm = 'r'
                rospy.loginfo('Setting arm to right because of limits')
            else:
                which_arm = 'l'
                rospy.loginfo('Setting arm to left because of limits')
        elif push_vec.push_angle > 0:
            which_arm = 'r'
            rospy.loginfo('Setting arm to right because of angle')
        else:
            which_arm = 'l'
            rospy.loginfo('Setting arm to left because of angle')

        return which_arm

    def perform_push(self, which_arm, action_primitive, push_vector_res, goal_pose,
                     controller_name, proxy_name, high_init = True):
        push_angle = push_vector_res.push.push_angle
        # NOTE: Use commanded push distance not visually decided minimal distance
        if push_vector_res is None:
            rospy.logwarn("push_vector_res is None. Exiting pushing");
            return (False, None)
        if push_vector_res.no_push:
            rospy.loginfo("No push. Exiting pushing.");
            return (False, None)
        # Decide push based on the orientation returned
        rospy.loginfo('Push start_point: (' +
                      str(push_vector_res.push.start_point.x) + ', ' +
                      str(push_vector_res.push.start_point.y) + ', ' +
                      str(push_vector_res.push.start_point.z) + ')')
        rospy.loginfo('Push angle: ' + str(push_angle))
        start_time = time.time()
        # TODO: Unify framework here, to call with action_primitive to a single feedback behavior
        if not _OFFLINE:
            if action_primitive == OVERHEAD_PUSH:
                result = self.overhead_feedback_push_object(which_arm,
                                                            push_vector_res.push, goal_pose,
                                                            controller_name, proxy_name, action_primitive)
            if action_primitive == GRIPPER_SWEEP:
                result = self.feedback_sweep_object(which_arm, push_vector_res.push,
                                                    goal_pose, controller_name, proxy_name, action_primitive)
            if action_primitive == GRIPPER_PUSH:
                result = self.gripper_feedback_push_object(which_arm,
                                                           push_vector_res.push, goal_pose,
                                                           controller_name, proxy_name, action_primitive)
        else:
            result = FeedbackPushResponse()

        # TODO: Make this more robust to other use cases
        # If the call aborted, recall with the same settings
        if result.action_aborted:
            rospy.logwarn('Push was aborted. Calling push behavior again.')
            return ('aborted', result)

        rospy.loginfo('Done performing push behavior.')
        return ('done', result)

    def analyze_push(self, action_primitive, controller_name, proxy_name,
                     which_arm, push_time, push_vector_res, goal_pose):
        push_angle = push_vector_res.push.push_angle
        analysis_res = self.request_learning_analysis()
        rospy.loginfo('Done getting analysis response.')
        rospy.loginfo('Push: ' + str(action_primitive))
        rospy.loginfo('Arm: ' + str(which_arm))
        rospy.loginfo('Push time: ' + str(push_time) + 's')
        rospy.loginfo('Init (X,Y,Theta): (' + str(push_vector_res.centroid.x) +
                      ', ' + str(push_vector_res.centroid.y) + ', ' +
                      str(push_vector_res.theta) +')')
        rospy.loginfo('Final (X,Y,Theta): (' + str(analysis_res.centroid.x) + ', ' +
                       str(analysis_res.centroid.y) + ', ' + str(analysis_res.theta)+ ')')
        rospy.loginfo('Desired (X,Y,Theta): (' + str(goal_pose.x) + ', ' +
                       str(goal_pose.y) + ', ' + str(goal_pose.theta) + ')')
        rospy.loginfo('Error (X,Y,Theta): (' + str(fabs(goal_pose.x-analysis_res.centroid.x)) +
                      ', ' + str(fabs(goal_pose.y-analysis_res.centroid.y)) + ', ' +
                      str(fabs(goal_pose.theta-analysis_res.theta)) + ')\n')
        if _USE_LEARN_IO:
            self.learn_io.write_line(
                push_vector_res.centroid, push_vector_res.theta,
                analysis_res.centroid, analysis_res.theta,
                goal_pose, action_primitive, controller_name, proxy_name,
                which_arm, push_time)

    def request_singulation_push(self, use_guided=True):
        push_vector_req = SingulationPushRequest()
        push_vector_req.use_guided = use_guided
        push_vector_req.initialize = False
        push_vector_req.no_push_calc = False
        rospy.loginfo("Calling singulation push vector service")
        try:
            push_vector_res = self.singulation_push_vector_proxy(push_vector_req)
            return push_vector_res
        except rospy.ServiceException, e:
            rospy.logwarn("Service did not process request: %s"%str(e))
            return None

    def request_feedback_push_start_pose(self, goal_pose, controller_name, proxy_name, action_primitive):
        push_vector_req = LearnPushRequest()
        push_vector_req.initialize = False
        push_vector_req.analyze_previous = False
        push_vector_req.goal_pose = goal_pose
        push_vector_req.controller_name = controller_name
        push_vector_req.proxy_name = proxy_name
        push_vector_req.action_primitive = action_primitive
        rospy.loginfo("Getting feedback push start service")
        try:
            push_vector_res = self.learning_push_vector_proxy(push_vector_req)
            return push_vector_res
        except rospy.ServiceException, e:
            rospy.logwarn("Service did not process request: %s"%str(e))
            return None

    def request_learning_analysis(self):
        push_vector_req = LearnPushRequest()
        push_vector_req.initialize = False
        push_vector_req.analyze_previous = True
        rospy.loginfo("Calling learning push vector service")
        try:
            push_vector_res = self.learning_push_vector_proxy(push_vector_req)
            return push_vector_res
        except rospy.ServiceException, e:
            rospy.logwarn("Service did not process request: %s"%str(e))
            return None

    def initialize_singulation_push_vector(self):
        push_vector_req = SingulationPushRequest()
        push_vector_req.initialize = True
        push_vector_req.use_guided = True
        push_vector_req.no_push_calc = False
        rospy.loginfo('Initializing singulation push vector service.')
        self.singulation_push_vector_proxy(push_vector_req)

    def initialize_learning_push(self):
        push_vector_req = LearnPushRequest()
        push_vector_req.initialize = True
        push_vector_req.analyze_previous = False
        rospy.loginfo('Initializing learning push vector service.')
        try:
            self.learning_push_vector_proxy(push_vector_req)
        except rospy.ServiceException, e:
            rospy.logwarn("Service did not process request: %s"%str(e))
            return False
        return True

    def raise_and_look(self, request_table=True, init_arms=False):
        if request_table:
            table_req = LocateTableRequest()
            table_req.recalculate = True
        raise_req = RaiseAndLookRequest()
        raise_req.point_head_only = True
        raise_req.camera_frame = 'head_mount_kinect_rgb_link'
        # First make sure the head is looking the correct way before estimating
        # the table centroid
        # Also make sure the arms are out of the way
        raise_req.init_arms = True
        rospy.loginfo("Moving head and arms")
        raise_res = self.raise_and_look_proxy(raise_req)
        if request_table:
            raise_req.have_table_centroid = True
            try:
                rospy.loginfo("Getting table pose")
                table_res = self.table_proxy(table_req);
            except rospy.ServiceException, e:
                rospy.logwarn("Service did not process request: %s"%str(e))
                return
            if not table_res.found_table:
                return
            raise_req.table_centroid = table_res.table_centroid
        else:
            raise_req.have_table_centroid = False

        # TODO: Make sure this requested table_centroid is valid

        rospy.loginfo("Raising spine");
        raise_req.point_head_only = False
        raise_req.init_arms = init_arms
        raise_res = self.raise_and_look_proxy(raise_req)

    def overhead_feedback_push_object(self, which_arm, push_vector, goal_pose, controller_name,
                                      proxy_name, action_primitive, high_init=True, open_gripper=False):
        # Convert pose response to correct push request format
        push_req = FeedbackPushRequest()
        push_req.start_point.header = push_vector.header
        push_req.start_point.point = push_vector.start_point
        push_req.open_gripper = open_gripper
        push_req.goal_pose = goal_pose

        # Use the sent wrist yaw
        wrist_yaw = push_vector.push_angle
        push_req.wrist_yaw = wrist_yaw

        # Offset pose to not hit the object immediately
        push_req.start_point.point.x += -self.gripper_offset_dist*cos(wrist_yaw)
        push_req.start_point.point.y += -self.gripper_offset_dist*sin(wrist_yaw)
        push_req.start_point.point.z = self.overhead_start_z
        push_req.left_arm = (which_arm == 'l')
        push_req.right_arm = not push_req.left_arm
        push_req.high_arm_init = high_init
        push_req.controller_name = controller_name
        push_req.proxy_name = proxy_name
        push_req.action_primitive = action_primitive

        rospy.loginfo('Gripper push augmented start_point: (' +
                      str(push_req.start_point.point.x) + ', ' +
                      str(push_req.start_point.point.y) + ', ' +
                      str(push_req.start_point.point.z) + ')')

        rospy.loginfo("Calling overhead feedback pre push service")
        pre_push_res = self.overhead_feedback_pre_push_proxy(push_req)
        rospy.loginfo("Calling overhead feedback push service")

        if _TEST_START_POSE:
            raw_input('waiting for input to recall arm: ')
            push_res = FeedbackPushResponse()
        else:
            push_res = self.overhead_feedback_push_proxy(push_req)
        rospy.loginfo("Calling overhead feedback post push service")
        post_push_res = self.overhead_feedback_post_push_proxy(push_req)
        return push_res

    def gripper_feedback_push_object(self, which_arm, push_vector, goal_pose, controller_name,
                                     proxy_name, action_primitive, high_init=True, open_gripper=False):
        # Convert pose response to correct push request format
        push_req = FeedbackPushRequest()
        push_req.start_point.header = push_vector.header
        push_req.start_point.point = push_vector.start_point
        push_req.open_gripper = open_gripper
        push_req.goal_pose = goal_pose

        # Use the sent wrist yaw
        wrist_yaw = push_vector.push_angle
        push_req.wrist_yaw = wrist_yaw

        # Offset pose to not hit the object immediately
        push_req.start_point.point.x += -self.gripper_offset_dist*cos(wrist_yaw)
        push_req.start_point.point.y += -self.gripper_offset_dist*sin(wrist_yaw)
        push_req.start_point.point.z = self.gripper_start_z
        push_req.left_arm = (which_arm == 'l')
        push_req.right_arm = not push_req.left_arm
        push_req.high_arm_init = high_init
        push_req.controller_name = controller_name
        push_req.proxy_name = proxy_name
        push_req.action_primitive = action_primitive

        rospy.loginfo('Gripper push augmented start_point: (' +
                      str(push_req.start_point.point.x) + ', ' +
                      str(push_req.start_point.point.y) + ', ' +
                      str(push_req.start_point.point.z) + ')')

        rospy.loginfo("Calling gripper feedback pre push service")
        pre_push_res = self.gripper_feedback_pre_push_proxy(push_req)
        rospy.loginfo("Calling gripper feedback push service")

        if _TEST_START_POSE:
            raw_input('waiting for input to recall arm: ')
            push_res = FeedbackPushResponse()
        else:
            push_res = self.gripper_feedback_push_proxy(push_req)
        rospy.loginfo("Calling gripper feedback post push service")
        post_push_res = self.gripper_feedback_post_push_proxy(push_req)
        return push_res

    def feedback_sweep_object(self, which_arm, push_vector, goal_pose, controller_name,
                              proxy_name, action_primitive, high_init=True, open_gripper=False):
        # Convert pose response to correct push request format
        push_req = FeedbackPushRequest()
        push_req.start_point.header = push_vector.header
        push_req.start_point.point = push_vector.start_point
        push_req.open_gripper = open_gripper
        push_req.goal_pose = goal_pose

        # if push_req.left_arm:
        if push_vector.push_angle > 0:
            y_offset_dir = -1
            wrist_yaw = push_vector.push_angle - pi/2
        else:
            y_offset_dir = +1
            wrist_yaw = push_vector.push_angle + pi/2

        push_req.wrist_yaw = wrist_yaw

        # Offset pose to not hit the object immediately
        push_req.start_point.point.x += -self.sweep_offset_dist*sin(wrist_yaw)
        push_req.start_point.point.y += y_offset_dir*self.sweep_offset_dist*cos(wrist_yaw)
        push_req.start_point.point.z = self.sweep_start_z
        push_req.left_arm = (which_arm == 'l')
        push_req.right_arm = not push_req.left_arm
        push_req.high_arm_init = high_init
        push_req.controller_name = controller_name
        push_req.proxy_name = proxy_name
        push_req.action_primitive = action_primitive

        rospy.loginfo('Gripper sweep augmented start_point: (' +
                      str(push_req.start_point.point.x) + ', ' +
                      str(push_req.start_point.point.y) + ', ' +
                      str(push_req.start_point.point.z) + ')')

        rospy.loginfo("Calling feedback pre sweep service")
        pre_push_res = self.gripper_feedback_pre_sweep_proxy(push_req)
        rospy.loginfo("Calling feedback sweep service")

        if _TEST_START_POSE:
            raw_input('waiting for input to recall arm: ')
            push_res = FeedbackPushResponse()
        else:
            push_res = self.gripper_feedback_sweep_proxy(push_req)
        rospy.loginfo("Calling feedback post sweep service")
        post_push_res = self.gripper_feedback_post_sweep_proxy(push_req)
        return push_res

    def generate_random_table_pose(self):
        # TODO: make these parameters
        min_x = 0.4
        max_x = 0.85
        max_y = 0.3
        min_y = -max_y
        max_theta = pi
        min_theta = -pi
        rand_pose = Pose2D()
        rand_pose.x = random.uniform(min_x, max_x)
        rand_pose.y = random.uniform(min_y, max_y)
        rand_pose.theta = random.uniform(min_theta, max_theta)
        rospy.loginfo('Rand table pose is: (' + str(rand_pose.x) + ', ' + str(rand_pose.y) +
                      ', ' + str(rand_pose.theta) + ')')
        return rand_pose

if __name__ == '__main__':
    random.seed()
    use_singulation = False
    use_learning = True
    use_guided = True
    max_pushes = 50
    action_primitive = OVERHEAD_PUSH # GRIPPER_PUSH, GRIPPER_SWEEP, OVERHEAD_PUSH
    node = TabletopExecutive(use_singulation, use_learning)
    if use_singulation:
        node.run_singulation(max_pushes, use_guided)
    else:
        # node.run_feedback_testing(action_primitive)
        node.run_push_exploration()
        node.finish_learning()
