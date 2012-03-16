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
import hrl_pr2_lib.linear_move as lm
import hrl_pr2_lib.pr2 as pr2
import hrl_lib.tf_utils as tfu
from geometry_msgs.msg import PoseStamped
import tf
import numpy as np
from tabletop_pushing.srv import *
from tabletop_pushing.msg import *
from math import sin, cos, pi, fabs
import sys

GRIPPER_PUSH = 0
GRIPPER_SWEEP = 1
OVERHEAD_PUSH = 2
OVERHEAD_PULL = 3

class TabletopExecutive:

    def __init__(self, use_singulation, use_learning):
        rospy.init_node('tabletop_executive_node',log_level=rospy.DEBUG)
        # TODO: Determine workspace limits for max here
        self.min_push_dist = rospy.get_param('~min_push_dist', 0.07)
        self.max_push_dist = rospy.get_param('~mix_push_dist', 0.3)
        self.use_overhead_x_thresh = rospy.get_param('~use_overhead_x_thresh',
                                                     0.55)
        self.use_sweep_angle_thresh = rospy.get_param('~use_sweep_angle_thresh',
                                                     pi*0.4)
        self.use_pull_angle_thresh = rospy.get_param('~use_sweep_angle_thresh',
                                                     pi*0.525)
        self.use_same_side_y_thresh = rospy.get_param('~use_same_side_y_thresh',
                                                     0.3)
        self.use_same_side_x_thresh = rospy.get_param('~use_same_side_x_thresh',
                                                      0.8)

        # TODO: Replace these parameters with learned / perceived values
        # The offsets should be removed and learned implicitly
        self.gripper_x_offset = rospy.get_param('~gripper_push_start_x_offset',
                                                -0.05)
        self.gripper_y_offset = rospy.get_param('~gripper_push_start_x_offset',
                                                0.0)
        self.gripper_start_z = rospy.get_param('~gripper_push_start_z',
                                                -0.25)

        self.sweep_x_offset = rospy.get_param('~gripper_sweep_start_x_offset',
                                              -0.01)
        self.sweep_y_offset = rospy.get_param('~gripper_sweep_start_y_offset',
                                              0.03)
        self.sweep_start_z = rospy.get_param('~gripper_sweep_start_z',
                                              -0.22)

        self.overhead_x_offset = rospy.get_param('~overhead_push_start_x_offset',
                                                 0.00)
        self.overhead_y_offset = rospy.get_param('~overhead_push_start_x_offset',
                                                 0.00)
        self.overhead_start_z = rospy.get_param('~overhead_push_start_z',
                                                 -0.25)
        self.pull_dist_offset = rospy.get_param('~overhead_pull_dist_offset',
                                                0.05)
        self.pull_start_z = rospy.get_param('~overhead_push_start_z',
                                            -0.27)
        # Setup service proxies
        self.gripper_push_proxy = rospy.ServiceProxy('gripper_push',
                                                     GripperPush)
        self.gripper_pre_push_proxy = rospy.ServiceProxy('gripper_pre_push',
                                                         GripperPush)
        self.gripper_post_push_proxy = rospy.ServiceProxy('gripper_post_push',
                                                          GripperPush)
        self.gripper_pre_sweep_proxy = rospy.ServiceProxy('gripper_pre_sweep',
                                                          GripperPush)
        self.gripper_sweep_proxy = rospy.ServiceProxy('gripper_sweep',
                                                      GripperPush)
        self.gripper_post_sweep_proxy = rospy.ServiceProxy('gripper_post_sweep',
                                                           GripperPush)
        self.overhead_pre_push_proxy = rospy.ServiceProxy('overhead_pre_push',
                                                          GripperPush)
        self.overhead_push_proxy = rospy.ServiceProxy('overhead_push',
                                                      GripperPush)
        self.overhead_post_push_proxy = rospy.ServiceProxy('overhead_post_push',
                                                           GripperPush)
        self.overhead_pre_pull_proxy = rospy.ServiceProxy('overhead_pre_pull',
                                                          GripperPush)
        self.overhead_pull_proxy = rospy.ServiceProxy('overhead_pull',
                                                      GripperPush)
        self.overhead_post_pull_proxy = rospy.ServiceProxy('overhead_post_pull',
                                                           GripperPush)
        self.raise_and_look_proxy = rospy.ServiceProxy('raise_and_look',
                                                       RaiseAndLook)
        self.table_proxy = rospy.ServiceProxy('get_table_location', LocateTable)

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
        # TODO: Setup save file
        self.learning_push_vector_proxy = rospy.ServiceProxy(
            'get_learning_push_vector', LearnPush)

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
                push_opt = OVERHEAD_PULL
            elif pose_res.start_point.x < self.use_overhead_x_thresh:
                push_opt = OVERHEAD_PUSH
            elif fabs(pose_res.push_angle) > self.use_sweep_angle_thresh:
                push_opt = GRIPPER_SWEEP
            else:
                push_opt = GRIPPER_PUSH

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
            if push_opt == GRIPPER_PUSH:
                self.gripper_push_object(push_dist, which_arm, pose_res)
            if push_opt == GRIPPER_SWEEP:
                self.sweep_object(push_dist, which_arm, pose_res)
            if push_opt == OVERHEAD_PUSH:
                self.overhead_push_object(push_dist, which_arm, pose_res)
            if push_opt == OVERHEAD_PULL:
                self.overhead_pull_object(push_dist, which_arm, pose_res)
            rospy.loginfo('Done performing push behavior.\n')

        if not (pose_res is None):
            rospy.loginfo('Singulated objects: ' + str(pose_res.singulated))
            rospy.loginfo('Final estimate of ' + str(pose_res.num_objects) +
                          ' objects')

    def run_learning(self, num_pushes=1):
        # Get table height and raise to that before anything else
        self.raise_and_look()
        # Initialize push pose
        self.initialize_learning_push();

        # TODO: Get angle and distance correctly...
        push_angle = 0.0 # radians
        push_dist = 0.4 # meters
        push_opts = [GRIPPER_PUSH, OVERHEAD_PUSH, GRIPPER_SWEEP, OVERHEAD_PULL]
        arms = ['l', 'r']
        # NOTE: Should exit before reaching num_pushes, this is just a backup
        for i in xrange(num_pushes):
            rospy.loginfo('Place item at new initial pose')
            for arm in arms:
                for push_opt in push_opts:
                    res = perform_learning_trial(arm, push_opt, push_angle, push_dist)

    def perform_learning_trial(self, which_arm, push_opt, push_angle, push_dist):
        raw_input('Reset item to inital pose and press any key to continue')
        push_res = self.request_learning_push(push_angle)
        if push_res is None:
            rospy.logwarn("push_res is None. Exiting pushing");
            break
        if push_res.no_push:
            rospy.loginfo("No push. Exiting pushing.");
            return False
        # Decide push based on the orientation returned
        rospy.loginfo('Push start_point: (' + str(push_res.push.start_point.x) +
                      ', ' + str(push_res.push.start_point.y) +
                      ', ' + str(push_res.push.start_point.z) + ')')
        rospy.loginfo('Push angle: ' + str(push_res.push.push_angle))
        # rospy.loginfo('Returned push dist: ' + str(push_res.push_dist))
        rospy.loginfo('Push dist: ' + str(push_dist))

        if push_opt == GRIPPER_PUSH:
            self.gripper_push_object(push_dist, which_arm, push_res.push)
        if push_opt == GRIPPER_SWEEP:
            self.sweep_object(push_dist, which_arm, push_res.push)
        if push_opt == OVERHEAD_PUSH:
            self.overhead_push_object(push_dist, which_arm, push_res.push)
        if push_opt == OVERHEAD_PULL:
            self.overhead_pull_object(push_dist, which_arm, push_res.push)
        rospy.loginfo('Done performing push behavior.')
        analysis_res = self.request_learning_analysis()
        rospy.loginfo('Done getting analysis response.')
        # TODO: Save analysis to disk
        # push_res.centroid.x, push_res.centroid.y, push_res.centroid.z, theta, dist,
        # which_arm, push_opt, score, analysis_res.moved.x, analysis_res.moved.y,
        # analysis_res.moved.z, analysis_res.dist
        # TODO: save initial (x,y)

    def request_singulation_push(self, use_guided=True):
        push_vector_req = SingulationPushRequst()
        push_vector_req.use_guided = use_guided
        push_vector_req.initialize = False
        push_vector_req.no_push_calc = False
        rospy.loginfo("Calling singulation push vector service")
        try:
            push_res = self.singulation_push_vector_proxy(push_vector_req)
            return push_res
        except rospy.ServiceException, e:
            rospy.logwarn("Service did not process request: %s"%str(e))
            return None

    def request_learning_push(self, push_angle):
        push_req = LearnPushRequest()
        push_req.initialize = False
        push_req.analyze_previous = False
        push_req.push_angle = push_angle
        rospy.loginfo("Calling learning push vector service")
        try:
            push_res = self.learning_push_vector_proxy(push_vector_req)
            return push_res
        except rospy.ServiceException, e:
            rospy.logwarn("Service did not process request: %s"%str(e))
            return None

    def request_learning_analysis(self):
        push_req = LearnPushRequest()
        push_req.initialize = False
        push_req.analyze_previous = True
        rospy.loginfo("Calling learning push vector service")
        try:
            push_res = self.learning_push_vector_proxy(push_vector_req)
            return push_res
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
        push_vector_req = SingulationPushRequest()
        push_vector_req.initialize = True
        push_vector_req.analyze_previous = False
        rospy.loginfo('Initializing learning push vector service.')
        self.learning_push_vector_proxy(push_vector_req)

    def raise_and_look(self):
        table_req = LocateTableRequest()
        table_req.recalculate = True
        raise_req = RaiseAndLookRequest()
        raise_req.point_head_only = True
        raise_req.camera_frame = 'openni_rgb_frame'
        # First make sure the head is looking the correct way before estimating
        # the table centroid
        # Also make sure the arms are out of the way
        raise_req.init_arms = True
        rospy.loginfo("Moving head and arms")
        raise_res = self.raise_and_look_proxy(raise_req)
        try:
            rospy.loginfo("Getting table pose")
            table_res = self.table_proxy(table_req);
        except rospy.ServiceException, e:
            rospy.logwarn("Service did not process request: %s"%str(e))
            return
        if not table_res.found_table:
            return
        raise_req.table_centroid = table_res.table_centroid

        # TODO: Make sure this requested table_centroid is valid

        rospy.loginfo("Raising spine");
        raise_req.point_head_only = False
        raise_req.init_arms = False
        raise_res = self.raise_and_look_proxy(raise_req)

    def gripper_push_object(self, push_dist, which_arm, pose_res):
        # Convert pose response to correct push request format
        push_req = GripperPushRequest()
        push_req.start_point.header = pose_res.header
        push_req.start_point.point = pose_res.start_point
        push_req.arm_init = True
        push_req.arm_reset = True

        # Use the sent wrist yaw
        wrist_yaw = pose_res.push_angle
        push_req.wrist_yaw = wrist_yaw
        push_req.desired_push_dist = push_dist + abs(self.gripper_x_offset)

        # Offset pose to not hit the object immediately
        push_req.start_point.point.x += self.gripper_x_offset*cos(wrist_yaw)
        push_req.start_point.point.y += self.gripper_x_offset*sin(wrist_yaw)
        push_req.start_point.point.z = self.gripper_start_z
        push_req.left_arm = (which_arm == 'l')
        push_req.right_arm = not push_req.left_arm

        rospy.loginfo("Calling gripper pre push service")
        pre_push_res = self.gripper_pre_push_proxy(push_req)
        rospy.loginfo("Calling gripper push service")
        push_res = self.gripper_push_proxy(push_req)
        rospy.loginfo("Calling gripper post push service")
        post_push_res = self.gripper_post_push_proxy(push_req)

    def sweep_object(self, push_dist, which_arm, pose_res):
        # Convert pose response to correct push request format
        sweep_req = GripperPushRequest()
        sweep_req.left_arm = (which_arm == 'l')
        sweep_req.right_arm = not sweep_req.left_arm

        # if sweep_req.left_arm:
        if pose_res.push_angle > 0:
            y_offset_dir = -1
        else:
            y_offset_dir = +1

        # Correctly set the wrist yaw
        if pose_res.push_angle > 0.0:
            wrist_yaw = pose_res.push_angle - pi/2
        else:
            wrist_yaw = pose_res.push_angle + pi/2
        sweep_req.wrist_yaw = wrist_yaw
        sweep_req.desired_push_dist = -y_offset_dir*(self.sweep_y_offset +
                                                     push_dist)

        # Set offset in x y, based on distance
        sweep_req.start_point.header = pose_res.header
        sweep_req.start_point.point = pose_res.start_point
        sweep_req.start_point.point.x += self.sweep_x_offset
        sweep_req.start_point.point.y += y_offset_dir*self.sweep_y_offset
        sweep_req.start_point.point.z = self.sweep_start_z
        sweep_req.arm_init = True
        sweep_req.arm_reset = True

        rospy.loginfo("Calling gripper pre sweep service")
        pre_sweep_res = self.gripper_pre_sweep_proxy(sweep_req)
        rospy.loginfo("Calling gripper sweep service")
        sweep_res = self.gripper_sweep_proxy(sweep_req)
        rospy.loginfo("Calling gripper post sweep service")
        post_sweep_res = self.gripper_post_sweep_proxy(sweep_req)

    def overhead_push_object(self, push_dist, which_arm, pose_res):
        # Convert pose response to correct push request format
        push_req = GripperPushRequest()
        push_req.start_point.header = pose_res.header
        push_req.start_point.point = pose_res.start_point
        push_req.arm_init = True
        push_req.arm_reset = True

        # Correctly set the wrist yaw
        wrist_yaw = pose_res.push_angle
        push_req.wrist_yaw = wrist_yaw
        push_req.desired_push_dist = push_dist

        # Offset pose to not hit the object immediately
        push_req.start_point.point.x += self.overhead_x_offset*cos(wrist_yaw)
        push_req.start_point.point.y += self.overhead_x_offset*sin(wrist_yaw)
        push_req.start_point.point.z = self.overhead_start_z
        push_req.left_arm = (which_arm == 'l')
        push_req.right_arm = not push_req.left_arm

        rospy.loginfo("Calling pre overhead push service")
        pre_push_res = self.overhead_pre_push_proxy(push_req)
        rospy.loginfo("Calling overhead push service")
        push_res = self.overhead_push_proxy(push_req)
        rospy.loginfo("Calling post overhead push service")
        post_push_res = self.overhead_post_push_proxy(push_req)

    def overhead_pull_object(self, push_dist, which_arm, pose_res):
        # Convert pose response to correct push request format
        push_req = GripperPushRequest()
        push_req.start_point.header = pose_res.header
        push_req.start_point.point = pose_res.start_point
        push_req.arm_init = True
        push_req.arm_reset = True

        wrist_yaw = pose_res.push_angle
        # Correctly set the wrist yaw
        while wrist_yaw < -pi*0.5:
            wrist_yaw += pi
        while wrist_yaw > pi*0.5:
            wrist_yaw -= pi
        push_req.wrist_yaw = wrist_yaw
        # Add offset distance to push to compensate
        push_req.desired_push_dist = push_dist + self.pull_dist_offset

        # Offset pose to not hit the object immediately
        rospy.loginfo('Pre pull offset (x,y): (' +
                      str(push_req.start_point.point.x) + ', ' +
                      str(push_req.start_point.point.y) + ')')
        push_req.start_point.point.x += self.pull_dist_offset*cos(wrist_yaw)
        push_req.start_point.point.y += self.pull_dist_offset*sin(wrist_yaw)
        push_req.start_point.point.z = self.pull_start_z
        push_req.left_arm = (which_arm == 'l')
        push_req.right_arm = not push_req.left_arm
        rospy.loginfo('Post pull offset (x,y): (' +
                      str(push_req.start_point.point.x) + ', ' +
                      str(push_req.start_point.point.y) + ')')

        rospy.loginfo("Calling pre overhead pull service")
        pre_push_res = self.overhead_pre_pull_proxy(push_req)
        rospy.loginfo("Calling overhead pull service")
        push_res = self.overhead_pull_proxy(push_req)
        rospy.loginfo("Calling post overhead pull service")
        post_push_res = self.overhead_post_pull_proxy(push_req)

if __name__ == '__main__':
    use_learning = True
    use_singulation = False
    use_guided = True
    node = TabletopExecutive(use_singulation, use_learning)
    if use_singulation:
        node.run_singulation(50, use_guided)
    else:
        node.run_learning(50)
