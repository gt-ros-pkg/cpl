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
from math import sin, cos, pi
import sys
import random

class TabletopExecutive:

    def __init__(self, use_fake_push_pose=False):
        random.seed()
        # TODO: Replace these parameters with learned / perceived values
        # The offsets should be removed and learned implicitly
        rospy.init_node('tabletop_executive_node',log_level=rospy.DEBUG)
        self.gripper_push_dist = rospy.get_param('~gripper_push_dist',
                                                 0.30)
        self.gripper_x_offset = rospy.get_param('~gripper_push_start_x_offset',
                                                -0.03)
        self.gripper_y_offset = rospy.get_param('~gripper_push_start_x_offset',
                                                0.0)
        self.gripper_z_offset = rospy.get_param('~gripper_push_start_z_offset',
                                                0.02)

        self.gripper_sweep_dist = rospy.get_param('~gripper_sweep_dist',
                                                 0.25)
        self.sweep_x_offset = rospy.get_param('~gripper_sweep_start_x_offset',
                                              0.10)
        self.sweep_y_offset = rospy.get_param('~gripper_sweep_start_y_offset',
                                              0.05)
        self.sweep_z_offset = rospy.get_param('~gripper_sweep_start_z_offset',
                                              0.01)

        self.overhead_push_dist = rospy.get_param('~overhead_push_dist',
                                                 0.25)
        self.overhead_x_offset = rospy.get_param('~overhead_push_start_x_offset',
                                                 0.00)
        self.overhead_y_offset = rospy.get_param('~overhead_push_start_x_offset',
                                                 0.00)
        self.overhead_z_offset = rospy.get_param('~overhead_push_start_z_offset',
                                                 0.01)

        self.push_pose_proxy = rospy.ServiceProxy('get_push_pose', PushPose)
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
        self.raise_and_look_proxy = rospy.ServiceProxy('raise_and_look',
                                                       RaiseAndLook)
        self.table_proxy = rospy.ServiceProxy('get_table_location', LocateTable)
        self.singulation_client = actionlib.SimpleActionClient(
            "singulation_action", ObjectSingulationAction)
        self.use_fake_push_pose = use_fake_push_pose
        self.push_count = 0

    def run_rand(self, num_pushes):
        # Get table height and raise to that before anything else
        self.raise_and_look()

        # Setup perception system
        self.init_singulation()
        self.num_total_pushes = 0
        for i in xrange(num_pushes):
            opt = random.randint(0,num_pushes-1)
            arm = random.randint(0,1)

            if arm == 0:
                which_arm='l'
            else:
                which_arm='r'
            if opt == 0:
                self.gripper_push_object(self.gripper_push_dist, which_arm)
            if opt == 1:
                self.sweep_object(self.gripper_push_dist, which_arm)
            if opt == 2:
                self.overhead_push_object(self.gripper_push_dist, which_arm)


    def run(self,
            num_l_gripper_pushes, num_l_sweeps, num_l_overhead_pushes,
            num_r_gripper_pushes, num_r_sweeps, num_r_overhead_pushes):

        # Get table height and raise to that before anything else
        self.raise_and_look()

        # Setup perception system
        self.init_singulation()

        self.push_count = 0
        self.num_total_pushes = num_l_gripper_pushes
        for i in xrange(num_l_gripper_pushes):
            self.gripper_push_object(self.gripper_push_dist, 'l')

        self.push_count = 0
        self.num_total_pushes = num_r_gripper_pushes
        for i in xrange(num_r_gripper_pushes):
            self.gripper_push_object(self.gripper_push_dist, 'r')

        self.sweep_count = 0
        self.num_total_sweeps = num_l_sweeps
        for i in xrange(num_l_sweeps):
            self.sweep_object(self.gripper_sweep_dist, 'l')

        self.sweep_count = 0
        self.num_total_sweeps = num_r_sweeps
        for i in xrange(num_r_sweeps):
            self.sweep_object(self.gripper_sweep_dist, 'r')

        self.push_count = 0
        self.num_total_pushes = num_l_overhead_pushes
        for i in xrange(num_l_overhead_pushes):
            self.overhead_push_object(self.overhead_push_dist, 'l')

        self.push_count = 0
        self.num_total_pushes = num_r_overhead_pushes
        for i in xrange(num_r_overhead_pushes):
            self.overhead_push_object(self.overhead_push_dist, 'r')

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

    def init_singulation(self):
        singulation_goal = ObjectSingulationGoal()
        singulation_goal.start = False
        singulation_goal.init = True
        self.singulation_client.send_goal(singulation_goal)
        rospy.loginfo('Waiting for singulation server')
        self.singulation_client.wait_for_server()

    def start_singulation(self):
        singulation_goal = ObjectSingulationGoal()
        singulation_goal.start = True
        singulation_goal.init = False
        self.singulation_client.send_goal(singulation_goal)
        rospy.loginfo('Waiting for singulation server')
        self.singulation_client.wait_for_server()

    def stop_singulation(self):
        singulation_goal = ObjectSingulationGoal()
        singulation_goal.start = False
        singulation_goal.init = False
        singulation_goal.get_singulation_vector = True
        self.singulation_client.send_goal(singulation_goal)
        rospy.loginfo('Waiting for singulation server')
        self.singulation_client.wait_for_result()
        result = self.singulation_client.get_result()
        return result.singulation_vector

    def gripper_push_object(self, push_dist, which_arm):
        # Make push_pose service request
        pose_req = PushPoseRequest()
        pose_req.use_laser = False

        # TODO: Pull this out to a higher level method
        # TODO: Keep track of statefullness as to whether it is an initial
        # probe or a testing probe
        # Get push pose
        rospy.loginfo("Calling push pose service")
        try:
            if self.use_fake_push_pose:
                pose_res = PushPoseResponse()
                pose_res.push_pose.pose.position.x = 0.5
                pose_res.push_pose.header.frame_id = '/torso_lift_link'
                pose_res.push_pose.header.stamp = rospy.Time(0)
                if which_arm == 'l':
                    pose_res.push_pose.pose.position.y = 0.3 - self.push_count*0.15
                else:
                    pose_res.push_pose.pose.position.y = -0.3 + self.push_count*0.15
                # TODO: -z_offset+8
                pose_res.push_pose.pose.position.z = -0.22
            else:
                rospy.loginfo('Calling push pose proxy')
                pose_res = self.push_pose_proxy(pose_req)
                if (pose_res.push_pose.pose.position.x == 0.0 and
                    pose_res.push_pose.pose.position.y == 0.0 and
                    pose_res.push_pose.pose.position.z == 0.0):
                    rospy.logwarn('No push pose found.')
                    return
                rospy.loginfo('Got push pose')
                if pose_res.push_pose.pose.position.y < 0.0:
                    which_arm = 'r'
                else:
                    which_arm = 'l'
        except rospy.ServiceException, e:
            rospy.logwarn("Service did not process request: %s"%str(e))
            return
        self.push_count += 1

        # Convert pose response to correct push request format
        push_req = GripperPushRequest()
        push_req.start_point.header = pose_res.push_pose.header
        push_req.start_point.point = pose_res.push_pose.pose.position
        # if self.push_count == 1:
        #     push_req.arm_init = True
        # else:
        #     push_req.arm_init = False
        # if self.push_count < self.num_total_pushes:
        #     push_req.arm_reset = False
        # else:
        #     push_req.arm_reset = True

        push_req.arm_init = True
        push_req.arm_reset = True

        # wrist_yaw = 0.0 # 0.5*pi # 0.25*pi # -0.25*pi
        wrist_yaw = pose_res.push_pose.pose.orientation.y
        push_req.wrist_yaw = wrist_yaw
        push_req.desired_push_dist = push_dist

        # TODO: Remove these offsets and incorporate them directly into the
        # perceptual inference
        # Offset pose to not hit the object immediately
        push_req.start_point.point.x += self.gripper_x_offset*cos(wrist_yaw)
        push_req.start_point.point.y += self.gripper_x_offset*sin(wrist_yaw)
        push_req.start_point.point.z += self.gripper_z_offset

        push_req.left_arm = (which_arm == 'l')
        push_req.right_arm = not push_req.left_arm

        rospy.loginfo("Calling gripper pre push service")
        pre_push_res = self.gripper_pre_push_proxy(push_req)
        self.start_singulation()
        rospy.loginfo("Calling gripper push service")
        push_res = self.gripper_push_proxy(push_req)
        # TODO: Get meaningful singulation response
        singulation_res = self.stop_singulation()
        rospy.loginfo("Calling gripper post push service")
        post_push_res = self.gripper_post_push_proxy(push_req)

    def sweep_object(self, push_dist, which_arm):
        # Make push_pose service request
        pose_req = PushPoseRequest()
        pose_req.use_laser = False

        # Get push pose
        rospy.loginfo("Calling push pose service")
        try:
            if self.use_fake_push_pose:
                pose_res = PushPoseResponse()
                pose_res.push_pose.header.frame_id = '/torso_lift_link'
                pose_res.push_pose.header.stamp = rospy.Time(0)
                pose_res.push_pose.pose.position.x = 0.75
                if which_arm == 'l':
                    pose_res.push_pose.pose.position.y = 0.05
                else:
                    pose_res.push_pose.pose.position.y = -0.15
                pose_res.push_pose.pose.position.z = -0.25
            else:
                rospy.loginfo('Calling push pose proxy')
                pose_res = self.push_pose_proxy(pose_req)
                if (pose_res.push_pose.pose.position.x == 0.0 and
                    pose_res.push_pose.pose.position.y == 0.0 and
                    pose_res.push_pose.pose.position.z == 0.0):
                    rospy.logwarn('No push pose found.')
                    return

                rospy.loginfo('Got push pose')
                if pose_res.push_pose.pose.position.y < 0.0:
                    which_arm = 'r'
                else:
                    which_arm = 'l'
                    # TODO: Set offset here

        except rospy.ServiceException, e:
            rospy.logwarn("Service did not process request: %s"%str(e))
            return
        self.sweep_count += 1
        # Convert pose response to correct push request format
        sweep_req = GripperPushRequest()
        sweep_req.left_arm = (which_arm == 'l')
        sweep_req.right_arm = not sweep_req.left_arm

        if sweep_req.left_arm:
            y_offset_dir = +1
        else:
            y_offset_dir = -1
        sweep_req.start_point.header = pose_res.push_pose.header
        sweep_req.start_point.point = pose_res.push_pose.pose.position
        sweep_req.start_point.point.y += y_offset_dir*self.sweep_y_offset
        sweep_req.start_point.point.z += self.sweep_z_offset
        if self.sweep_count == 1:
            sweep_req.arm_init = True
        else:
            sweep_req.arm_init = False
        if self.sweep_count < self.num_total_sweeps:
            sweep_req.arm_reset = False
        else:
            sweep_req.arm_reset = True

        sweep_req.arm_init = True
        sweep_req.arm_reset = True

        # rospy.loginfo('Push pose point:' + str(sweep_req.start_point.point))

        # TODO: Correctly set the wrist yaw
        # orientation = pose_res.push_pose.pose.orientation

        wrist_yaw = 0.0 # 0.25*pi
        sweep_req.wrist_yaw = wrist_yaw
        sweep_req.desired_push_dist = -y_offset_dir*push_dist


        # TODO: Remove these offsets and incorporate them directly into the perceptual inference
        # Offset pose to not hit the object immediately
        sweep_req.start_point.point.x += (self.sweep_x_offset*cos(wrist_yaw) +
                                          self.sweep_y_offset*sin(wrist_yaw))
        sweep_req.start_point.point.y += y_offset_dir*(
            self.sweep_x_offset*sin(wrist_yaw) + self.sweep_y_offset*cos(wrist_yaw))
        sweep_req.start_point.point.z += self.sweep_z_offset

        # rospy.loginfo('Sweep start point:' + str(sweep_req.start_point.point))
        rospy.loginfo("Calling gripper pre sweep service")
        pre_sweep_res = self.gripper_pre_sweep_proxy(sweep_req)
        self.start_singulation()
        rospy.loginfo("Calling gripper sweep service")
        sweep_res = self.gripper_sweep_proxy(sweep_req)
        self.stop_singulation()
        rospy.loginfo("Calling gripper post sweep service")
        post_sweep_res = self.gripper_post_sweep_proxy(sweep_req)

    def overhead_push_object(self, push_dist, which_arm):
        # Make push_pose service request
        pose_req = PushPoseRequest()
        pose_req.use_laser = False

        # Get push pose
        rospy.loginfo("Calling push pose service")
        try:
            if self.use_fake_push_pose:
                pose_res = PushPoseResponse()
                pose_res.push_pose.header.frame_id = '/torso_lift_link'
                pose_res.push_pose.header.stamp = rospy.Time(0)
                pose_res.push_pose.pose.position.x = 0.7
                pose_res.push_pose.pose.position.y = 0.0
                pose_res.push_pose.pose.position.z = -0.25
            else:
                rospy.loginfo('Calling push pose proxy')
                pose_res = self.push_pose_proxy(pose_req)
                if (pose_res.push_pose.pose.position.x == 0.0 and
                    pose_res.push_pose.pose.position.y == 0.0 and
                    pose_res.push_pose.pose.position.z == 0.0):
                    rospy.logwarn('No push pose found.')
                    return

                rospy.loginfo('Got push pose')
                if pose_res.push_pose.pose.position.y < 0.0:
                    which_arm = 'r'
                else:
                    which_arm = 'l'

        except rospy.ServiceException, e:
            rospy.logwarn("Service did not process request: %s"%str(e))
            return

        self.push_count += 1

        # Convert pose response to correct push request format
        push_req = GripperPushRequest()
        push_req.start_point.header = pose_res.push_pose.header
        push_req.start_point.point = pose_res.push_pose.pose.position
        if self.push_count == 1:
            push_req.arm_init = True
        else:
            push_req.arm_init = False
        if self.push_count < self.num_total_pushes:
            push_req.arm_reset = False
        else:
            push_req.arm_reset = True

        push_req.arm_init = True
        push_req.arm_reset = True

        # TODO: Correctly set the wrist yaw
        # orientation = pose_res.push_pose.pose.orientation
        wrist_yaw = 0.0 # -0.25*pi
        push_req.wrist_yaw = wrist_yaw
        push_req.desired_push_dist = push_dist

        push_req.left_arm = (which_arm == 'l')
        push_req.right_arm = not push_req.left_arm

        if push_req.left_arm:
            y_offset_dir = +1
        else:
            y_offset_dir = -1

        # TODO: Remove these offsets and incorporate them directly into the perceptual inference
        # Offset pose to not hit the object immediately
        push_req.start_point.point.x += (self.overhead_x_offset*cos(wrist_yaw) +
                                          self.overhead_y_offset*sin(wrist_yaw))
        push_req.start_point.point.y += y_offset_dir*(
            self.overhead_x_offset*sin(wrist_yaw) +
            self.overhead_y_offset*cos(wrist_yaw))
        push_req.start_point.point.z += self.overhead_z_offset

        # rospy.loginfo('Push start point:' + str(push_req.start_point.point))

        rospy.loginfo("Calling pre overhead push service")
        pre_push_res = self.overhead_pre_push_proxy(push_req)
        self.start_singulation()
        rospy.loginfo("Calling overhead push service")
        push_res = self.overhead_push_proxy(push_req)
        self.stop_singulation()
        rospy.loginfo("Calling post overhead push service")
        post_push_res = self.overhead_post_push_proxy(push_req)

if __name__ == '__main__':
    node = TabletopExecutive(False)
    # node.run(3, 3, 0, 0, 0, 0)
    node.run_rand(5)
