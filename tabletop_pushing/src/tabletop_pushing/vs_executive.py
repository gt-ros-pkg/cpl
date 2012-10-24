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
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import PoseStamped
import tf
from visual_servo.srv import *
# import tabletop_pushing.position_feedback_push_node as pn

import tabletop_pushing.position_feedback_push_node as pn
import sys
import numpy as np
import math
from math import sqrt

class ArmController:

  def __init__(self):

    rospy.loginfo('Initializing Arm Controller')
    self.pn = pn.PositionFeedbackPushNode()
    self.pn.init_spine_pose()
    self.pn.init_head_pose(self.pn.head_pose_cam_frame)
    self.pn.init_arms()

    rospy.loginfo('Done moving to robot initial pose')

  def run(self):
    rospy.loginfo("[vs_exec] === New Iter ===") # %s"%str(start_pose))
    start_pose = PoseStamped()
    start_pose.header.frame_id = 'torso_lift_link'
    start_pose.pose.position.x = 0.4
    start_pose.pose.position.y = 0.15
    start_pose.pose.position.z = -0.10
    start_pose.pose.orientation.x = 0
    start_pose.pose.orientation.y = 0
    start_pose.pose.orientation.z = 1
    start_pose.pose.orientation.w = 0
    which_arm = 'l'

    self.pn.move_to_cart_pose(start_pose, which_arm)
    r = rospy.Rate(0.5)
    r.sleep()

    vs_pose = PoseStamped()
    vs_pose.header.frame_id = 'torso_lift_link'
    vs_pose.pose.position.x = 0.48
    vs_pose.pose.position.y = 0
    vs_pose.pose.position.z = -0.23
    vs_pose.pose.orientation.x = 0
    vs_pose.pose.orientation.y = 0
    vs_pose.pose.orientation.z = 1
    vs_pose.pose.orientation.w = 0
    self.pn.vs_action_exec(vs_pose, which_arm)

    r = rospy.Rate(1.0)
    r.sleep()
    raw_input('Wait for input for another iteration: ')


if __name__ == '__main__':
  try:
    node = ArmController()
    while True:
      node.run()
    rospy.spin()
  except rospy.ROSInterruptException: pass

  rospy.loginfo("Terminating")
