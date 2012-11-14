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

import roslib; roslib.load_manifest('pr2_gripper_segmentation')
import rospy
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import PoseStamped
import tf
from pr2_gripper_segmentation.srv import *
import tabletop_pushing.position_feedback_push_node as pn
import sys
import numpy as np
import math
from math import sqrt

class ArmController:

    def __init__(self):

        rospy.loginfo('Initializing Arm Controller')
        # Initialize vel controller
        self.pn = pn.PositionFeedbackPushNode()
        self.l_cart_pub = self.pn.l_arm_cart_pub
        self.r_cart_pub = self.pn.r_arm_cart_pub
        self.pn.init_spine_pose()
        self.pn.init_head_pose(self.pn.head_pose_cam_frame)
        self.pn.init_arms()
        rospy.loginfo('Done moving to robot initial pose')
# open both gripper for init
# self.pn.gripper_open('l')
# self.pn.gripper_open('r')


    def handle_pose_request(self, req):
      print req.p
      pose = req.p # PoseStamped
      which_arm = req.arm
      if which_arm == 'l':
        self.pn.move_to_cart_pose(pose, 'l')
      else:
        self.pn.move_to_cart_pose(pose, 'r')
      rospy.loginfo('[arm_controller] pose')
      rospy.loginfo(pose)
      return GripperPoseResponse()

if __name__ == '__main__':
  try:
    node = ArmController()
    rospy.loginfo('Done initializing... Now advertise the Service')
    rospy.Service('pgs_pose', GripperPose , node.handle_pose_request)
    rospy.loginfo('Ready to move arm')
    rospy.spin()

  except rospy.ROSInterruptException: pass
