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

import roslib; roslib.load_manifest('visual_servo')
import rospy
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import PoseStamped
import tf
from visual_servo.srv import *
# import tabletop_pushing.position_feedback_push_node as pn

import visual_servo.position_feedback_push_node as pn
import sys
import numpy as np
from math import sqrt

LEFT_ARM_READY_JOINTS = np.matrix([[0.62427649, 0.4556137,
                                    1.63411927, -2.11931035,
                                    -15.38839978, -1.64163257,
                                    -17.2947453]]).T
RIGHT_ARM_READY_JOINTS = np.matrix([[-0.42427649, 0.0656137,
                                     -1.43411927, -2.11931035,
                                     15.78839978, -1.64163257,
                                     8.64421842e+01]]).T

class ArmController:

    def __init__(self):

        rospy.loginfo('Initializing Arm Controller')
        # Initialize vel controller
        self.pn = pn.PositionFeedbackPushNode()
        self.l_cart_twist_pub = self.pn.l_arm_cart_vel_pub
        self.r_cart_twist_pub = self.pn.r_arm_cart_vel_pub
        self.l_cart_pub = self.pn.l_arm_cart_pub
        self.r_cart_pub = self.pn.r_arm_cart_pub
        self.pn.init_spine_pose()
        self.pn.init_head_pose(self.pn.head_pose_cam_frame)
        self.pn.init_arms()
        # open both gripper for init
        self.pn.gripper_open('l')
        self.pn.gripper_open('r')

        # self.init_arm_servo()
        self.pn.switch_to_cart_controllers()

        # Setup parameters
        self.vel_sat_param = rospy.get_param('/arm_controller/vel_sat_param', 0.20)
        self.vel_scale_param = rospy.get_param('/arm_controller/vel_scale_param', 0.20)

        rospy.loginfo('Done moving to robot initial pose')

        # self.pn.gripper_pose()

    def adjust_velocity(self, vel):
      ret = vel * self.vel_scale
      if ret > self.vel_sat:
        ret = self.vel_sat
      elif -ret > self.vel_sat:
        ret = -self.vel_sat
      return ret 

    def handle_pose_request(self, req):
      pose = req.p # PoseStamped
      which_arm = req.arm
      if which_arm == 'l':
        self.pn.move_to_cart_pose(pose, 'l')
      else:
        self.pn.move_to_cart_pose(pose, 'r')
      rospy.loginfo('[arm_controller] pose')
      rospy.loginfo(pose)
      return {'result': 0}

    def handle_twist_request(self, req):
      t = req.twist # service call
      e = req.error
      which_arm = req.arm

      self.vel_scale = sqrt(e + self.vel_scale_param)
      self.vel_sat = sqrt(e * self.vel_sat_param)
      if self.vel_sat > 0.08:
        self.vel_sat =  0.08
      try:
        twist = TwistStamped()
        twist.header.stamp = rospy.Time(0)
        twist.header.frame_id = 'torso_lift_link'

        # task specific gain control. we know that we are aligned in y... 
        twist.twist.linear.x  = self.adjust_velocity(t.twist.linear.x)
        twist.twist.linear.y  = self.adjust_velocity(t.twist.linear.y) 
        twist.twist.linear.z  = self.adjust_velocity(t.twist.linear.z) 
        twist.twist.angular.x = self.adjust_velocity(t.twist.angular.x)
        twist.twist.angular.y = self.adjust_velocity(t.twist.angular.y)
        twist.twist.angular.z = self.adjust_velocity(t.twist.angular.z)
        if which_arm == 'l':
            vel_pub = self.pn.l_arm_cart_vel_pub
            posture_pub = self.pn.l_arm_vel_posture_pub
            m = self.pn.get_desired_posture('l')
        else:
            vel_pub = self.pn.r_arm_cart_vel_pub
            posture_pub = self.pn.r_arm_vel_posture_pub
            m = self.pn.get_desired_posture('r')
        posture_pub.publish(m)
        vel_pub.publish(twist)

        # after(before) adjustment
        rospy.loginfo('[arm_controller] twist')
        rospy.loginfo('[e=%.4f][sca=%.4f][sat=%.4f] x:%+.3f(%+.3f) y:%+.3f(%+.3f) z:%+.3f(%+.3f)', e, self.vel_scale, self.vel_sat, \
           twist.twist.linear.x, t.twist.linear.x, \
           twist.twist.linear.y, t.twist.linear.y, \
           twist.twist.linear.z, t.twist.linear.z)
        rospy.loginfo('[angular] x:%+.3f(%+.3f) y:%+.3f(%+.3f) z:%+.3f(%+.3f)', \
           twist.twist.angular.x, t.twist.angular.x, \
           twist.twist.angular.y, t.twist.angular.y, \
           twist.twist.angular.z, t.twist.angular.z)
      except rospy.ServiceException, e:
        if which_arm == 'l':
          self.pn.stop_moving_vel('l')
        else:
          self.pn.stop_moving_vel('r')

      return VisualServoTwistResponse()

    def handle_init_request(self, req):
      self.pn.init_arms()

if __name__ == '__main__':
  try:
    node = ArmController()
    rospy.loginfo('Done initializing... Now advertise the Service')
    rospy.Service('vs_pose', VisualServoPose , node.handle_pose_request)
    rospy.Service('vs_twist', VisualServoTwist , node.handle_twist_request)
    rospy.Service('vs_init', Empty , node.handle_init_request)
    rospy.loginfo('Ready to move arm')
    rospy.spin()

  except rospy.ROSInterruptException: pass
