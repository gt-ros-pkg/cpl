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

import roslib; roslib.load_manifest('visual_servo')
import rospy
import tabletop_pushing.position_push_node as pn
from geometry_msgs.msg import TwistStamped
import tf
from visual_servo.srv import *
import sys

class VNode:

    def __init__(self):
        rospy.init_node('v_node', log_level=rospy.DEBUG)
        
        # Setup parameters
        self.vel_sat = rospy.get_param('~vel_sat', 0.10)
        
        # Initialize vel controller
        self.pn = pn.PositionFeedbackPushNode()
        self.pub = self.pn.r_arm_cart_vel_pub
        
        rospy.loginfo('Waiting for Visual Servo Node Service')
        rospy.wait_for_service('visual_servo_twist')
        self.srv = rospy.ServiceProxy('visual_servo_twist', VisualServoTwist)


    # Main Control Loop
    #
    def run(self):
        '''
        Main control loop for the node
        '''
        while not rospy.is_shutdown():
          try:
            twist = TwistStamped()
            twist.header.stamp = rospy.Time(0)
            twist.header.frame_id = 'torso_lift_link'

            resp = self.srv()
            twist.twist.linear.x = self.adjust_velocity(resp.vx)
            twist.twist.linear.y = self.adjust_velocity(resp.vy)
            twist.twist.linear.z = self.adjust_velocity(resp.vz)
            twist.twist.angular.x = self.adjust_velocity(resp.wx)
            twist.twist.angular.y = self.adjust_velocity(resp.wy)
            twist.twist.angular.z = self.adjust_velocity(resp.wz)
            self.pub.publish(twist)
            rospy.sleep(0.2) 
          except rospy.ServiceException, e:
            self.pn.stop_moving_vel('r')
            rospy.sleep(0.2) 
  
    # util
    #
    def adjust_velocity(self, vel):
      ret = vel 
      if ret > self.vel_sat:
        ret = self.vel_sat
      elif -ret > self.vel_sat:
        ret = -self.vel_sat
      return -ret 


if __name__ == '__main__':
    node = VNode()
    node.run()
