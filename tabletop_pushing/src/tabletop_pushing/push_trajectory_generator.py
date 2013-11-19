#!/usr/bin/env python
# Software License Agreement (BSD License)
#
#  Copyright (c) 2013, Georgia Institute of Technology
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
from geometry_msgs.msg import PoseStamped, TwistStamped, Twist, Pose2D
from tabletop_pushing.srv import *
from tabletop_pushing.msg import *
from math import copysign, pi
import svmutil
import numpy as np

class StraightLineTrajectoryGenerator:
    def generate_trajectory(self, H, start_pose, end_pose):
        start_loc = np.asarray([start_pose.x, start_pose.y, start_pose.theta])
        end_loc = np.asarray([end_pose.x, end_pose.y, end_pose.theta])
        step = np.asarray([(end_pose.x - start_pose.x)/H, (end_pose.y - start_pose.y)/H, 0.0])
        trajectory = [start_loc]
        for i in xrange(H):
            next_loc = trajectory[i] + step
            trajectory.append(next_loc)
        return trajectory

class ArcTrajectoryGenerator:
    def generate_trajectory(self, H, start_pose, end_pose, arc_width=0.25):
        start_loc = np.array([start_pose.x, start_pose.y, start_pose.theta])
        end_loc = np.array([end_pose.x, end_pose.y, end_pose.theta])
        mid_loc = 0.5*(start_loc+end_loc)
        mid_loc[1] += arc_width

        print 'start_loc =', start_loc
        print 'mid_loc =',mid_loc
        print 'end_loc =',end_loc

        # Generate first half
        step = (mid_loc-start_loc)/(0.5*H)
        print 'step =',step
        trajectory = [start_loc]
        for i in xrange(H/2):
            next_loc = trajectory[i] + step
            trajectory.append(next_loc)
        # Generate second half
        step = (end_loc-mid_loc)/(0.5*H)
        print 'step =',step
        for i in range(H/2,H):
            next_loc = trajectory[i] + step
            trajectory.append(next_loc)
        return trajectory
