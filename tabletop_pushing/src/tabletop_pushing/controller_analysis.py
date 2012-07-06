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
from geometry_msgs.msg import Point
from math import sin, cos, pi, sqrt, fabs
import cv2
import tf.transformations as tr
import numpy as np
from tabletop_pushing.srv import *
from tabletop_pushing.msg import *
import sys
import rospy

_HEADER_LINE = '# c_x c_y c_z theta push_opt arm c_x\' c_y\' c_z\' push_dist high_init push_time'

class ControlState:
    def __init__(self):
        self.x = None
        self.x_dot = None
        self.x_desired = None
        self.theta0 = None
        self.u = None
        self.u_g = None
        self.u_spin = None
        self.k_g = None
        self.k_s_d = None
        self.k_s_p = None

    def __str__(self):
        return str((self.x, self.x_dot, self.x_desired, self.theta0, self.u, self.u_g, self.u_spin,
                    self.k_g, self.k_s_d, self.k_s_p))

class ControlAnalysisIO:
    def __init__(self):
        self.data_out = None
        self.data_in = None

    def write_line(self, x, x_dot, x_desired, theta0, u, u_g, u_spin, k_g, k_s_d, k_s_p):
        if self.data_out is None:
            rospy.logerr('Attempting to write to file that has not been opened.')
            return
        data_line = str(x)+' '+str(x_dot)+' '+str(x_desired)+' '+str(theta0)+' '+str(u)+' '+\
            str(u_g)+' '+str(u_spin)+' '+str(k_g)+' '+str(k_s_d)+' '+str(k_s_p)+'\n'
        self.data_out.write(data_line)
        self.data_out.flush()

    def parse_line(self, line):
        # TODO: Read in the stuff and display / analyze it
        pass

    def read_in_data_file(self, file_name):
        data_in = file(file_name, 'r')
        x = [self.parse_line(l) for l in data_in.readlines()]
        data_in.close()
        return filter(None, x)

    def open_out_file(self, file_name):
        self.data_out = file(file_name, 'a')
        self.data_out.write(_HEADER_LINE+'\n')
        self.data_out.flush()

    def close_out_file(self):
        self.data_out.close()
