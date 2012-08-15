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
from geometry_msgs.msg import Pose2D, Twist
import sys
import rospy
from pylab import *
from math import cos, sin

_HEADER_LINE = '# x.x x.y x.theta x_dot.x x_dot.y x_dot.theta x_desired.x x_desired.y x_desired.theta theta0 u.linear.x u.linear.y u.linear.z u.angular.x u.angular.y u.angular.z time'

class ControlTimeStep:
    def __init__(self, x, x_dot, x_desired, theta0, u, t):
        self.x = x
        self.x_dot = x_dot
        self.x_desired = x_desired
        self.theta0 = theta0
        self.u = u
        self.t = t

class ControlAnalysisIO:
    def __init__(self):
        self.data_out = None
        self.data_in = None

    def write_line(self, x, x_dot, x_desired, theta0, u, time):
        if self.data_out is None:
            rospy.logerr('Attempting to write to file that has not been opened.')
            return
        data_line = str(x.x)+' '+str(x.y)+' '+str(x.theta)+' '+\
            str(x_dot.x)+' '+str(x_dot.y)+' '+str(x_dot.theta)+' '+\
            str(x_desired.x)+' '+str(x_desired.y)+' '+str(x_desired.theta)+' '+\
            str(theta0)+' '+str(u.linear.x)+' '+str(u.linear.y)+' '+str(u.linear.z)+' '+\
            str(u.angular.x)+' '+str(u.angular.y)+' '+str(u.angular.z)+' '+str(time)+'\n'
        self.data_out.write(data_line)
        self.data_out.flush()

    def parse_line(self, line):
        if line.startswith('#'):
            return None
        data = [float(s) for s in line.split()]
        x = Pose2D()
        x_dot = Pose2D()
        x_desired = Pose2D()
        u = Twist()
        x.x = data[0]
        x.y = data[1]
        x.theta = data[2]
        x_dot.x = data[3]
        x_dot.y = data[4]
        x_dot.theta = data[5]
        x_desired.x = data[6]
        x_desired.y = data[7]
        x_desired.theta = data[8]
        theta0 = data[9]
        u.linear.x = data[10]
        u.linear.y = data[11]
        u.linear.z = data[12]
        u.angular.x = data[13]
        u.angular.y = data[14]
        u.angular.z = data[15]
        t = data[16]
        cts = ControlTimeStep(x, x_dot, x_desired, theta0, u, t)
        return cts

    def read_in_data_file(self, file_name):
        data_in = file(file_name, 'r')
        x = [self.parse_line(l) for l in data_in.readlines()[1:]]
        data_in.close()
        return filter(None, x)

    def open_out_file(self, file_name):
        self.data_out = file(file_name, 'a')
        self.data_out.write(_HEADER_LINE+'\n')
        self.data_out.flush()

    def close_out_file(self):
        self.data_out.close()


def plot_results(file_name, spin=False):
    io = ControlAnalysisIO()
    controls = io.read_in_data_file(file_name)
    XS = [c.x.x for c in controls]
    YS = [c.x.y for c in controls]
    Thetas = [c.x.y for c in controls]
    init_x = XS[0]
    init_y = YS[0]
    goal_x = controls[0].x_desired.x
    goal_y = controls[0].x_desired.y
    goal_theta = controls[0].x_desired.theta
    figure()
    plot(XS,YS)
    scatter(XS,YS)
    ax = gca()
    print ylim()
    print ax.set_xlim(xlim()[1]-(ylim()[1]-ylim()[0]), xlim()[1])
    headings = [Arrow(c.x.x, c.x.y, cos(c.x.theta)*0.01, sin(c.x.theta)*0.01, 0.05, axes=ax) for c in controls]
    arrows = [ax.add_patch(h) for h in headings]
    init_arrow = Arrow(controls[0].x.x, controls[0].x.y,
                       cos(controls[0].x.theta)*0.01,
                       sin(controls[0].x.theta)*0.01, 0.05,
                       axes=ax, facecolor='red')
    goal_arrow = Arrow(controls[0].x_desired.x, controls[0].x_desired.y,
                       cos(controls[0].x_desired.theta)*0.01,
                       sin(controls[0].x_desired.theta)*0.01, 0.05,
                       axes=ax, facecolor='green')
    ax.add_patch(goal_arrow)
    ax.add_patch(init_arrow)
    scatter(init_x, init_y, c='r')
    scatter(goal_x, goal_y, c='g')
    xlabel('x (meters)')
    ylabel('y (meters)')
    title('Tracker Ouput')
    savefig('/home/thermans/sandbox/tracker-output.png')
    figure()
    ux = [c.u.linear.x for c in controls]
    uy = [c.u.linear.y for c in controls]
    if spin:
        uy = [c.u.linear.z for c in controls]
    plot(ux)
    plot(uy)
    xlabel('Time')
    ylabel('Velocity (m/s)')
    legend(['u_x', 'u_y'])
    title('Feedback Controller - Input Velocities')
    savefig('/home/thermans/sandbox/feedback-input.png')
    figure()
    xdots = [c.x_dot.x for c in controls]
    ydots = [c.x_dot.y for c in controls]
    plot(xdots)
    plot(ydots)
    xlabel('Time')
    ylabel('Velocity (m/s)')
    title('Tracker Velocities')
    legend(['x_dot', 'y_dot'])
    savefig('/home/thermans/sandbox/tracker-velocities.png')
    figure()
    thetadots = [c.x_dot.theta for c in controls]
    plot(thetadots,c='r')
    xlabel('Time')
    ylabel('Velocity (rad/s)')
    title('Tracker Velocities')
    legend(['theta_dot'])
    savefig('/home/thermans/sandbox/tracker-theta-vel.png')
    figure()
    x_err = [c.x_desired.x - c.x.x for c in controls]
    y_err = [c.x_desired.y - c.x.y for c in controls]
    plot(x_err)
    plot(y_err)
    xlabel('Time')
    ylabel('Error (meters)')
    title('Position Error')
    savefig('/home/thermans/sandbox/pos-err.png')
    figure()
    if spin:
        theta_err = [c.x_desired.theta - c.x.theta for c in controls]
    else:
        theta_err = [c.theta0 - c.x.theta for c in controls]
    plot(theta_err, c='r')
    xlabel('Time')
    ylabel('Error (radians)')
    if spin:
        title('Error in Orientation')
    else:
        title('Heading Deviation from Initial')
    savefig('/home/thermans/sandbox/theta-err.png')
    show()

if __name__ == '__main__':
    plot_results(sys.argv[1])
