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
from math import copysign, pi, sqrt, isnan
from numpy import finfo
import svmutil
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plotter
import sys
import push_learning
import push_trajectory_generator as ptg
from model_based_pushing_control import *

def test_svm_stuff():
    aff_file_name  = sys.argv[1]
    delta_t = 1./9.
    n = 5
    m = 2

    plio = push_learning.CombinedPushLearnControlIO()
    plio.read_in_data_file(aff_file_name)

    svm_dynamics = SVRPushDynamics(delta_t, n, m)
    svm_dynamics.learn_model(plio.push_trials)
    base_path = '/u/thermans/data/svm_dyn/'
    output_paths = []
    output_paths.append(base_path+'delta_x_dyn.model')
    output_paths.append(base_path+'delta_y_dyn.model')
    output_paths.append(base_path+'delta_theta_dyn.model')
    svm_dynamics.save_models(output_paths)

    svm_dynamics2 = SVRPushDynamics(delta_t, n, m, svm_file_names=output_paths)

    test_pose = VisFeedbackPushTrackingFeedback()
    test_pose.x.x = 0.2
    test_pose.x.y = 0.0
    test_pose.x.theta = pi*0.5
    test_ee = PoseStamped()
    test_ee.pose.position.x = test_pose.x.x - 0.2
    test_ee.pose.position.y = test_pose.x.y - 0.2
    test_u = TwistStamped()
    test_u.twist.linear.x = 0.3
    test_u.twist.linear.y = 0.3

    next_state = svm_dynamics2.predict_state(test_pose, test_ee, test_u)

    print 'test_state.x: ', test_pose.x
    print 'next_state.x: ', next_state.x

def test_mpc():
    delta_t = 2.0
    H = 10
    n = 5
    m = 2
    u_max = 0.5
    sigma = 0.01
    # plot_output_path = '/home/thermans/sandbox/mpc_plots/'
    plot_output_path = ''
    xtra = []
    plot_all_t = False
    plot_gt = True
    test_trajectory = False

    # print 'H = ', H
    # print 'delta_t = ', delta_t
    # print 'u_max = ', u_max
    # print 'max displacement = ', delta_t*u_max
    # print 'Total max displacement = ', delta_t*u_max*H
    # print 'x_d = ', np.array(x_d)

    cur_state = VisFeedbackPushTrackingFeedback()
    cur_state.x.x = 0.0
    cur_state.x.y = 0.0
    cur_state.x.theta = pi*0.5
    ee_pose = PoseStamped()
    ee_pose.pose.position.x = cur_state.x.x - 0.2
    ee_pose.pose.position.y = cur_state.x.y - 0.2
    cur_u = TwistStamped()
    cur_u.twist.linear.x = u_max
    cur_u.twist.linear.y = 0.0

    x0 = np.array([cur_state.x.x, cur_state.x.y, cur_state.x.theta,
                   ee_pose.pose.position.x, ee_pose.pose.position.y])

    goal_loc = Pose2D()
    goal_loc.x = 2.0
    goal_loc.y = 0.0

    p1 = Pose2D()
    p1.x = 0.5
    p1.y = 0.5

    p2 = Pose2D()
    p2.x = 1.0
    p2.y = -0.25

    p4 = Pose2D()
    p4.x = 2.5
    p4.y = 0.0

    pose_list = [p1, p2, goal_loc, p4]
    # trajectory_generator = ptg.PiecewiseLinearTrajectoryGenerator()
    trajectory_generator = ptg.ViaPointTrajectoryGenerator()
    x_d = trajectory_generator.generate_trajectory(H*2, cur_state.x, pose_list)

    if test_trajectory:
        for i in xrange(len(x_d)):
            print 'x_d[',i,'] =', x_d[i]
        q_fake = np.zeros((n+m)*H)

        plot_desired_vs_controlled(q_fake, x_d, x0, n, m, show_plot=True, suffix='-piecewise-linear',
                                   out_path=plot_output_path)

    # TODO: Test with a more complicated dynamics model
    dyn_model = NaiveInputDynamics(delta_t, n, m)

    # TODO: Improve the way noise is added to make this better
    sim_model = StochasticNaiveInputDynamics(delta_t, n, m, sigma)

    mpc =  ModelPredictiveController(dyn_model, H, u_max, delta_t)

    q_gt = []
    q_stars = []
    u_gt = []
    for i in xrange(len(x_d)-1):
        # Update desired trajectory
        x_d_i = x_d[i:]
        mpc.H = min(mpc.H, len(x_d_i)-1)
        mpc.regenerate_bounds()

        # Compute optimal control
        x_i = [cur_state.x.x, cur_state.x.y, cur_state.x.theta, ee_pose.pose.position.x, ee_pose.pose.position.y]
        q_star = mpc.feedbackControl(x_i, x_d_i, xtra)
        mpc.init_from_previous = True

        # Convert q_star to correct form for prediction
        u_i = [q_star[0], q_star[1]]

        # Plot performance so far
        q_cur = q_gt[:]
        q_cur.extend(q_star)
        q_cur = np.array(q_cur)
        if plot_all_t:
            plot_controls(q_cur, x_d, x0, n, m, u_max, show_plot=False, suffix='-q*['+str(i)+']', t=i,
                          out_path=plot_output_path)

            plot_desired_vs_controlled(q_cur, x_d, x0, n, m, show_plot=False, suffix='-q*['+str(i)+']', t=i,
                                       out_path=plot_output_path)

        # Generate next start point based on simulation model
        y_i = sim_model.predict(x_i, u_i)

        # Store for evaluation later
        q_gt.extend(u_i)
        q_gt.extend(y_i)
        q_stars.append(q_star)
        u_gt.extend(q_star[:2])

        # Convert result to form for input at next time step
        cur_state.x.x = y_i[0]
        cur_state.x.y = y_i[1]
        cur_state.x.theta = y_i[2]
        ee_pose.pose.position.x = y_i[3]
        ee_pose.pose.position.y = y_i[4]

    q_gt = np.array(q_gt)
    u_gt = np.array(u_gt)
    u_mean = np.mean(u_gt)
    print 'Control input SNR = ', u_mean/sigma

    # Plot final ground truth trajectory
    if plot_gt:
        plot_controls(q_gt, x_d, x0, n, m, u_max, show_plot=False, suffix='-q*['+str(len(x_d)-1)+']', t=len(x_d),
                      out_path=plot_output_path)
        plot_desired_vs_controlled(q_gt, x_d, x0, n, m, show_plot=True, suffix='-q*['+str(len(x_d)-1)+']',
                                   t=len(x_d), out_path=plot_output_path)

    # Plot initial guess trajectory
    # mpc.H = 10
    # U_init = mpc.get_U_init(x0, x_d)
    # q0 = mpc.get_q0(x0, U_init, xtra)
    # plot_desired_vs_controlled(q0, x_d, x0, n, m, show_plot=True, suffix='-q0', opt_path=plot_output_path)

if __name__ == '__main__':
    test_svm_stuff()
    # test_mpc()
