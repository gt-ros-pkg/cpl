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
from geometry_msgs.msg import PoseStamped, TwistStamped, Pose2D
from tabletop_pushing.srv import *
from tabletop_pushing.msg import *
from math import copysign, pi, sqrt, isnan
from numpy import finfo
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plotter
import sys
import push_learning
import push_trajectory_generator as ptg
from model_based_pushing_control import *
from pushing_dynamics_models import *
import dynamics_learning
import subprocess
import os
from tf.transformations import euler_from_quaternion, quaternion_from_euler

_KULER_RED = (178./255, 18./255, 18./255)
_KULER_YELLOW = (1., 252./255, 25./255)
_KULER_GREEN = (0., 178./255, 51./255)
_KULER_BLUE = (20./255, 133./255, 204./255)
_KULER_RED1 = (1., 0., 0.)
_KULER_BLUE1 = (9./255, 113./255, 178./255)
_KULER_GREEN1 = (0., 255./255, 72./255)

class MPCSolutionIO:
    def __init__(self):
        self.out_file = None
        self.out_lines = []

    def open_out_file(self, file_name):
        self.out_file = file(file_name, 'a')

    def close_out_file(self):
        if self.out_file is not None:
            self.out_file.close()

    def buffer_line(self, k0, q):
        out_line = self.generate_line(k0, q)
        self.out_lines.append(out_line)

    def write_line(self, k0, q):
        out_line = self.generate_line(k0, q)
        self.out_file.write(out_line)

    def write_buffer_to_disk(self):
        for line in self.out_lines:
            self.out_file.write(line)
        self.out_file.flush()
        self.out_lines = []

    def generate_line(self, k0, q):
        out_line = str(k0) + ' ' + str(q)
        out_line += '\n'
        return out_line

    def parse_line(self, line):
        elems = line.split()
        k0 = int(elems[0])
        q = [float(x.lstrip('[').rstrip(',').rstrip(']')) for x in elems[1:]]
        return (k0, q)

    def read_file(self, file_name):
        in_file = file(file_name, 'r')
        data = [self.parse_line(l) for l in in_file.readlines()]
        in_file.close()
        return data

class PushTrajectoryIO:
    def __init__(self):
        self.out_file = None
        self.out_lines = []

    def open_out_file(self, file_name):
        self.out_file = file(file_name, 'a')

    def close_out_file(self):
        if self.out_file is not None:
            self.out_file.close()

    def buffer_line(self, k0, trajectory):
        out_line = self.generate_line(k0, trajectory)
        self.out_lines.append(out_line)

    def write_line(self, k0, trajectory):
        out_line = self.generate_line(k0, trajectory)
        self.out_file.write(out_line)

    def write_buffer_to_disk(self):
        for line in self.out_lines:
            self.out_file.write(line)
        self.out_file.flush()
        self.out_lines = []

    def generate_line(self, k0, trajectory):
        out_line = str(k0)
        for p in trajectory:
            out_line += ' ' + str(p)
        out_line += '\n'
        return out_line

    def parse_line(self, line):
        l0 = line.split()[0]
        k0 = int(l0)
        line = line[len(l0)+1:]
        elems = line.split('[')
        traj = []
        for e in elems[1:]:
            traj.append(np.array([float(g) for g in e.rstrip().rstrip(']').split() ]))
        return (k0, traj)

    def read_file(self, file_name):
        in_file = file(file_name, 'r')
        data = [self.parse_line(l) for l in in_file.readlines()]
        in_file.close()
        return data

def plot_desired_vs_controlled_with_history(trial_traj, q_planned, X_d, n, m,
                                            show_plot = True, suffix = '', out_path = ''):
    # Convert trials into decision vector for use with the base function
    x0_state = trial_traj[0]
    x0 = np.array([x0_state.x.x, x0_state.x.y, x0_state.x.theta,
                   x0_state.ee.position.x, x0_state.ee.position.y])

    q_star = []
    for i, state in enumerate(trial_traj):
        if i > 0:
            q_star.extend([state.u.linear.x, state.u.linear.y])
            q_star.extend([state.x.x, state.x.y, state.x.theta,
                           state.ee.position.x, state.ee.position.y])
    q_star.extend(q_planned)

    # Where to start past vs future
    t = len(trial_traj)-1
    plot_desired_vs_controlled(q_star, X_d, x0, n, m, show_plot, suffix, t, out_path)

def plot_desired_vs_controlled(q_star, X_d, x0, n, m, show_plot=True, suffix = '', t = 0, out_path = ''):
    H = len(q_star)/(n+m)
    X,U =  get_x_u_from_q(q_star, x0, H, n, m)

    plotter.figure()

    plan_color = _KULER_RED
    gt_color = _KULER_GREEN
    predicted_color = _KULER_BLUE
    ee_predicted_color = _KULER_BLUE1
    gt_ee_color = _KULER_GREEN1

    # Plot desired
    x_d = [X_d_k[0] for X_d_k in X_d]
    y_d = [X_d_k[1] for X_d_k in X_d]
    theta_d = [X_d_k[2] for X_d_k in X_d]
    plotter.plot(x_d, y_d, color = plan_color)
    plotter.plot(x_d, y_d, color = plan_color, marker = 'o')

    # Plot predicted
    x_hat = [X_k[0] for X_k in X[t:]]
    y_hat = [X_k[1] for X_k in X[t:]]
    theta_hat = [X_k[2] for X_k in X[t:]]
    ee_x_hat = [X_k[3] for X_k in X[t:]]
    ee_y_hat = [X_k[4] for X_k in X[t:]]
    plotter.plot(x_hat, y_hat, color = predicted_color)
    plotter.plot(x_hat, y_hat, color = predicted_color, marker = '+')
    # plotter.plot(ee_x_hat, ee_y_hat, color = ee_predicted_color)
    # plotter.plot(ee_x_hat, ee_y_hat, color = ee_predicted_color, marker = '+')

    # Plot observed / GT
    x_gt = [X_k[0] for X_k in X[:t+1]]
    y_gt = [X_k[1] for X_k in X[:t+1]]
    theta_gt = [X_k[2] for X_k in X[:t+1]]
    ee_x_gt = [X_k[3] for X_k in X[:t+1]]
    ee_y_gt = [X_k[4] for X_k in X[:t+1]]
    plotter.plot(x_gt, y_gt, color = gt_color)
    plotter.plot(x_gt, y_gt, color = gt_color, marker = '+')
    plotter.plot(ee_x_gt, ee_y_gt, color = gt_ee_color)
    plotter.plot(ee_x_gt, ee_y_gt, color = gt_ee_color, marker ='x')

    # Make axes equal scales
    xlim_auto = plotter.xlim()
    ylim_auto = plotter.ylim()
    xlim_auto_range = xlim_auto[1] - xlim_auto[0]
    ylim_auto_range = ylim_auto[1] - ylim_auto[0]
    custom_range = max(xlim_auto_range, ylim_auto_range)
    xlim_custom = (xlim_auto[0], xlim_auto[0] + custom_range)
    ylim_custom = (ylim_auto[0], ylim_auto[0] + custom_range)
    plotter.xlim(xlim_custom)
    plotter.ylim(ylim_custom)

    # Write stuff
    plot_title = 'MPC_Push_Control_Trajectory'+suffix
    plotter.title(plot_title)
    plotter.xlabel('x (meters)')
    plotter.ylabel('y (meters)')

    if len(out_path) > 0:
        plotter.savefig(out_path+plot_title+'.png')
    if show_plot:
        plotter.show()

def plot_all_planned_trajectories(trajs, trials, show_plot=True, suffix='', out_path='', show_headings=False):
    trajectories_segmented = []
    for i, traj in enumerate(trajs):
        if traj[0] == 0:
            if i > 0:
                trajectories_segmented.append(segment)
            segment = []
        segment.append(traj[1])
    trajectories_segmented.append(segment)

    plan_color = _KULER_RED
    gt_color = _KULER_GREEN

    for i, (trial, plans) in enumerate(zip(trials, trajectories_segmented)):
        plotter.figure()

        for X_d in plans:
            # Plot desired
            x_d = [X_d_k[0] for X_d_k in X_d]
            y_d = [X_d_k[1] for X_d_k in X_d]
            theta_d = [X_d_k[2] for X_d_k in X_d]
            plotter.plot(x_d, y_d, color=plan_color, ls='--')

        x_gt = [state.x.x for state in trial.trial_trajectory]
        y_gt = [state.x.y for state in trial.trial_trajectory]
        theta_gt = [state.x.theta for state in trial.trial_trajectory]
        plotter.plot(x_gt, y_gt, c=gt_color, ls='-')
        if show_headings:
            ax = plotter.gca()
            headings = [plotter.Arrow(c.x.x, c.x.y, cos(c.x.theta)*0.005,
                                      sin(c.x.theta)*0.005, 0.005, axes=ax,
                                      color=gt_color) for c in trial.trial_trajectory]
            arrows = [ax.add_patch(h) for h in headings]
        else:
            plotter.plot(x_gt, y_gt, c=gt_color, marker='o')
        # Make axes equal scales
        xlim_auto = plotter.xlim()
        ylim_auto = plotter.ylim()
        xlim_auto_range = xlim_auto[1] - xlim_auto[0]
        ylim_auto_range = ylim_auto[1] - ylim_auto[0]
        custom_range = max(xlim_auto_range, ylim_auto_range)
        xlim_custom = (xlim_auto[0], xlim_auto[0] + custom_range)
        ylim_custom = (ylim_auto[0], ylim_auto[0] + custom_range)
        plotter.xlim(xlim_custom)
        plotter.ylim(ylim_custom)
        # Write stuff
        plot_title = 'Planned_Trajectories_'+str(i)+suffix
        plotter.title(plot_title)
        plotter.xlabel('x (meters)')
        plotter.ylabel('y (meters)')

        if len(out_path) > 0:
            plotter.savefig(out_path+plot_title+'.png')

    if show_plot:
        plotter.show()

def plot_tracks(trials, show_plot = False, suffix='', out_path=''):
    for i, trial in enumerate(trials):
        suffix_i = '_'+str(i)+suffix
        plot_track(trial, show_plot = False, suffix=suffix_i, out_path=out_path)
    if show_plot:
        plotter.show()

def plot_track(trial, show_plot = False, suffix='', out_path=''):
    X = [state.x.x for state in trial.trial_trajectory]
    Y = [state.x.y for state in trial.trial_trajectory]
    Theta = [state.x.theta for state in trial.trial_trajectory]

    x_color = _KULER_RED
    y_color = _KULER_BLUE
    theta_color = _KULER_GREEN

    plot_title = 'Object Pose vs Time' + suffix
    x_plot_title = 'Object x-Position vs Time' + suffix
    y_plot_title = 'Object y-Position vs Time' + suffix
    theta_plot_title = 'Object Orientation vs Time' + suffix

    xlabel = 'Time Step'
    loc_ylabel = 'Location (m)'
    theta_ylabel = 'Orientation (rad)'

    xloc_ylim = (0, 1.0)
    yloc_ylim = (-0.5, 0.5)
    theta_ylim = (-pi, pi)

    series_ylabel = 'Location (m) & Orientation (rad)'
    legend = ['x', 'y', 'theta']
    plot_time_series(X, x_color, xlabel = xlabel, ylabel = series_ylabel)
    plot_time_series(Y, y_color, new_fig=False, marker = '+')
    plot_time_series(Theta, theta_color, out_path = out_path, ylim = theta_ylim, new_fig=False,
                     legend = legend, marker='x', plot_title = plot_title)

    plot_time_series(X, x_color, x_plot_title, xlabel, loc_ylabel, out_path, xloc_ylim)
    plot_time_series(Y, y_color, y_plot_title, xlabel, loc_ylabel, out_path, yloc_ylim)
    plot_time_series(Theta, theta_color, theta_plot_title, xlabel, theta_ylabel, out_path, theta_ylim)

    if show_plot:
        plotter.show()

def plot_time_series(data, color, plot_title=None, xlabel=None, ylabel=None, out_path='', ylim=None,
                     new_fig=True, legend=None, marker='o'):
    if new_fig:
        plotter.figure()

    plotter.plot(data, color = color, label='_nolegend_')
    plotter.plot(data, color = color, marker = marker)
    if len(data) > 0:
        plotter.xlim((0, len(data)-1))

    if plot_title is not None:
        plotter.title(plot_title)
    if xlabel is not None:
        plotter.xlabel(xlabel)
    if ylabel is not None:
        plotter.ylabel(ylabel)
    if ylim is not None:
        plotter.ylim(ylim)
    if legend is not None:
        plotter.legend(legend, loc = 0)

    if len(out_path) > 0 and plot_title is not None:
        plotter.savefig(out_path + plot_title + '.png')

def plot_all_controls(trials, u_max, show_plot=True, suffix='', out_path=''):
    for i, trial in enumerate(trials):
        Ux = [state.u.linear.x for state in trial.trial_trajectory]
        Uy = [state.u.linear.y for state in trial.trial_trajectory]
        suffix_i = '_' + str(i) + suffix
        plot_controls_base(Ux, Uy, u_max, show_plot = False, suffix = suffix_i, out_path = out_path)
    if show_plot:
        plotter.show()

def plot_controls(q_star, x0, n, m, u_max, show_plot=True, suffix='', out_path=''):
    H = len(q_star)/(n+m)
    X,U =  get_x_u_from_q(q_star, x0, H, n, m)

    Ux = [u_k[0] for u_k in U]
    Uy = [u_k[1] for u_k in U]
    plot_controls_base(Ux, Uy, u_max, show_plot, suffix, out_path)

def plot_controls_with_history(trial_traj, q_star, x0, n, m, u_max, show_plot=True, suffix='', out_path=''):
    Ux = [state.u.linear.x for state in trial_traj]
    Uy = [state.u.linear.y for state in trial_traj]

    H = len(q_star)/(n+m)
    X,U =  get_x_u_from_q(q_star, x0, H, n, m)

    Ux.extend([u_k[0] for u_k in U])
    Uy.extend([u_k[1] for u_k in U])
    plot_controls_base(Ux, Uy, u_max, show_plot, suffix, out_path, history_start=len(trial_traj)-0.5)

def plot_controls_base(Ux, Uy, u_max, show_plot=True, suffix='', out_path='', history_start=None):
    x_color = _KULER_RED
    y_color = _KULER_BLUE
    lim_color = _KULER_GREEN
    history_mark_color = (0.0, 0.0, 0.0)

    custom_ylim = (-1.1*u_max, 1.1*u_max)

    plotter.figure()
    # Plot a vertical dashed line showing where the history ends and prediction starts
    if history_start is not None:
        history_x = [history_start, history_start]
        history_y = custom_ylim
        plotter.plot(history_x, history_y, ls='--', color = history_mark_color)

    plotter.plot(np.ones(len(Ux))*u_max, color = lim_color)
    plotter.plot(np.ones(len(Ux))*-u_max, color = lim_color)

    plotter.plot(Ux, color = x_color)
    plotter.plot(Ux, color = x_color, marker = 'o')

    plotter.plot(Uy, color = y_color)
    plotter.plot(Uy, color = y_color, marker = '+')

    plot_title = 'MPC_Push_Control_Inputs'+suffix
    plotter.title(plot_title)
    plotter.ylim(custom_ylim)
    plotter.xlim([0, len(Ux)-1])
    plotter.xlabel('Time Step')
    plotter.ylabel('U (meters/sec)')
    if len(out_path) > 0:
        plotter.savefig(out_path+plot_title+'.png')
    if show_plot:
        plotter.show()

def plot_predicted_vs_observed_tracks(Y, Y_hat, X, suffix = '', out_path = '',
                                      show_plot = False):

    # Display parameters
    gt_color = _KULER_GREEN
    pred_color = _KULER_BLUE
    pred_cascade_color = _KULER_RED
    gt_marker = 'o'
    pred_marker = 'x'
    pred_cascade_marker = '+'

    xlabel = 'Time Step'
    loc_ylabel = 'Location (m)'
    orientation_ylabel = 'Orientation (rad)'

    Obj_plot_title = 'Object Location Predicted vs Observed' + suffix
    Theta_plot_title = 'Object Theta Predicted vs Observed' + suffix
    EE_plot_title = 'End Effector Location Predicted vs Observed' + suffix

    Obj_cascaded_plot_title = 'Object Location Predicted vs Observed (Cascaded)' + suffix
    Theta_cascaded_plot_title = 'Object Theta Predicted vs Observed (Cascaded)' + suffix
    EE_cascaded_plot_title = 'End Effector Location Predicted vs Observed (Cascaded)' + suffix

    # Data lists
    x_o_gt = []
    y_o_gt = []
    theta_o_gt = []
    x_ee_gt = []
    y_ee_gt = []

    x_o_pred_cascade = []
    y_o_pred_cascade = []
    theta_o_pred_cascade = []
    x_ee_pred_cascade = []
    y_ee_pred_cascade = []

    x_o_pred_fb = []
    y_o_pred_fb = []
    theta_o_pred_fb = []
    x_ee_pred_fb = []
    y_ee_pred_fb = []

    # Transform deltas into Xs and Ys
    # Get two sets of predicted: cascaded and feedback
    # NOTE: There is one more prediction / gt delta we could add to this...
    for i, x in enumerate(X):
        x_o = x[0]
        y_o = x[1]
        theta_o = x[2]
        x_ee = x[3]
        y_ee = x[4]

        # Append ground truth observations
        x_o_gt.append(x_o)
        y_o_gt.append(y_o)
        theta_o_gt.append(theta_o)
        x_ee_gt.append(x_ee)
        y_ee_gt.append(y_ee)

        if i == 0:
            # Initialize all tracks with gt initial
            x_o_pred_cascade.append(x_o)
            y_o_pred_cascade.append(y_o)
            theta_o_pred_cascade.append(theta_o)
            x_ee_pred_cascade.append(x_ee)
            y_ee_pred_cascade.append(y_ee)

            x_o_pred_fb.append(x_o)
            y_o_pred_fb.append(y_o)
            theta_o_pred_fb.append(theta_o)
            x_ee_pred_fb.append(x_ee)
            y_ee_pred_fb.append(y_ee)
            continue

        delta_x_o_hat = Y_hat[0][i]
        delta_y_o_hat = Y_hat[1][i]
        delta_theta_o_hat = Y_hat[2][i]
        delta_x_ee_hat = Y_hat[3][i]
        delta_y_ee_hat = Y_hat[4][i]

        # Integrate deltas with previous deltas
        x_o_pred_cascade.append(x_o_pred_cascade[-1] + delta_x_o_hat)
        y_o_pred_cascade.append(y_o_pred_cascade[-1] + delta_y_o_hat)
        theta_o_pred_cascade.append(theta_o_pred_cascade[-1] + delta_y_o_hat)
        x_ee_pred_cascade.append(x_ee_pred_cascade[-1] + delta_x_ee_hat)
        y_ee_pred_cascade.append(y_ee_pred_cascade[-1] + delta_y_ee_hat)

        # Add deltas to previous time step gt observations
        x_o_pred_fb.append(x_o_gt[-2] + delta_x_o_hat)
        y_o_pred_fb.append(y_o_gt[-2] + delta_y_o_hat)
        theta_o_pred_fb.append(theta_o_gt[-2] + delta_y_o_hat)
        x_ee_pred_fb.append(x_ee_gt[-2] + delta_x_ee_hat)
        y_ee_pred_fb.append(y_ee_gt[-2] + delta_y_ee_hat)

    # Plot observed vs predicted x,y trajectory for object
    legend = ['Ground Truth', 'Predicted FB', 'Pred Cascade']
    plotter.figure()
    plotter.plot(x_o_gt, y_o_gt, color = gt_color, label = '_nolegend_')
    plotter.plot(x_o_gt, y_o_gt, color = gt_color, marker = gt_marker)
    plotter.plot(x_o_pred_fb, y_o_pred_fb, color = pred_color, label = '_nolegend_')
    plotter.plot(x_o_pred_fb, y_o_pred_fb, color = pred_color, marker = pred_marker)
    plotter.plot(x_o_pred_cascade, y_o_pred_cascade, color = pred_cascade_color, label = '_nolegend_')
    plotter.plot(x_o_pred_cascade, y_o_pred_cascade, color = pred_cascade_color, marker = pred_cascade_marker)
    # Set scale on axes
    xlim_auto = plotter.xlim()
    ylim_auto = plotter.ylim()
    xlim_auto_range = xlim_auto[1] - xlim_auto[0]
    ylim_auto_range = ylim_auto[1] - ylim_auto[0]
    custom_range = max(xlim_auto_range, ylim_auto_range)
    xlim_custom = (xlim_auto[0], xlim_auto[0] + custom_range)
    ylim_custom = (ylim_auto[0], ylim_auto[0] + custom_range)
    plotter.xlim(xlim_custom)
    plotter.ylim(ylim_custom)
    # Write stuff
    plotter.title(Obj_plot_title)
    plotter.xlabel('x location (m)')
    plotter.xlabel('y location (m)')
    plotter.legend(legend, loc = 0)
    if len(out_path) > 0:
        plotter.savefig(out_path + Obj_plot_title + '.png')

    # Plot observed vs predicted x,y for end effector
    legend = ['Ground Truth', 'Predicted FB', 'Pred Cascade']
    plotter.figure()
    plotter.plot(x_ee_gt, y_ee_gt, color = gt_color, label = '_nolegend_')
    plotter.plot(x_ee_gt, y_ee_gt, color = gt_color, marker = gt_marker)
    plotter.plot(x_ee_pred_fb, y_ee_pred_fb, color = pred_color, label = '_nolegend_')
    plotter.plot(x_ee_pred_fb, y_ee_pred_fb, color = pred_color, marker = pred_marker)
    plotter.plot(x_ee_pred_cascade, y_ee_pred_cascade, color = pred_cascade_color, label = '_nolegend_')
    plotter.plot(x_ee_pred_cascade, y_ee_pred_cascade, color = pred_cascade_color, marker = pred_cascade_marker)
    # Make axes equal scales
    xlim_auto = plotter.xlim()
    ylim_auto = plotter.ylim()
    xlim_auto_range = xlim_auto[1] - xlim_auto[0]
    ylim_auto_range = ylim_auto[1] - ylim_auto[0]
    custom_range = max(xlim_auto_range, ylim_auto_range)
    xlim_custom = (xlim_auto[0], xlim_auto[0] + custom_range)
    ylim_custom = (ylim_auto[0], ylim_auto[0] + custom_range)
    plotter.xlim(xlim_custom)
    plotter.ylim(ylim_custom)
    # Write stuff
    plotter.title(EE_plot_title)
    plotter.xlabel('x location (m)')
    plotter.xlabel('y location (m)')
    plotter.legend(legend, loc = 0)
    if len(out_path) > 0:
        plotter.savefig(out_path + EE_plot_title + '.png')

    # Plot orientation vs time for object
    plot_time_series(theta_o_gt, gt_color, new_fig = True)
    plot_time_series(theta_o_pred_fb, pred_color, Theta_plot_title, xlabel, orientation_ylabel,
                     new_fig = False, marker = pred_marker)#, legend = legend, out_path = out_path)
    # plot_time_series(theta_o_gt, gt_color, new_fig = True)
    plot_time_series(theta_o_pred_cascade, pred_cascade_color, Theta_plot_title, xlabel, orientation_ylabel,
                     out_path = out_path, new_fig = False, legend = legend, marker = pred_cascade_marker)

    if show_plot:
        plotter.show()

def plot_predicted_vs_observed_deltas(gt_all, pred_all, suffix = '', out_path = '', show_plot = False):
    X_gt = gt_all[0]
    Y_gt = gt_all[1]
    Theta_gt = gt_all[2]
    if len(gt_all) > 3:
        Xee_gt = gt_all[3]
        Yee_gt = gt_all[4]
    else:
        Xee_gt = []
        Yee_gt = []

    X_pred = pred_all[0]
    Y_pred = pred_all[1]
    Theta_pred = pred_all[2]
    if len(pred_all) > 3:
        Xee_pred = pred_all[3]
        Yee_pred = pred_all[4]
    else:
        Xee_pred = []
        Yee_pred = []

    # Display parameters
    gt_color = _KULER_GREEN
    pred_color = _KULER_BLUE
    xlabel = 'Time Step'
    loc_ylabel = 'Distance (m)'
    orientation_ylabel = 'Rotation (rad)'
    X_plot_title = 'Object Delta X Predicted vs Observed' + suffix
    Y_plot_title = 'Object Delta Y Predicted vs Observed' + suffix
    Theta_plot_title = 'Object Delta Theta Predicted vs Observed' + suffix
    Xee_plot_title = 'End Effector Delta X Predicted vs Observed' + suffix
    Yee_plot_title = 'End Effector Delta Y Location Predicted vs Observed' + suffix

    # Plot observed vs predicted for all outputs individually
    legend = ['Ground Truth', 'Predicted']
    pred_marker = 'x'
    plot_time_series(X_gt, gt_color, new_fig = True)
    plot_time_series(X_pred, pred_color, X_plot_title, xlabel, loc_ylabel,
                     out_path = out_path, new_fig = False, legend = legend, marker=pred_marker)

    plot_time_series(Y_gt, gt_color, new_fig = True)
    plot_time_series(Y_pred, pred_color, Y_plot_title, xlabel, loc_ylabel,
                     out_path = out_path, new_fig = False, legend = legend, marker=pred_marker)

    plot_time_series(Theta_gt, gt_color, new_fig = True)
    plot_time_series(Theta_pred, pred_color, Theta_plot_title, xlabel, orientation_ylabel,
                     out_path = out_path, new_fig = False, legend = legend, marker=pred_marker)

    plot_time_series(Xee_gt, gt_color, new_fig = True)
    plot_time_series(Xee_pred, pred_color, Xee_plot_title, xlabel, loc_ylabel,
                     out_path = out_path, new_fig = False, legend = legend, marker=pred_marker)

    plot_time_series(Yee_gt, gt_color, new_fig = True)
    plot_time_series(Yee_pred, pred_color, Yee_plot_title, xlabel, loc_ylabel,
                     out_path = out_path, new_fig = False, legend = legend, marker=pred_marker)

    if show_plot:
        plotter.show()

def test_svm_stuff(aff_file_name=None):
    delta_t = 1./9.
    n = 5
    m = 2
    use_obj_frame = False
    base_path = '/u/thermans/src/gt-ros-pkg/cpl/tabletop_pushing/cfg/SVR_DYN/'
    output_paths = []
    epsilons = [1e-5, 1e-5, 1e-5]
    plio = push_learning.CombinedPushLearnControlIO()
    if aff_file_name is not None:
        base_file_string = base_path + 'test_svm_stuff'
        plio.read_in_data_file(aff_file_name)

        svm_dynamics = SVRPushDynamics(delta_t, n, m, object_frame_feats=use_obj_frame, epsilons=epsilons)
        svm_dynamics.learn_model(plio.push_trials)
        base_file_string = svm_dynamics.save_models(base_file_string)
    else:
        base_file_string = '/u/thermans/src/gt-ros-pkg/cpl/tabletop_pushing/cfg/SVR_DYN/test_svm_stuff_LINEAR_linearEE_worldFrame'

    svm_dynamics2 = SVRPushDynamics(delta_t, n, m, base_file_string)

    test_pose = VisFeedbackPushTrackingFeedback()
    test_pose.x.x = 0.2
    test_pose.x.y = 0.0
    test_pose.x.theta = 0#pi*0.5
    test_ee = PoseStamped()
    test_ee.pose.position.x = test_pose.x.x - 0.2
    test_ee.pose.position.y = test_pose.x.y - 0.4
    test_u = TwistStamped()
    test_u.twist.linear.x = 0.3
    test_u.twist.linear.y = 0.3

    test_x_k = np.array([test_pose.x.x, test_pose.x.y, test_pose.x.theta,
                         test_ee.pose.position.x, test_ee.pose.position.y])
    test_u_k = np.array([test_u.twist.linear.x, test_u.twist.linear.y])

    next_state = svm_dynamics2.predict(test_x_k, test_u_k)
    gradient = svm_dynamics2.jacobian(test_x_k, test_u_k)

    print 'test_state.x:\n', test_x_k, test_u_k
    print 'next_state.x:\n', next_state
    print 'Jacobian:\n', gradient

    plot_out_path = '/home/thermans/sandbox/dynamics/'
    for i, trial in enumerate(plio.push_trials):
        (Y_hat, Y, X) = svm_dynamics2.test_batch_data(trial)
        plot_predicted_vs_observed_deltas(Y, Y_hat, out_path = plot_out_path, show_plot = False,
                                          suffix = ' '+str(i) )
        plot_predicted_vs_observed_tracks(Y, Y_hat, X, show_plot = False, out_path = plot_out_path,
                                          suffix = ' '+str(i) )
    # print 'Jacobian', svm_dynamics2.J
    return svm_dynamics2

def test_svm_new(base_dir_name):
    target_names = [dynamics_learning._DELTA_OBJ_X_OBJ,
                    dynamics_learning._DELTA_OBJ_Y_OBJ,
                    dynamics_learning._DELTA_OBJ_THETA_WORLD,
                    dynamics_learning._DELTA_EE_X_OBJ,
                    dynamics_learning._DELTA_EE_Y_OBJ]

    feature_names = [dynamics_learning._EE_X_OBJ,
                     dynamics_learning._EE_Y_OBJ,
                     dynamics_learning._U_X_OBJ,
                     dynamics_learning._U_Y_OBJ]

    train_file_base_name = base_dir_name + 'train'
    val_file_base_name = base_dir_name + 'validate'

    # Read data from disk
    (train_X, train_Y) = dynamics_learning.read_dynamics_learning_example_files(train_file_base_name)
    # TODO: Create SVM class
    epsilons = []
    for i in xrange(len(target_names)):
        epsilons.append(1e-3)
    delta_t = 1./9.
    n = 5
    m = 2

    svr_dynamics = SVRPushDynamics(delta_t, n, m, epsilons=epsilons, feature_names = feature_names,
                                   target_names = target_names, kernel_type='RBF')
    # Do Learning
    kernel_params = {}
    for i in xrange(len(feature_names)):
        kernel_params[i] = '-g 0.05'
    svr_dynamics.learn_model(train_X, train_Y, kernel_params)

    # Test saving and loading
    svr_base_output_path = '/home/thermans/sandbox/dynamics/SVR_FILES/shitty'
    svr_param_file_name = '/home/thermans/sandbox/dynamics/SVR_FILES/shitty_params.txt'
    svr_dynamics.save_models(svr_base_output_path)

    svr_dynamics2 = SVRPushDynamics(param_file_name = svr_param_file_name)

    # Do verification on training set
    (Y_hat_train, Y_gt_train, X_train) = svr_dynamics.test_batch_data(train_X, train_Y)

    # Do Testing on validation set
    (val_X, val_Y) = dynamics_learning.read_dynamics_learning_example_files(val_file_base_name)
    (Y_hat, Y_gt, X) = svr_dynamics.test_batch_data(val_X, val_Y)

    # print 'svr_dynamics', svr_dynamics.kernel_types, svr_dynamics.feature_names, svr_dynamics.target_names, svr_dynamics.delta_t, svr_dynamics.n, svr_dynamics.m
    # print 'svr_dynamics2', svr_dynamics2.kernel_types, svr_dynamics2.feature_names, svr_dynamics2.target_names, svr_dynamics2.delta_t, svr_dynamics2.n, svr_dynamics2.m

    # Visualize training and validation results
    plot_out_path = '/home/thermans/sandbox/dynamics/'
    # plot_predicted_vs_observed_deltas(Y_gt_train, Y_hat_train, out_path = plot_out_path, show_plot = True,
    #                                   suffix = ' '+'train' )

    # plot_predicted_vs_observed_deltas(Y_gt, Y_hat, out_path = plot_out_path, show_plot = True,
    #                                   suffix = ' '+'val' )
    # TODO: Make this work, need trial based data
    # plot_predicted_vs_observed_tracks(Y_gt, Y_hat, X, show_plot = False, out_path = plot_out_path,
    #                                   suffix = ' '+'val' )

    cts0 = push_learning.ControlTimeStep()
    cts0.x.x = 1.0
    cts0.x.y = 2.0
    cts0.x.theta = 0.5*pi
    cts0.z = 2.0
    cts0.ee.position.x = -0.5
    cts0.ee.position.y = 3.0
    cts0.ee.position.z = 2.5
    q = quaternion_from_euler(0.0,0.0,0.5*pi)
    cts0.ee.orientation.x = q[0]
    cts0.ee.orientation.y = q[1]
    cts0.ee.orientation.z = q[2]
    cts0.ee.orientation.w = q[3]

    cts0.u.linear.x = 2.0
    cts0.u.linear.y = 1.0
    cts0.u.angular.x = -pi
    cts0.t = 0.0

    cts1 = push_learning.ControlTimeStep()
    cts1.x.x = cts0.x.x + 1.0
    cts1.x.y = cts0.x.y + 1.5
    cts1.x.theta = cts0.x.theta + 0.5*pi
    cts1.z = cts0.z+0.25

    cts1.ee.position.x = cts0.ee.position.x + 0.75
    cts1.ee.position.y = cts0.ee.position.y + 1.75
    cts1.ee.position.z = cts0.ee.position.z - 0.25
    q = quaternion_from_euler(0.0,0.0,0.25*pi)
    cts1.ee.orientation.x = q[0]
    cts1.ee.orientation.y = q[1]
    cts1.ee.orientation.z = q[2]
    cts1.ee.orientation.w = q[3]
    cts1.t = 0.25
    # trial.trial_trajectory.append(cts0)
    # trial.trial_trajectory.append(cts1)
    # trial.trial_trajectory.append(cts1)

    x_k = [cts0.x.x, cts0.x.y, -cts0.x.theta, cts0.ee.position.x, cts0.ee.position.y]
    u_k = [cts0.u.linear.x, cts0.u.linear.y]
    deltas = [0.5, 1.0, 0.5*pi, -1.0, 0.5]
    print 'x_k',x_k
    print 'u_k',u_k
    print svr_dynamics2.transform_opt_vector_to_feat_vector(x_k, u_k, [])
    print 'deltas',deltas
    print svr_dynamics2.transform_svm_results_to_opt_vector(x_k, u_k, deltas)

    return svr_dynamics2

def test_mpc(base_dir_name):
    delta_t = 1.0/9.0
    H = 10
    n = 5
    m = 2
    u_max = 0.05
    sigma = 0.0001
    plot_output_path = '/home/thermans/sandbox/mpc_plots/with_ee/'
    # plot_output_path = ''
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
    goal_loc.x = 0.75
    goal_loc.y = -0.25

    p1 = Pose2D()
    p1.x = 0.5
    p1.y = 0.0#5

    p2 = Pose2D()
    p2.x = 1.0
    p2.y = -0.25

    p4 = Pose2D()
    p4.x = 2.5
    p4.y = 0.0

    pose_list = [goal_loc]
    trajectory_generator = ptg.PiecewiseLinearTrajectoryGenerator(0.01, 2)
    # trajectory_generator = ptg.ViaPointTrajectoryGenerator()
    x_d = trajectory_generator.generate_trajectory(cur_state.x, pose_list)

    if test_trajectory:
        for i in xrange(len(x_d)):
            print 'x_d[',i,'] =', x_d[i]
        q_fake = np.zeros((n+m)*H)

        plot_desired_vs_controlled(q_fake, x_d, x0, n, m, show_plot=True, suffix='-piecewise-linear',
                                   out_path=plot_output_path)

    # TODO: Test with a more complicated dynamics model
    # dyn_model = NaiveInputDynamics(delta_t, n, m)
    dyn_model = test_svm_new(base_dir_name)

    # TODO: Improve the way noise is added to make this better
    sim_model = StochasticNaiveInputDynamics(delta_t, n, m, sigma)

    mpc =  ModelPredictiveController(dyn_model, H, u_max, delta_t)

    q_gt = []
    q_stars = []
    u_gt = []
    for i in xrange(len(x_d)-1):
        # Update desired trajectory
        # TODO: Recompute trajectory from current pose
        # x_d = trajectory_generator.generate_trajectory(cur_state.x, pose_list)
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
            plot_controls(q_cur, x0, n, m, u_max, show_plot=False, suffix='-q*['+str(i)+']',
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
        plot_controls(q_gt, x0, n, m, u_max, show_plot=False, suffix='-q*['+str(len(x_d)-1)+']',
                      out_path=plot_output_path)
        plot_desired_vs_controlled(q_gt, x_d, x0, n, m, show_plot=True, suffix='-q*['+str(len(x_d)-1)+']',
                                   t=len(x_d), out_path=plot_output_path)

    # Plot initial guess trajectory
    # mpc.H = 10
    # U_init = mpc.get_U_init(x0, x_d)
    # q0 = mpc.get_q0(x0, U_init, xtra)
    # plot_desired_vs_controlled(q0, x_d, x0, n, m, show_plot=True, suffix='-q0', opt_path=plot_output_path)

def analyze_mpc_trial_data(aff_file_name, wait_for_renders=False):
    # Fixed parameters
    n = 5
    m = 2
    u_max = 0.03

    # Get derived names from aff file name
    q_star_file_name = aff_file_name[:-4]+'-q_star.txt'
    traj_file_name = aff_file_name[:-4]+'-trajectory.txt'
    aff_dir_path = aff_file_name[:-len(aff_file_name.split('/')[-1])]

    # Create output directories to store analysis
    analysis_dir = aff_file_name[:-4]+'-analysis/'
    render_out_dir = analysis_dir + 'tracking/'
    traj_out_dir = analysis_dir + 'planning/'

    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)
    if not os.path.exists(render_out_dir):
        os.mkdir(render_out_dir)
    if not os.path.exists(traj_out_dir):
        os.mkdir(traj_out_dir)

    # Read in all files
    plio = push_learning.CombinedPushLearnControlIO()
    plio.read_in_data_file(aff_file_name)
    traj_io = PushTrajectoryIO()
    trajs = traj_io.read_file(traj_file_name)
    print 'Read in ', len(trajs), ' planned trajectories\n'
    q_star_io = MPCSolutionIO()
    q_stars = q_star_io.read_file(q_star_file_name)
    print 'Read in ', len(q_stars), ' control plans\n'

    # TODO: Do analysis of dynamics learning too
    dynamics_model = None
    if dynamics_model is not None:
        print 'Plotting learned SVM predictions'
        for i, trial in enumerate(plio.push_trials):
            (Y_hat, Y, X) = dynamics_model.test_batch_data(trial)
            plot_predicted_vs_observed_deltas(Y, Y_hat, out_path = anlaysis_dir, show_plot = False,
                                              suffix = '_' + str(i))
            plot_predicted_vs_observed_tracks(Y, Y_hat, X, out_path = anlaysis_dir, show_plot = False,
                                              suffix = '_' + str(i))

    # Plot all desired trajectories on a single plot
    # Plot actual trajectory (over desired path(s)?)
    print 'Plotting batch trajectories'
    plot_tracks(plio.push_trials, out_path = analysis_dir, show_plot = False)

    plot_all_planned_trajectories(trajs, plio.push_trials, out_path = analysis_dir, show_plot = False,
                                  suffix = '-headings', show_headings = True)
    plot_all_planned_trajectories(trajs, plio.push_trials, out_path = analysis_dir, show_plot = False)

    # Plot actual controls
    plot_all_controls(plio.push_trials, u_max, out_path = analysis_dir, show_plot = False)

    print 'Plotting individual trajectories'
    trial_idx = -1
    for traj, q_star in zip(trajs, q_stars):
        if q_star[0] == 0:
            trial_idx += 1
        try:
            x0_state = plio.push_trials[trial_idx].trial_trajectory[q_star[0]]
        except IndexError:
            print 'Error trying to read trial', trial_idx, 'trajectory step', q_star[0]
            continue

        x0 = np.array([x0_state.x.x, x0_state.x.y, x0_state.x.theta,
                       x0_state.ee.position.x, x0_state.ee.position.y])

        print 'Plotting trial', trial_idx, ':', q_star[0]
        plot_suffix = '-'+str(trial_idx)+'-q_star['+str(q_star[0])+']'
        plot_desired_vs_controlled(q_star[1], traj[1], x0, n, m, show_plot=False,
                                   suffix=plot_suffix, out_path=traj_out_dir)
        plot_controls(q_star[1], x0, n, m, u_max, show_plot=False, suffix=plot_suffix, out_path=traj_out_dir)

        # Plot with history
        plot_suffix += '-history'
        plot_desired_vs_controlled_with_history(plio.push_trials[trial_idx].trial_trajectory[:q_star[0]+1],
                                                q_star[1], traj[1], n, m, show_plot = False,
                                                suffix = plot_suffix, out_path = traj_out_dir)

        plot_controls_with_history(plio.push_trials[trial_idx].trial_trajectory[:q_star[0]],
                                   q_star[1], x0, n, m, u_max, show_plot=False,
                                   suffix=plot_suffix, out_path=traj_out_dir)

    # Run render data script
    render_bin_name = roslib.packages.get_pkg_dir('tabletop_pushing')+'/bin/render_saved_data'
    if wait_for_renders:
        wait_time = 0
    else:
        wait_time = 1

    # Find correct starting index in the directory to render
    start_idx = -1
    # Read list of files in directory for feedback_K_n.pcd
    render_input_files = os.listdir(aff_dir_path)
    file_prefix_string = 'feedback_control_input_'
    movie_indices = {}
    for input_file in render_input_files:
        if input_file.startswith(file_prefix_string):
            k = int(input_file.split('_')[3])
            movie_indices[k] = k
            if start_idx < 0 or start_idx > k:
                start_idx = k

    if start_idx < 0:
        print 'Error: No files for rendering state and contact pt information'
        return

    print 'Chose start_idx of', start_idx

    p = subprocess.Popen([render_bin_name, aff_file_name, aff_dir_path, render_out_dir, str(wait_time),
                          str(start_idx)], shell=False)
    p.wait()

    # Render resultant state and object image sequences into movies using ffmpeg
    movie_render_bin_name = 'avconv'
    input_rate = 10
    output_rate = 29.97
    for i in movie_indices.keys():
        print 'Rendering state movie', i
        state_movie_in_name = render_out_dir+'state_'+str(i)+'_%d.png'
        state_movie_out_name = render_out_dir+'state_'+str(i)+'.mp4'
        p = subprocess.Popen([movie_render_bin_name, '-y', '-r', str(input_rate), '-i', state_movie_in_name,
                              '-r', str(output_rate), '-b', '1000k', state_movie_out_name])
        p.wait()
        print 'Rendering contact pt movie', i
        contact_pt_movie_in_name = render_out_dir+'contact_pt_'+str(i)+'_%d.png'
        contact_pt_movie_out_name = render_out_dir+'contact_pt_'+str(i)+'.mp4'
        p = subprocess.Popen([movie_render_bin_name, '-y', '-r', str(input_rate), '-i', contact_pt_movie_in_name,
                              '-r', str(output_rate), '-b', '1000k', contact_pt_movie_out_name])
        p.wait()

if __name__ == '__main__':
    # analyze_mpc_trial_data(sys.argv[1])
    # test_svm_stuff(sys.argv[1])
    test_mpc()
