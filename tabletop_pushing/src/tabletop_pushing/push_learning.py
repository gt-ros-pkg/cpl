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
from geometry_msgs.msg import Point, Pose2D, Twist
import tf.transformations as tr
from push_primitives import *
from tabletop_pushing.srv import *
from tabletop_pushing.msg import *
import rospy
import cv2
import numpy as np
import sys
from math import sin, cos, pi, sqrt, fabs, atan2, hypot, acos, isnan
#from pylab import *
import matplotlib.pyplot as plotter
import random
import os
import subprocess

_VERSION_LINE = '# v0.6'
_LEARN_TRIAL_HEADER_LINE = '# object_id/trial_id init_x init_y init_z init_theta final_x final_y final_z final_theta goal_x goal_y goal_theta push_start_point.x push_start_point.y push_start_point.z behavior_primitive controller proxy which_arm push_time precondition_method score [shape_descriptors]'
_CONTROL_HEADER_LINE = '# x.x x.y x.theta x_dot.x x_dot.y x_dot.theta x_desired.x x_desired.y x_desired.theta theta0 u.linear.x u.linear.y u.linear.z u.angular.x u.angular.y u.angular.z time hand.x hand.y hand.z'
_BAD_TRIAL_HEADER_LINE='#BAD_TRIAL'
_DEBUG_IO = False

def subPIAngle(theta):
    while theta < -pi:
        theta += 2.0*pi
    while theta > pi:
        theta -= 2.0*pi
    return theta

def point_line_dist(pt, a, b):
    '''
    Get the perpendicular distance from pt to the line defined through (a,b)
    '''
    A = np.asarray(a)
    B = np.asarray(b)
    P = np.asarray(pt)
    q = A - P
    n = B-A
    n_hat = n / np.linalg.norm(n)
    return np.linalg.norm(q-np.dot(q,n_hat)*n_hat)

class PushTrial:
    def __init__(self):
        self.object_id = ''
        self.controller = ''
        self.behavior_primitive = ''
        self.proxy = ''
        self.which_arm = ''
        self.precondition_method = ''
        self.init_centroid = Point()
        self.init_orientation = 0.0
        self.final_centroid = Point()
        self.final_orientation = 0.0
        self.goal_pose = Pose2D()
        self.start_point = Point()
        self.push_time = 0.0
        # NOTE: Everything below not saved to disk, just computed for convenience
        self.push_angle = 0.0
        self.push_dist = 0.0
        self.continuation = False
        self.score = -1.0
        self.shape_descriptor = []

    def __str__(self):
        return (self.object_id +
                ' (' + self.proxy + ', ' + self.controller + ', ' + self.behavior_primitive + ', ' +
                self.which_arm + '_arm'+', ' +self.precondition_method+'):\n' +
                'init_centroid:\n' + str(self.init_centroid) + '\n'+
                'init_orientation: ' + str(self.init_orientation) +'\n'+
                'final_centroid:\n' + str(self.final_centroid) + '\n'+
                'final_orientation: ' + str(self.final_orientation) + '\n'+
                'goal_pose:\n' + str(self.goal_pose) + '\n'+
                'push_time: ' + str(self.push_time))

class ControlTimeStep:
    def __init__(self, x, x_dot, x_desired, theta0, u, t):
        self.x = x
        self.x_dot = x_dot
        self.x_desired = x_desired
        self.theta0 = theta0
        self.u = u
        self.t = t

class PushCtrlTrial:
    def __init__(self):
        self.trial_start = None
        self.trial_end = None
        self.trial_trajectory = []

    def __str__(self):
        return (str(self.trial_start) + '\n' + str(self.trial_trajectory) + '\n' + str(self.trial_end))

class PushLearningIO:
    def __init__(self):
        self.data_out = None
        self.data_in = None

    def write_line(self, init_centroid, init_orientation, final_centroid,
                   final_orientation, goal_pose, push_start_point, behavior_primitive,
                   controller, proxy, which_arm, push_time, object_id,
                   push_point, precondition_method='centroid_push', push_score=-1):
        if self.data_out is None:
            rospy.logerr('Attempting to write to file that has not been opened.')
            return
        rospy.logdebug('Writing output line.\n')
        data_line = object_id+' '+str(init_centroid.x)+' '+str(init_centroid.y)+' '+str(init_centroid.z)+' '+\
            str(init_orientation)+' '+str(final_centroid.x)+' '+str(final_centroid.y)+' '+\
            str(final_centroid.z)+' '+str(final_orientation)+' '+\
            str(goal_pose.x)+' '+str(goal_pose.y)+' '+str(goal_pose.theta)+' '+\
            str(push_start_point.x)+' '+str(push_start_point.y)+' '+str(push_start_point.z)+' '+\
            behavior_primitive+' '+controller+' '+proxy+' '+which_arm+' '+str(push_time)+' '+precondition_method+' '+\
            str(push_score)+'\n'
        self.data_out.write(_LEARN_TRIAL_HEADER_LINE+'\n')
        self.data_out.write(data_line)
        self.data_out.flush()

    def parse_line(self, line):
        if line.startswith('#'):
            return None
        l  = line.split()
        l.reverse()
        num_objs = len(l)
        push = PushTrial()
        push.object_id = l.pop()
        push.init_centroid.x = float(l.pop())
        push.init_centroid.y = float(l.pop())
        push.init_centroid.z = float(l.pop())
        push.init_orientation = float(l.pop())
        push.final_centroid.x = float(l.pop())
        push.final_centroid.y = float(l.pop())
        push.final_centroid.z = float(l.pop())
        push.final_orientation = float(l.pop())
        push.goal_pose.x = float(l.pop())
        push.goal_pose.y = float(l.pop())
        push.goal_pose.theta = float(l.pop())
        push.start_point.x = float(l.pop())
        push.start_point.y = float(l.pop())
        push.start_point.z = float(l.pop())
        push.behavior_primitive = l.pop()
        push.controller = l.pop()
        push.proxy = l.pop()
        push.which_arm = l.pop()
        push.push_time = float(l.pop())
        if len(l) > 0:
            push.precondition_method = l.pop()
        else:
            push.precondition_method = 'push_centroid'
        if len(l) > 0:
            push.score = float(l.pop())
        else:
            push.score = -1.0
        push.shape_descriptor = []
        while len(l) > 0:
            push.shape_descriptor.append(float(l.pop()))

        push.push_angle = atan2(push.goal_pose.y-push.init_centroid.y,
                                push.goal_pose.x-push.init_centroid.x)
        push.push_dist = hypot(push.goal_pose.y-push.init_centroid.y,
                               push.goal_pose.x-push.init_centroid.x)
        if push.init_centroid.x > 1.0:
            print 'Greater than 1.0 x: ', str(push)
        return push

    def write_pre_push_line(self, init_centroid, init_orientation, goal_pose, push_start_point, behavior_primitive,
                            controller, proxy, which_arm, object_id, precondition_method,
                            predicted_score, shape_descriptor=None):
        if self.data_out is None:
            rospy.logerr('Attempting to write to file that has not been opened.')
            return
        rospy.logdebug('Writing pre push output line.\n')
        data_line = object_id+' '+str(init_centroid.x)+' '+str(init_centroid.y)+' '+str(init_centroid.z)+' '+\
            str(init_orientation)+' '+str(0.0)+' '+str(0.0)+' '+\
            str(0.0)+' '+str(0.0)+' '+\
            str(goal_pose.x)+' '+str(goal_pose.y)+' '+str(goal_pose.theta)+' '+\
            str(push_start_point.x)+' '+str(push_start_point.y)+' '+str(push_start_point.z)+' '+\
            behavior_primitive+' '+controller+' '+proxy+' '+which_arm+' '+str(0.0)+' '+precondition_method+ ' ' +\
            str(predicted_score)

        if shape_descriptor is not None:
            for s in shape_descriptor:
                data_line += ' '+str(s)
        data_line+='\n'
        self.data_out.write(_LEARN_TRIAL_HEADER_LINE+'\n')
        self.data_out.write(data_line)
        self.data_out.flush()

    def write_bad_trial_line(self):
        self.data_out.write(_BAD_TRIAL_HEADER_LINE+'\n')
        self.data_out.flush()

    def read_in_data_file(self, file_name):
        data_in = file(file_name, 'r')
        x = [self.parse_line(l) for l in data_in.readlines()]
        data_in.close()
        x = filter(None, x)
        return self.link_repeat_trials(x)

    def link_repeat_trials(self, trials):
        for i in range(1,len(trials)):
            if (trials[i].goal_pose.y == trials[i-1].goal_pose.y and
                trials[i].goal_pose.x == trials[i-1].goal_pose.x and
                trials[i].behavior_primitive == trials[i-1].behavior_primitive and
                trials[i].proxy == trials[i-1].proxy and
                trials[i].controller == trials[i-1].controller):
                if trials[i].which_arm == trials[i-1].which_arm:
                    # print 'Continuation with different arm'
                    pass
                else:
                    trials[i].continuation = True
        return trials

    def open_out_file(self, file_name):
        self.data_out = file(file_name, 'a')
        self.data_out.write(_VERSION_LINE+'\n')
        # self.data_out.write(_LEARN_TRIAL_HEADER_LINE+'\n')
        self.data_out.flush()

    def close_out_file(self):
        self.data_out.close()

class ControlAnalysisIO:
    def __init__(self):
        self.data_out = None
        self.data_in = None

    def write_line(self, x, x_dot, x_desired, theta0, u, time, hand_pose):
        if self.data_out is None:
            rospy.logerr('Attempting to write to file that has not been opened.')
            return
        data_line = str(x.x)+' '+str(x.y)+' '+str(x.theta)+' '+\
            str(x_dot.x)+' '+str(x_dot.y)+' '+str(x_dot.theta)+' '+\
            str(x_desired.x)+' '+str(x_desired.y)+' '+str(x_desired.theta)+' '+\
            str(theta0)+' '+str(u.linear.x)+' '+str(u.linear.y)+' '+str(u.linear.z)+' '+\
            str(u.angular.x)+' '+str(u.angular.y)+' '+str(u.angular.z)+' '+str(time)+' '+\
            str(hand_pose.position.x)+' '+str(hand_pose.position.y)+' '+str(hand_pose.position.z)+\
            '\n'
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
        self.data_out.write(_CONTROL_HEADER_LINE+'\n')
        self.data_out.flush()

    def close_out_file(self):
        self.data_out.close()

class CombinedPushLearnControlIO:
    def __init__(self):
        self.pl_io = PushLearningIO()
        self.ctrl_io = ControlAnalysisIO()
        self.push_trials = []

    def read_in_data_file(self, file_name):
        data_in = file(file_name, 'r')
        read_pl_trial_line = False
        read_ctrl_line = False
        trial_is_start = True

        trial_starts = 0
        bad_stops = 0
        object_comments = 0
        control_headers = 0

        self.push_trials = []
        current_trial = PushCtrlTrial()
        for line in  data_in.readlines():
            if line.startswith(_VERSION_LINE):
                if _DEBUG_IO:
                    print 'Ignoring version line'
                continue
            elif line.startswith(_LEARN_TRIAL_HEADER_LINE):
                object_comments += 1

                if _DEBUG_IO:
                    print 'Read learn trial header'
                if trial_is_start:
                    trial_starts += 1
                read_pl_trial_line = True
                read_ctrl_line = False
            elif line.startswith(_CONTROL_HEADER_LINE):
                if _DEBUG_IO:
                    print 'Read control header'
                control_headers += 1
                read_ctrl_line = True
            elif line.startswith(_BAD_TRIAL_HEADER_LINE):
                bad_stops += 1
                if _DEBUG_IO:
                    print 'BAD TRIAL: not adding current trial to list'
                # Reset trial and switch to read next pl_trial_line as start trial
                current_trial = PushCtrlTrial()
                trial_is_start = True
            elif read_pl_trial_line:
                if _DEBUG_IO:
                    print 'Reading trial'
                trial = self.pl_io.parse_line(line)
                read_pl_trial_line = False
                if trial_is_start:
                    current_trial.trial_start = trial
                    if _DEBUG_IO:
                        print 'Trial is start trial'
                else:
                    if _DEBUG_IO:
                        print 'Trial is end trial'
                    current_trial.trial_end = trial
                    self.push_trials.append(current_trial)
                    current_trial = PushCtrlTrial()
                trial_is_start = not trial_is_start
            elif read_ctrl_line:
                if _DEBUG_IO:
                    print 'Read ctrl pt'
                traj_pt = self.ctrl_io.parse_line(line)
                current_trial.trial_trajectory.append(traj_pt)
            else:
                if _DEBUG_IO:
                    print 'None of these?'
        data_in.close()
        print 'object_comments',object_comments
        print 'trial_starts',trial_starts
        print 'bad_stops',bad_stops
        print 'control_headers',control_headers

    def write_example_file(self, file_name, X, Y, normalize=False, debug=False):
        data_out = file(file_name, 'w')
        # print 'Normalize:', normalize
        k = 0
        for x,y in zip(X,Y):
            k += 1
            if debug:
                print y, x
            if isnan(y):
                print 'Skipping writing example: ', k
                continue
            data_line = str(y)
            if normalize:
                feature_sum = 0.0
                for xi in x:
                    feature_sum += xi
                for i, xi in enumerate(x):
                    if xi > 0:
                        data_line += ' ' + str(i+1)+':'+str(sqrt(xi/float(feature_sum)))
            else:
                for i, xi in enumerate(x):
                    if xi > 0:
                        data_line += ' ' + str(i+1)+':'+str(xi)
            data_line +='\n'
            data_out.write(data_line)
        print 'Wrote', k, 'examples'
        data_out.close()

    def read_example_file(self, file_name):
        data_in = file(file_name, 'r')
        lines = [l.split() for l in data_in.readlines()]
        data_in.close()
        Y = []
        X = []
        for line in lines:
            y = float(line.pop(0))
            Y.append(y)
            x = []
            for pair in line:
                idx, val = pair.split(':')
                idx = int(idx) - 1
                val = float(val)
                while len(x) < idx:
                    x.append(0)
                x.append(val)
            X.append(x)
        return (X,Y)

    def read_regression_prediction_file(self, file_name):
        data_in = file(file_name, 'r')
        Y_hat = [float(y.strip()) for y in data_in.readlines()]
        data_in.close()
        return Y_hat

class StartLocPerformanceAnalysis:

    def __init__(self):
        self.analyze_straight_line_push = self.analyze_straight_line_push_line_dist
        self.analyze_spin_push = self.analyze_spin_push_total_spin

    def compare_predicted_and_observed_push_scores(self, in_file_name, out_file_name=None):
        # Read in data
        plio = CombinedPushLearnControlIO()
        plio.read_in_data_file(in_file_name)
        file_out = None
        if out_file_name is not None:
            file_out = file(out_file_name, 'w')
        for i, p in enumerate(plio.push_trials):
            pred_score = p.trial_start.score
            # Compute observed push score
            observed_score = self.analyze_straight_line_push(p)
            print 'Trial [',i,'] : Pred: ', pred_score, '\tObserved: ', observed_score
            if file_out is not None:
                trial_line = str(pred_score) + ' ' + str(observed_score) + '\n'
                file_out.write(trial_line)
        if file_out is not None:
            file_out.close()

    def get_trial_features(self, file_name, use_spin=False):
        self.plio = CombinedPushLearnControlIO()
        self.plio.read_in_data_file(file_name)
        Y = []
        X = []
        for i, p in enumerate(self.plio.push_trials):
            if use_spin:
                y = self.analyze_spin_push(p)
            else:
                y = self.analyze_straight_line_push(p)
            if y < 0:
                continue
            x = p.trial_start.shape_descriptor
            Y.append(y)
            X.append(x)
        return (X,Y)

    def generate_example_file(self, file_in_name, file_out_name, normalize=False, set_train=False, set_test=False):
        (X, Y) = self.get_trial_features(file_in_name)
        print 'Read in', len(X), 'sample locations'
        self.plio.write_example_file(file_out_name, X, Y, normalize)
        if set_train:
            self.X_train = X[:]
            self.Y_train = Y[:]
        if set_test:
            self.X_test = X[:]
            self.Y_test = Y[:]

    def generate_train_and_test_files(self, file_in, file_out, normalize=False, train_percent=0.6,
                                      must_move_epsilon=0.05):
        (X,Y) = self.get_trial_features(file_in)
        Z = zip(X,Y)
        # random.shuffle(Z)
        if file_out.endswith('.txt'):
            train_file_out = file_out[:-4]+'_train'+'.txt'
            test_file_out = file_out[:-4]+'_test'+'.txt'
        else:
            train_file_out = file_out+'_train'
            test_file_out = file_out+'_test'

        # TODO: Make more informed based on object_id
        num_train_examples = int(train_percent*len(X))
        print 'Read in', len(X), 'examples'
        print 'Training file has', num_train_examples, 'examples'
        print 'Test file has', len(X)-num_train_examples, 'examples'
        self.Y_train = []
        self.X_train = []
        self.Y_test = []
        self.X_test = []

        for i, xy in enumerate(Z):
            x = xy[0]
            y = xy[1]
            if i < num_train_examples:
                self.Y_train.append(y)
                self.X_train.append(x)
            else:
                self.Y_test.append(y)
                self.X_test.append(x)
        print 'len(Y_train)', len(self.Y_train)
        self.plio.write_example_file(train_file_out, self.X_train, self.Y_train, normalize)
        print 'len(Y_test)', len(self.Y_test)
        self.plio.write_example_file(test_file_out, self.X_test, self.Y_test, normalize)

    def plot_svm_results(self, pred_file):
        self.Y_hat = self.plio.read_regression_prediction_file(pred_file)

        max_score = 0.0
        Y_test_norm = self.Y_test[:]
        Y_hat_norm = self.Y_hat[:]
        Y_diffs = []
        for i in xrange(len(Y_test_norm)):
            # TODO: Get max and normalize
            if self.Y_test[i] > max_score:
                max_score = self.Y_test[i]
            if self.Y_hat[i] > max_score:
                max_score = self.Y_hat[i]
            Y_diffs.append(fabs(self.Y_hat[i]-self.Y_test[i]))
        for i in xrange(len(Y_test_norm)):
            Y_test_norm[i] = self.Y_test[i]/max_score
            Y_hat_norm[i] = self.Y_hat[i]/max_score
        print 'Average error is', sum(Y_diffs)/len(Y_diffs)
        print 'Max error is', max(Y_diffs)
        print 'Min error is', min(Y_diffs)
        p1, = plotter.plot(Y_test_norm, Y_hat_norm,'bx')
        plotter.xlabel('True Score')
        plotter.ylabel('Predicted score')
        plotter.title('Push Scoring Evaluation')
        plotter.xlim((0.0,1.0))
        plotter.ylim((0.0,1.0))
        plotter.show()

    def plot_svm_results_old(self, pred_file):
        self.Y_hat = self.plio.read_regression_prediction_file(pred_file)

        Ys_ordered = zip(self.Y_test[:], self.Y_hat[:])
        Ys_ordered.sort(key=lambda test_value: test_value[0])
        Y_test_ordered = []
        Y_hat_ordered = []
        Y_diffs = []
        for Ys in Ys_ordered:
            Y_test_ordered.append(Ys[0])
            Y_hat_ordered.append(Ys[1])
            Y_diffs.append(fabs(Ys[0]-Ys[1]))
        print 'Average error is', sum(Y_diffs)/len(Y_diffs)
        print 'Max error is', max(Y_diffs)
        print 'Min error is', min(Y_diffs)
        p1, = plotter.plot(Y_test_ordered,'bx')
        p2, = plotter.plot(Y_hat_ordered, 'r+')
        plotter.xlabel('Test index')
        plotter.ylabel('Straight push score')
        plotter.title('Push Scoring Evaluation')
        plotter.legend([p1,p2],['True Score', 'Predicted Score'], loc=2)
        plotter.show()

    def lookup_push_trial_by_shape_descriptor(self, trials, descriptor):
        for t in trials:
            match = True
            for a,b in zip(t.trial_start.shape_descriptor, descriptor):
                if a != b:
                    match = False
            if match:
                print t
                return t
        return None

    def analyze_straight_line_push_delta_theta(self, push_trial):
        '''
        Compute the average frame to frame change in orientation while pushing
        '''
        init_theta = push_trial.trial_start.init_orientation
        angle_errs = []
        for i, pt in enumerate(push_trial.trial_trajectory):
            if i == 0:
                theta_prev = init_theta
            else:
                theta_prev = push_trial.trial_trajectory[i-1].x.theta
            angle_errs.append(abs(subPIAngle(pt.x.theta - theta_prev)))

        if len(angle_errs) < 1:
            return -1
        mean_angle_errs = sum(angle_errs)/len(angle_errs)
        # print mean_angle_errs, 'rad'

        return mean_angle_errs

    def analyze_straight_line_push_line_dist(self, push_trial):
        '''
        Get the average distance of the current point to the desired straight line path
        '''
        init_pose = push_trial.trial_start.init_centroid
        goal_pose = push_trial.trial_start.goal_pose
        init_loc = (init_pose.x, init_pose.y)
        goal_loc = (goal_pose.x, goal_pose.y)

        line_dists = []
        for i, pt in enumerate(push_trial.trial_trajectory):
            cur_pt = (pt.x.x, pt.x.y)
            line_dist = point_line_dist(cur_pt, init_loc, goal_loc)
            line_dists.append(line_dist)

        if len(line_dists) < 1:
            return -1
        mean_line_dist = sum(line_dists)/len(line_dists)
        return mean_line_dist

    def analyze_straight_line_push_goal_vector_diff(self, push_trial, normalize_score=False):
        '''
        Get the average angler error between the instantaneous velocity direction of the object and the
        direction towards the goal
        '''
        init_pose = push_trial.trial_start.init_centroid
        goal_pose = push_trial.trial_start.goal_pose
        final_pose = push_trial.trial_end.final_centroid
        push_trial.trial_trajectory
        err_x = goal_pose.x - final_pose.x
        err_y = goal_pose.y - final_pose.y
        final_error =  hypot(err_x, err_y)
        total_change = hypot(final_pose.x - init_pose.x, final_pose.y - init_pose.y)
        angle_errs = []
        init_goal_vector = np.asarray([goal_pose.x - init_pose.x, goal_pose.y - init_pose.y])
        init_final_vector = np.asarray([final_pose.x - init_pose.x, final_pose.y - init_pose.y])
        final_angle_diff = self.angle_between_vectors(init_goal_vector, init_final_vector)
        final_angle_area = 0.5*fabs(init_goal_vector[0]*init_final_vector[1]-
                                    init_goal_vector[1]*init_final_vector[0])
        for i, pt in enumerate(push_trial.trial_trajectory):
            if i == 0:
                continue
            prev_pt = push_trial.trial_trajectory[i-1]
            goal_vector = np.asarray([goal_pose.x - prev_pt.x.x, goal_pose.y - prev_pt.x.y])
            push_vector = np.asarray([pt.x.x - prev_pt.x.x, pt.x.y - prev_pt.x.y])
            if hypot(push_vector[0], push_vector[1]) == 0:
                continue
            else:
                angle_errs.append(self.angle_between_vectors(goal_vector, push_vector))
        mean_angle_errs = sum(angle_errs)/len(angle_errs)
        # print mean_angle_errs, 'rad'
        # print final_error, 'cm'
        if normalize_score:
            score = 1.0-mean_angle_errs/pi
        else:
            score = mean_angle_errs

        return (score, final_error, total_change, final_angle_diff, final_angle_area)

    def analyze_spin_push_total_spin(self, push_trial):
        '''
        Get the average change in heading aggregated over the entire trial
        '''
        init_theta = push_trial.trial_start.init_theta
        goal_theta = push_trial.trial_start.goal_pose.theta

        delta_thetas = []
        prev_theta = init_theta
        for i, pt in enumerate(push_trial.trial_trajectory):
            cur_theta = pt.x.theta
            delta_thetas.append(abs(cur_theta-prev_theta))
            prev_theta = cur_theta

        if len(delta_thetas) < 1:
            return -1
        mean_delta_theta = sum(delta_thetas)/len(line_dists)
        return mean_delta_theta

    def analyze_spin_push_net_spin(self, push_trial):
        init_theta = push_trial.trial_start.init_theta
        final_theta = push_trial.trial_end.final_orientation
        return abs(final_theta - init_theta)

    def angle_between_vectors(self, a, b):
        a_dot_b = sum(a*b)
        norm_a = hypot(a[0], a[1])
        norm_b = hypot(b[0], b[1])
        if norm_a == 0 or norm_b == 0:
            # print 'Bad angle, returning max value'
            return pi
        return acos(a_dot_b/(norm_a*norm_b))

    def show_data_affinity(self):
        X = self.X_train[:]
        X.extend(self.X_test[:])
        Y = self.Y_train[:]
        Y.extend(self.Y_test[:])
        X_aff = np.zeros((len(X),len(X)))
        Y_aff = np.zeros((len(Y),len(Y)))
        for r in xrange(len(X)):
            for c in range(r, len(X)):
                X_aff[r,c] = sum(np.fabs(np.asarray(X[r]) - np.asarray(X[c])))
                Y_aff[r,c] = fabs(Y[r] - Y[c])
                print '(',r,',',c,'): ', X_aff[r,c], '\t', Y_aff[r,c]
        plotter.imshow(X_aff, plotter.cm.gray,interpolation='nearest')
        plotter.title('Xaff')
        plotter.figure()
        plotter.imshow(Y_aff, plotter.cm.gray, interpolation='nearest')
        plotter.title('Yaff')
        plotter.show()
        return X_aff,Y_aff

class BruteForceKNN:
    def __init__(self, data):
        self.data = data

    def find_k_neighbors(self, element, k=3, comp_func=None):
        if comp_func is None:
            comp_func = self.xy_dist
        k_closest = []
        k_dists = []
        for d in self.data:
            comp_dist = comp_func(element, d)
            inserted = False
            for i, close in enumerate(k_closest):
                if comp_dist < close:
                    k_closest.insert(d, i)
                    k_dists.insert(comp_dist,i)
                    inserted = True
            if not inserted and len(k_closest) < k:
                k_closest.append(d)
                k_dists.append(comp_dist)
            elif len(k_closest) > k:
                k_closest = k_closest[:k]
                k_dists = k_dists[:k]
        return (k_closest, k_dists)

    def xy_dist(self, a, b):
        return hypot(a.init_centroid.x - b.init_centroid.x,
                     a.init_centroid.y - b.init_centroid.y)

class PushLearningAnalysis:

    def __init__(self):
        self.raw_data = None
        self.io = PushLearningIO()
        self.xy_hash_precision = 20.0 # bins/meter
        self.num_angle_bins = 8

    def workspace_span(self, push):
        # score_fnc=self.compute_normalized_push_time
        score_fnc = self.compute_push_error_xy
        # score_fnc = self.compute_change_in_push_error_xy
        # Evaluate all push trials
        for t in self.all_trials:
            t.score = score_fnc(t)
        knn = BruteForceKNN(self.all_trials)
        neighbors, dists = knn.find_k_neighbors(push)
        # TODO: What are the best performing neighbors?

    #
    # Methods for computing marginals
    #
    def workspace_ranking(self):
        # likelihood_arg=['behavior_primitive','proxy','controller','which_arm']
        # likelihood_arg=['which_arm']
        likelihood_arg=['behavior_primitive','proxy','controller']
        score_fnc=self.compute_change_in_push_error_xy
        score_fnc=self.compute_push_error_xy
        workspace_ranks = self.rank_avg_pushes(trial_hash_fnc=self.hash_push_xy_angle,
                                               likelihood_arg=likelihood_arg,
                                               mean_push_fnc=self.get_mean_workspace_push,
                                               score_fnc=score_fnc)
        # print "Workspace push rankings:"
        for ranks in workspace_ranks:
            self.output_push_ranks(ranks,
                                   ['init_centroid.x','init_centroid.y','push_angle'],
                                   likelihood_arg, n=3)
        return workspace_ranks

    def object_ranking(self, use_raw=False, count_successes=True):
        '''
        Method to find the best performing (on average) behavior_primitive as a function of object_id
        '''
        cond_arg='object_id'
        likelihood_arg=['behavior_primitive','proxy','controller']
        # score_fnc=self.compute_normalized_push_time
        score_fnc=self.compute_push_error_xy
        # score_fnc=self.compute_change_in_push_error_xy
        if use_raw:
            rankings = self.rank_raw_pushes(trial_grouping_arg=cond_arg,
                                            score_fnc=score_fnc)
            print "\Raw object behavior rankings:"
            for ranks in rankings:
                self.output_push_ranks(ranks, cond_arg, likelihood_arg, n=5)
        elif count_successes:
            rankings = self.count_successful_pushes(trial_grouping_arg=cond_arg,
                                                    likelihood_arg=likelihood_arg,
                                                    score_fnc=score_fnc)
            print "\Successful counts for affordance-behaviors on different objects:"
            total_good = 0
            total_attempts = 0
            for ranks in rankings:
                num_good, num_attempts = self.output_push_counts(ranks, cond_arg, likelihood_arg, n=50)
                total_good += num_good
                total_attempts += num_attempts
            print 'Total good', total_good
            print 'Total attempts', total_attempts
        else:
            rankings = self.rank_avg_pushes(trial_grouping_arg=cond_arg,
                                            likelihood_arg=likelihood_arg,
                                            score_fnc=score_fnc)
            print "\nObject behavior rankings:"
            for ranks in rankings:
                self.output_push_ranks(ranks, cond_arg, likelihood_arg, n=1)
        return rankings

    def object_proxy_ranking(self):
        '''
        Method to find the best performing (on average) behavior_primitive as a function of object_id
        '''
        cond_arg='object_id'
        likelihood_arg='proxy'
        # score_fnc=self.compute_normalized_push_time
        # score_fnc=self.compute_push_error_xy
        score_fnc=self.compute_change_in_push_error_xy
        rankings = self.rank_avg_pushes(trial_grouping_arg=cond_arg,
                                        likelihood_arg=likelihood_arg,
                                        score_fnc=score_fnc)
        print "\nObject proxy rankings:"
        for ranks in rankings:
            self.output_push_ranks(ranks, cond_arg, likelihood_arg, n=3)
        return rankings

    def angle_ranking(self):
        '''
        Method to find the best performing (on average) behavior_primitive as a function of push_angle
        '''
        likelihood_arg=['behavior_primitive','proxy','controller']
        angle_ranks = self.rank_avg_pushes(
            trial_hash_fnc=self.hash_push_angle, likelihood_arg=likelihood_arg,
            mean_push_fnc=self.get_mean_angle_push)
        print "Angle push rankings:"
        for rank in angle_ranks:
            self.output_push_ranks(rank, 'push_angle', likelihood_arg, n=3)
        return angle_ranks

    def rank_avg_pushes(self, trial_hash_fnc=None, trial_grouping_arg=None,
                        likelihood_arg=None, mean_push_fnc=None,score_fnc=None):
        '''
        '''
        # Compute scores
        if score_fnc is None:
            score_fnc = self.compute_push_error_xy
        for t in self.all_trials:
            t.score = score_fnc(t)
        # Group objects based on conditional variables
        if trial_hash_fnc is not None:
            trial_groups = self.group_trials(self.all_trials, hash_function=trial_hash_fnc)
        else:
            trial_groups = self.group_trials(self.all_trials, trial_grouping_arg)

        # Group multiple trials of same type
        mean_groups = []
        for i, group_key in enumerate(trial_groups):
            trial_group = trial_groups[group_key]
            likelihood_arg_dict = self.group_trials(trial_group, likelihood_arg)
            # Average errors for each push key
            mean_group = []
            for arg_key in likelihood_arg_dict:
                if trial_grouping_arg is not None:
                    mean_push = self.get_mean_push(likelihood_arg_dict[arg_key],
                                                   trial_grouping_arg, likelihood_arg)
                else:
                    mean_push = mean_push_fnc(likelihood_arg_dict[arg_key])
                mean_group.append(mean_push)
            mean_groups.append(mean_group)

        # Choose best push for each conditional group
        ranked_pushes = []
        for i, group in enumerate(mean_groups):
            group.sort(compare_pushes)
            ranked_pushes.append(group)
        return ranked_pushes

    def count_successful_pushes(self, trial_hash_fnc=None, trial_grouping_arg=None,
                                likelihood_arg=None, mean_push_fnc=None,score_fnc=None, score_thresh=0.02):
        '''
        '''
        # Compute scores
        if score_fnc is None:
            score_fnc = self.compute_push_error_xy
        for t in self.all_trials:
            t.score = score_fnc(t)
        # Group objects based on conditional variables
        if trial_hash_fnc is not None:
            trial_groups = self.group_trials(self.all_trials, hash_function=trial_hash_fnc)
        else:
            trial_groups = self.group_trials(self.all_trials, trial_grouping_arg)

        # Group multiple trials of same type
        count_groups = []
        for i, group_key in enumerate(trial_groups):
            trial_group = trial_groups[group_key]
            likelihood_arg_dict = self.group_trials(trial_group, likelihood_arg)
            # Average errors for each push key
            count_group = []
            for arg_key in likelihood_arg_dict:
                push_count = self.get_count_push(likelihood_arg_dict[arg_key],
                                                 trial_grouping_arg, likelihood_arg, score_thresh)
                count_group.append(push_count)
            count_groups.append(count_group)

        # Choose best push for each conditional group
        ranked_pushes = []
        for i, group in enumerate(count_groups):
            group.sort(compare_counts)
            ranked_pushes.append(group)
        return ranked_pushes

    def get_successful_pushes(self, trial_hash_fnc=None, trial_grouping_arg=None,
                              likelihood_arg=None, mean_push_fnc=None,score_fnc=None, score_thresh=0.02):
        '''
        '''
        # Compute scores
        if score_fnc is None:
            score_fnc = self.compute_push_error_xy
        for t in self.all_trials:
            t.score = score_fnc(t)
        # Group objects based on conditional variables
        if trial_hash_fnc is not None:
            trial_groups = self.group_trials(self.all_trials, hash_function=trial_hash_fnc)
        else:
            trial_groups = self.group_trials(self.all_trials, trial_grouping_arg)

        # Group multiple trials of same type
        count_groups = []
        for i, group_key in enumerate(trial_groups):
            trial_group = trial_groups[group_key]
            likelihood_arg_dict = self.group_trials(trial_group, likelihood_arg)
            # Average errors for each push key
            count_group = []
            for arg_key in likelihood_arg_dict:
                for push in likelihood_arg_dict[arg_key]:
                    if push.score < score_thresh:
                        count_group.append(push)
            count_groups.append(count_group)
        return count_groups

    def rank_raw_pushes(self, trial_hash_fnc=None, trial_grouping_arg=None, score_fnc=None):
        '''
        '''
        # Compute scores
        if score_fnc is None:
            score_fnc = self.compute_push_error_xy
        for t in self.all_trials:
            t.score = score_fnc(t)
        # Group objects based on conditional variables
        if trial_hash_fnc is not None:
            trial_groups = self.group_trials(self.all_trials, hash_function=trial_hash_fnc)
        else:
            trial_groups = self.group_trials(self.all_trials, trial_grouping_arg)

        # Group multiple trials of same type
        ranked_pushes = []
        for i, group_key in enumerate(trial_groups):
            trial_group = trial_groups[group_key]
            trial_group.sort(compare_pushes)
            ranked_pushes.append(trial_group)
        return ranked_pushes

    def get_mean_push(self, arg_list, trial_grouping_arg, likelihood_arg):
        mean_push = PushTrial()
        mean_score, score_var = self.mean_and_variance(arg_list)
        mean_push.score = mean_score
        mean_push.var = score_var
        if type(trial_grouping_arg) is not list:
            trial_grouping_arg = [trial_grouping_arg]
        if type(likelihood_arg) is not list:
            likelihood_arg = [likelihood_arg]
        for arg in trial_grouping_arg:
            setattr(mean_push,arg, get_attr(arg_list[0],arg))
        for arg in likelihood_arg:
            setattr(mean_push,arg, get_attr(arg_list[0],arg))
        return mean_push


    def get_count_push(self, pushes, trial_grouping_arg, likelihood_arg, score_thresh):
        count_push = PushTrial()
        successful_count = 0
        for push in pushes:
            if push.score < score_thresh:
                successful_count += 1
        count_push.successful_count = successful_count
        count_push.total_count = len(pushes)
        if type(trial_grouping_arg) is not list:
            trial_grouping_arg = [trial_grouping_arg]
        if type(likelihood_arg) is not list:
            likelihood_arg = [likelihood_arg]
        for arg in trial_grouping_arg:
            setattr(count_push,arg, get_attr(pushes[0],arg))
        for arg in likelihood_arg:
            setattr(count_push,arg, get_attr(pushes[0],arg))
        return count_push

    def get_mean_angle_push(self, arg_list):
        mean_push = PushTrial()
        mean_score, score_var = self.mean_and_variance(arg_list)
        mean_push.score = mean_score
        mean_push.var = score_var
        mean_push.push_angle = self.hash_angle(arg_list[0].push_angle)
        mean_push.behavior_primitive = arg_list[0].behavior_primitive
        mean_push.controller = arg_list[0].controller
        mean_push.proxy = arg_list[0].proxy
        return mean_push

    def get_mean_workspace_push(self, arg_list):
        mean_push = PushTrial()
        mean_score, score_var = self.mean_and_variance(arg_list)
        mean_push.score = mean_score
        mean_push.var = score_var
        mean_push.init_centroid.x, mean_push.init_centroid.y = self.hash_xy(
            arg_list[0].init_centroid.x, arg_list[0].init_centroid.y)
        mean_push.push_angle = self.hash_angle(arg_list[0].push_angle)
        mean_push.behavior_primitive = arg_list[0].behavior_primitive
        mean_push.controller = arg_list[0].controller
        mean_push.proxy = arg_list[0].proxy
        mean_push.which_arm = arg_list[0].which_arm
        return mean_push

    def mean_and_variance(self, pushes):
        n = len(pushes)
        mean_score = 0
        for push in pushes:
            mean_score += push.score
        mean_score = mean_score / n

        if n <= 1:
            return (mean_score, 0)

        var_sum = 0
        for push in pushes:
            var_sum += push.score**2
        score_var = (var_sum-n*mean_score**2)/(n-1)

        return (mean_score, score_var)
    #
    # Grouping methods
    #
    def group_trials(self, all_trials, hash_attribute=None, hash_function=None):
        group_dict = {}
        # Group scored pushes by push angle
        for t in all_trials:
            if hash_function is not None:
                group_key = hash_function(t)
            elif type(hash_attribute) == list and len(hash_attribute) > 1:
                group_key = []
                for attr in hash_attribute:
                    group_key.append(get_attr(t,attr))
                group_key = tuple(group_key)
            elif type(hash_attribute) == list:
                group_key = get_attr(t, hash_attribute[0])
            else:
                group_key = get_attr(t, hash_attribute)
            try:
                group_dict[group_key].append(t)
            except KeyError:
                group_dict[group_key] = [t]
        return group_dict

    def group_trials_by_xy_and_push_angle(self, all_trials):
        '''
        Method to group all trials by initial table location and push angle
        '''
        groups = {}
        # Group scored pushes by push angle
        for t in all_trials:
            angle_key = self.hash_angle(t.push_angle)
            x_key, y_key = self.hash_xy(t.init_centroid.x, t.init_centroid.y)
            group_key = (x_key, y_key, angle_key)
            try:
                groups[group_key].append(t)
            except KeyError:
                groups[group_key] = [t]
        return groups

    def group_trials_by_push_angle(self, all_trials):
        '''
        Method to group all trials by initial table location and push angle
        '''
        return self.group_trials(all_trials, hash_function=self.hash_push_angle)

    def group_trials_by_xy(self, all_trials):
        '''
        Method to group all trials by initial table location and push angle
        '''
        xy_dict = {}
        for t in all_trials:
            xy_key = self.hash_xy(t.init_centroid.x, t.init_centroid.y)
            # Check if x_key exists
            try:
                xy_dict[xy_key].append(t)
            except KeyError:
                xy_dict[xy_key] = [t]
        return xy_dict
    #
    # Visualization functions
    #
    def visualize_push_choices(self, ranks, score_threshold=None, object_id=None):
        choices = []
        for rank in ranks:
            choices.append(rank[0])
        # load in image from dropbox folder
        disp_img = cv2.imread('/u/thermans/Dropbox/Data/choose_push/use_for_display.png')

        xy_groups = self.group_trials_by_xy(choices)

        for group_key in xy_groups:
            disp_img = self.draw_push_choices_on_image(xy_groups[group_key], disp_img,
                                                       score_threshold=score_threshold)
        cv2.imshow('Chosen pushes', disp_img)
        file_name = '/u/thermans/Desktop/push_learn_out'
        if object_id is not None:
            file_name += '_'+object_id
        if score_threshold is not None:
            file_name += '_'+str(score_threshold)

        file_name += '.png'
        cv2.imwrite(file_name, disp_img)
        cv2.waitKey()

    def draw_push_choices_on_image(self, choices, img, score_threshold=None):
        # TODO: Double check that this is all correct
        # TODO: load in transform and camera parameters from saved info file
        K = np.matrix([[525, 0, 319.5, 0.0],
                       [0, 525, 239.5, 0.0],
                       [0, 0, 1, 0.0]])
        tl = np.asarray([-0.0115423, 0.441939, 0.263569])
        q = np.asarray([0.693274, -0.685285, 0.157732, 0.157719])
        num_downsamples = 0
        table_height = -0.3
        P_w = np.matrix([[choices[0].init_centroid.x], [choices[0].init_centroid.y],
                         [table_height], [1.0]])
        # Transform choice location into camera frame
        T = (np.matrix(tr.translation_matrix(tl)) *
             np.matrix(tr.quaternion_matrix(q)))
        P_c = T*P_w
        # Transform camera point into image frame
        P_i = K*P_c
        P_i = P_i / P_i[2]
        u = P_i[0]/pow(2,num_downsamples)
        v = P_i[1]/pow(2,num_downsamples)

        valid_choices = 0
        for c in choices:
            if score_threshold is None:
                valid_choices += 1
            elif c.score < score_threshold:
                valid_choices += 1
        if valid_choices == 0:
            return img
        # Draw circle for the location
        radius = 13
        cv2.circle(img, (u,v), 3, [0.0,0.0,0.0],3)
        cv2.circle(img, (u,v), 3, [255.0,255.0,255.0], 1)
        # Draw Shadows for all angles
        for c in choices:
            if score_threshold is None:
                pass
            elif c.score > score_threshold:
                continue
            end_point = (u-sin(c.push_angle)*(radius), v-cos(c.push_angle)*(radius))
            color = [0.0, 0.0, 0.0] # Black
            #cv2.line(img, (u,v), end_point, color,3)
        for c in choices:
            if score_threshold is None:
                pass
            elif c.score > score_threshold:
                continue
            # Choose color by push type
            if True or c.which_arm == 'l':
                if c.behavior_primitive == GRIPPER_PUSH:
                    color = [255.0, 0.0, 0.0] # Blue
                elif c.behavior_primitive == GRIPPER_SWEEP:
                    color = [0.0, 255.0, 0.0] # Green
                elif c.behavior_primitive == OVERHEAD_PUSH:
                    color = [0.0, 0.0, 255.0] # Red
                elif c.behavior_primitive == GRIPPER_PULL:
                    color = [0.0, 255.0, 255.0] # Yellow
                # color = [0.0, 255.0, 0.0] # Green
            else:
                if c.behavior_primitive == GRIPPER_PUSH:
                    color = [128.0, 0.0, 128.0] # Magenta
                elif c.behavior_primitive == GRIPPER_SWEEP:
                    color = [0.0, 64.0, 129.0] # Brown
                elif c.behavior_primitive == OVERHEAD_PUSH:
                    color = [128.0, 128.0, 0.0] # Cyan
                elif c.behavior_primitive == GRIPPER_PULL:
                    color = [200.0, 200.0, 200.0] # White
                # color = [0.0, 0.0, 255.0] # Red
            # Draw line depicting the angle
            end_point = (u-sin(c.push_angle)*(radius), v-cos(c.push_angle)*(radius))
            cv2.line(img, (u,v), end_point, color, 2)
        return img

    def visualize_angle_push_choices(self, ranks, score_threshold=None):
        choices = []
        for rank in ranks:
            choices.append(rank[0])
        # load in image from dropbox folder
        disp_img = cv2.imread('/u/thermans/Dropbox/Data/choose_push/use_for_display.png')

        # TODO: Create a single group and display for all (x,y) locs

        for x in x_locs:
            for y in y_locs:
                for c in choices:
                    c.init_centroid.x = x
                    c.init_centroid.y = y
                    disp_img = self.draw_push_choices_on_image(choices,
                                                               disp_img,
                                                               score_threshold=score_threshold)
        cv2.imshow('Chosen pushes', disp_img)
        cv2.imwrite('/u/thermans/Desktop/push_learn_angle_out.png', disp_img)
        cv2.waitKey()

    #
    # IO Functions
    #
    def read_in_push_trials(self, file_name, object_id=None):
        self.all_trials = []
        object_ids = {}
        if object_id is not None:
            all_trials = self.io.read_in_data_file(file_name)
            for t in all_trials:
                if t.object_id in object_ids:
                    pass
                else:
                    object_ids[t.object_id] = t.object_id
                    print "object_id is:", t.object_id
                if object_id == t.object_id:
                    self.all_trials.append(t)
        else:
            self.all_trials = self.io.read_in_data_file(file_name)



    def output_push_choice(self, c, conditional_value, arg_value):
        '''
        Method takes strings [or lists of strings] of what attributes of PushTrail instance c to display
        '''
        if type(conditional_value) == list:
            conditional_str = ''
            for val in conditional_value:
                conditional_str += str(get_attr(c,val))+', '
            conditional_str = conditional_str[:-2]
        else:
            conditional_str = str(get_attr(c,conditional_value))
        if type(arg_value) == list:
            arg_str = ''
            for val in arg_value:
                arg_str += str(get_attr(c,val))+', '
            arg_str = arg_str[:-2]
        else:
            arg_str = str(get_attr(c,arg_value))
        print 'Choice for (' +  conditional_str + '): ('+ arg_str+ ') : ' + str(c.score) + ', ' + str(c.var)

    def output_push_ranks(self, ranks, conditional_value, arg_value, n=3):
        '''
        Method takes strings [or lists of strings] of what attributes of PushTrail instance c to display
        '''
        if len(ranks) == 0:
            return
        if type(conditional_value) == list:
            conditional_str = ''
            for val in conditional_value:
                conditional_str += str(get_attr(ranks[0], val))+', '
            conditional_str = conditional_str[:-2]
        else:
            conditional_str = str(get_attr(ranks[0], conditional_value))
        # print 'Ranks for (' +  conditional_str + '):'
        for i, c in enumerate(ranks):
            if i >= n:
                break
            if type(arg_value) == list:
                arg_str = ''
                for val in arg_value:
                    arg_str += str(get_attr(c,val))+', '
                arg_str = arg_str[:-2]
            else:
                arg_str = str(get_attr(c,arg_value))
            # print '\t ('+ arg_str+ ') : ' + str(c.score) + ', ' + str(c.var)

    def output_push_counts(self, ranks, conditional_value, arg_value, n=3):
        '''
        Method takes strings [or lists of strings] of what attributes of PushTrail instance c to display
        '''
        total_good = 0
        total_attempts = 0
        if len(ranks) == 0:
            return
        if type(conditional_value) == list:
            conditional_str = ''
            for val in conditional_value:
                conditional_str += str(get_attr(ranks[0], val))+', '
            conditional_str = conditional_str[:-2]
        else:
            conditional_str = str(get_attr(ranks[0], conditional_value))
        print conditional_str + ''
        for i, c in enumerate(ranks):
            if i >= n:
                break
            if type(arg_value) == list:
                arg_str = ''
                for val in arg_value:
                    arg_str += str(get_attr(c,val))+' & '
                arg_str = arg_str[:-3]
            else:
                arg_str = str(get_attr(c,arg_value))
            if c.successful_count > 0:
                total_good += c.successful_count
                total_attempts += c.total_count
            #     print ' & '+ arg_str+ ' & & ' + str(c.successful_count) + ' & ' + str(c.total_count) + ' \\\\'
        # print 'Total good: ', total_good
        # print 'Total attempts: ', total_attempts
        return total_good, total_attempts
    def output_loc_push_choices(self, choices):
        print "Loc push choices:"
        for c in choices:
            self.output_push_choice(c, ['init_centroid.x','init_centroid.y','push_angle'],
                                    ['behavior_primitive','controller','proxy'])

    #
    # Hashing Functions
    #

    def hash_push_angle(self, push):
        return self.hash_angle(push.push_angle)

    def hash_push_xy_angle(self, push):
        x_prime, y_prime = self.hash_xy(push.init_centroid.x, push.init_centroid.y)
        return (x_prime, y_prime, self.hash_angle(push.push_angle))

    def hash_angle(self, theta):
        # Group different pushes by push_angle
        bin = int((theta + pi)/(2.0*pi)*self.num_angle_bins)
        bin = max(min(bin, self.num_angle_bins-1), 0)
        return -pi+(2.0*pi)*float(bin)/self.num_angle_bins

    def hash_xy(self, x,y):
        # Group different pushes by push_angle
        x_prime = (round(x*self.xy_hash_precision)-1)/self.xy_hash_precision
        y_prime = (round(y*self.xy_hash_precision)-1)/self.xy_hash_precision
        return (x_prime, y_prime)

    #
    # Scoring Functions
    #

    def compute_push_error_xy(self, push):
        err_x = push.goal_pose.x - push.final_centroid.x
        err_y = push.goal_pose.y - push.final_centroid.y
        return hypot(err_x,err_y)

    def compute_change_in_push_error_xy(self, push):
        init_x_error = push.goal_pose.x - push.init_centroid.x
        init_y_error = push.goal_pose.y - push.init_centroid.y
        final_x_error = push.goal_pose.x - push.final_centroid.x
        final_y_error = push.goal_pose.y - push.final_centroid.y

        if push.final_centroid == 0.0 and push.final_centroid.y == 0.0:
            print 'Knocked off object'
            return 100.0
        init_error = hypot(init_x_error, init_y_error)
        final_error = hypot(final_x_error, final_y_error)
        return final_error / init_error

    def compute_push_error_push_dist_diff(self, push):
        desired_x = push.init_centroid.x + cos(push.push_angle)*push.push_dist
        desired_y = push.init_centroid.y + sin(push.push_angle)*push.push_dist
        d_x = push.final_centroid.x - push.init_centroid.x
        d_y = push.final_centroid.y - push.init_centroid.y
        actual_dist = sqrt(d_x*d_x + d_y*d_y)
        return fabs(actual_dist - push.push_dist)

    def compute_normalized_push_time(self, push):
        return push.push_time / push.push_dist

# Helper functions
def get_attr(instance, attribute):
    '''
    A wrapper for the builtin python getattr which handles recursive attributes of attributes
    '''
    attr_list = attribute.split('.')
    def get_nested_attr(instance, attr_list):
        if len(attr_list) == 1:
            return getattr(instance, attr_list[0])
        return get_nested_attr(getattr(instance, attr_list[0]), attr_list[1:])
    return get_nested_attr(instance, attr_list)

def compare_pushes(a, b):
    if a.score < b.score:
        return -1
    elif a.score > b.score:
        return 1
    else:
        return 0

def compare_counts(a, b):
    if a.successful_count > b.successful_count:
        return -1
    elif a.successful_count < b.successful_count:
        return 1
    else:
        return 0

def plot_controller_results(file_name, spin=False):
    # TODO: Plot with dashes and dots for no color distinction
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
    plotter.figure()
    plotter.plot(XS,YS)
    plotter.scatter(XS,YS)
    ax = plotter.gca()
    print ylim()
    print ax.set_xlim(xlim()[1]-(ylim()[1]-ylim()[0]), xlim()[1])
    headings = [plotter.Arrow(c.x.x, c.x.y, cos(c.x.theta)*0.01, sin(c.x.theta)*0.01, 0.05, axes=ax) for c in controls]
    arrows = [ax.add_patch(h) for h in headings]
    init_arrow = plotter.Arrow(controls[0].x.x, controls[0].x.y,
                               cos(controls[0].x.theta)*0.01,
                               sin(controls[0].x.theta)*0.01, 0.05,
                               axes=ax, facecolor='red')
    goal_arrow = plotter.Arrow(controls[0].x_desired.x, controls[0].x_desired.y,
                               cos(controls[0].x_desired.theta)*0.01,
                               sin(controls[0].x_desired.theta)*0.01, 0.05,
                               axes=ax, facecolor='green')
    ax.add_patch(goal_arrow)
    ax.add_patch(init_arrow)
    plotter.scatter(init_x, init_y, c='r')
    plotter.scatter(goal_x, goal_y, c='g')
    plotter.xlabel('x (meters)')
    plotter.ylabel('y (meters)')
    plotter.title('Tracker Ouput')
    plotter.savefig('/home/thermans/sandbox/tracker-output.png')
    plotter.figure()
    ux = [c.u.linear.x for c in controls]
    uy = [c.u.linear.y for c in controls]
    if spin:
        uy = [c.u.linear.z for c in controls]
    plotter.plot(ux)
    plotter.plot(uy)
    plotter.xlabel('Time')
    plotter.ylabel('Velocity (m/s)')
    plotter.legend(['u_x', 'u_y'])
    plotter.title('Feedback Controller - Input Velocities')
    plotter.savefig('/home/thermans/sandbox/feedback-input.png')
    plotter.figure()
    xdots = [c.x_dot.x for c in controls]
    ydots = [c.x_dot.y for c in controls]
    plotter.plot(xdots)
    plotter.plot(ydots)
    plotter.xlabel('Time')
    plotter.ylabel('Velocity (m/s)')
    plotter.title('Tracker Velocities')
    plotter.legend(['x_dot', 'y_dot'])
    plotter.savefig('/home/thermans/sandbox/tracker-velocities.png')
    plotter.figure()
    thetadots = [c.x_dot.theta for c in controls]
    plotter.plot(thetadots,c='r')
    plotter.xlabel('Time')
    plotter.ylabel('Velocity (rad/s)')
    plotter.title('Tracker Velocities')
    plotter.legend(['theta_dot'])
    plotter.savefig('/home/thermans/sandbox/tracker-theta-vel.png')
    plotter.figure()
    x_err = [c.x_desired.x - c.x.x for c in controls]
    y_err = [c.x_desired.y - c.x.y for c in controls]
    plotter.plot(x_err)
    plotter.plot(y_err)
    plotter.xlabel('Time')
    plotter.ylabel('Error (meters)')
    plotter.title('Position Error')
    plotter.legend(['x_err', 'y_err'])
    plotter.savefig('/home/thermans/sandbox/pos-err.png')
    plotter.figure()
    if spin:
        theta_err = [c.x_desired.theta - c.x.theta for c in controls]
    else:
        theta_err = [c.theta0 - c.x.theta for c in controls]
    plotter.plot(theta_err, c='r')
    plotter.xlabel('Time')
    plotter.ylabel('Error (radians)')
    if spin:
        plotter.title('Error in Orientation')
    else:
        plotter.title('Heading Deviation from Initial')
    plotter.savefig('/home/thermans/sandbox/theta-err.png')
    plotter.show()

def plot_junk():
    # plot_controller_results(sys.argv[1])
    req_object_id = None
    thresh = 1.0
    # TODO: Add more options to command line
    if len(sys.argv) > 1:
        data_path = str(sys.argv[1])
    else:
        print 'Usage:',sys.argv[0],'input_file [object_id] [distance_thresh]'
        quit()
    if len(sys.argv) > 2:
        req_object_id = str(sys.argv[2])
        if req_object_id == "None":
            req_object_id = None
    if len(sys.argv) > 3:
        thresh = float(sys.argv[3])
        print "thresh:", thresh

    pla = PushLearningAnalysis()
    pla.read_in_push_trials(data_path, req_object_id)

    # TODO: Use command line arguments to choose these
    workspace_ranks = pla.workspace_ranking()
    pla.visualize_push_choices(workspace_ranks, thresh, object_id=req_object_id)
    # angle_ranks = pla.angle_ranking()
    # pla.object_ranking()
    # pla.object_proxy_ranking()
    print 'Num trials: ' + str(len(pla.all_trials))

def read_example_file(file_name):
    data_in = file(file_name, 'r')
    lines = [l.split() for l in data_in.readlines()]
    data_in.close()
    Y = []
    X = []
    for line in lines:
        y = float(line.pop(0))
        Y.append(y)
        x = []
        for pair in line:
            idx, val = pair.split(':')
            idx = int(idx) - 1
            val = float(val)
            while len(x) < idx:
                x.append(0)
            x.append(val)
        X.append(x)
    return (X,Y)

def read_feature_file(file_name):
    data_in = file(file_name, 'r')
    lines = [l.split() for l in data_in.readlines()]
    data_in.close()
    X = []
    for line in lines:
        x = []
        for val in line:
            x.append(float(val))
        X.append(x)
    return X

def write_example_file(file_name, X, Y, normalize=False, debug=False):
    data_out = file(file_name, 'w')
    # print 'Normalize:', normalize
    i = 0
    for x,y in zip(X,Y):
        i += 1
        if debug:
            print y, x
        if isnan(y):
            print 'Skipping writing example: ', i
            continue
        data_line = str(y)
        if normalize:
            feature_sum = 0.0
            for xi in x:
                feature_sum += xi
            for i, xi in enumerate(x):
                if xi > 0:
                    data_line += ' ' + str(i+1)+':'+str(sqrt(xi/float(feature_sum)))
        else:
            for i, xi in enumerate(x):
                if xi > 0:
                    data_line += ' ' + str(i+1)+':'+str(xi)
        data_line +='\n'
        data_out.write(data_line)
    data_out.close()

def rewrite_example_file_features(original_file_name, feat_file_name, out_file_name, normalize=False, debug=False):
    old_X, Y = read_example_file(original_file_name)
    X = read_feature_file(feat_file_name)
    write_example_file(out_file_name, X, Y, normalize, debug)

def extract_shape_features_batch():
  base_dir = '/home/thermans/Dropbox/Data/start_loc_learning/point_push/'
  class_dirs = ['camcorder3', 'food_box3', 'large_brush3', 'small_brush3','soap_box3', 'toothpaste3']
  # class_dirs = ['toothpaste3']
  out_dir = base_dir+'examples_line_dist/'
  feat_dir = base_dir+'examples_line_dist/'
  # subprocess.Popen(['mkdir', '-p', out_dir], shell=False)

  for c in class_dirs:
      print 'Class:', c
      class_dir = base_dir+c+'/'
      files = os.listdir(class_dir)
      data_file = None
      for f in files:
          if f.startswith('aff_learn_out'):
              data_file = f
      if data_file is None:
          print 'ERROR: No data file in directory:', c
          continue
      aff_file = class_dir+data_file
      score_file = base_dir+'examples_line_dist/'+c[:-1]+'.txt'
      file_out = out_dir+c[:-1]+'_gt_scores.png'
      print '/home/thermans/src/gt-ros-pkg/cpl/tabletop_pushing/bin/extract_shape_features', aff_file, \
          class_dir, file_out, score_file
      p = subprocess.Popen(['/home/thermans/src/gt-ros-pkg/cpl/tabletop_pushing/bin/extract_shape_features',
                            aff_file, class_dir, file_out, score_file], shell=False)
      p.wait()

def read_and_score_raw_files():
  base_dir = '/home/thermans/Dropbox/Data/start_loc_learning/point_push/'
  class_dirs = ['camcorder3', 'food_box3', 'large_brush3', 'small_brush3','soap_box3', 'toothpaste3']
  out_dir = base_dir+'examples_line_dist/'
  for c in class_dirs:
      in_dir = base_dir+c+'/'
      files = os.listdir(in_dir)
      file_name = None
      for f in files:
          if f.startswith('aff_learn_out'):
              file_name = f
      if file_name is None:
          continue
      file_in = in_dir+file_name
      file_out = out_dir+c[:-1]+'.txt'
      print file_out
      slp = StartLocPerformanceAnalysis()
      slp.generate_example_file(file_in, file_out)

def compare_predicted_and_observed_push_scores(in_file_name, out_file_name=None):
    slp = StartLocPerformanceAnalysis()
    slp.compare_predicted_and_observed_push_scores(in_file_name, out_file_name)

def compare_predicted_and_observed_batch():
  base_dir = '/home/thermans/Dropbox/Data/ichr2013-results/hold_out_straight_line_results/'
  class_dirs = ['camcorder0', 'food_box0', 'large_brush0_offset07', 'small_brush0','soap_box0', 'toothpaste0']
  # base_dir = '/home/thermans/Dropbox/Data/ichr2013-results/rand_straight_line_results/'
  # class_dirs = ['camcorder0', 'food_box0', 'large_brush0', 'small_brush0','soap_box0', 'toothpaste0']
  out_dir = base_dir+'analysis/'
  for c in class_dirs:
      in_dir = base_dir+c+'/'
      files = os.listdir(in_dir)
      file_name = None
      for f in files:
          if f.startswith('aff_learn_out'):
              file_name = f
      if file_name is None:
          continue
      file_in = in_dir+file_name
      file_out = out_dir+c+'.txt'
      compare_predicted_and_observed_push_scores(file_in, file_out)

def rank_straw_scores(file_path):
    straw_file = file(file_path, 'r')
    lines = [l.split() for l in straw_file.readlines()]
    scores = [float(l[0])*100 for l in lines]
    major_dists = [float(l[1]) for l in lines]
    minor_dists = [float(l[2]) for l in lines]
    min_major_dist = 10000
    major_dist_idx = 0
    min_minor_dist = 10000
    minor_dist_idx = 0
    for i in xrange(len(major_dists)):
        if major_dists[i] < min_major_dist:
            min_major_dist = major_dists[i]
            major_dist_idx = i
        if minor_dists[i] < min_minor_dist:
            min_minor_dist = minor_dists[i]
            minor_dist_idx = i
    major_score = scores[major_dist_idx]
    minor_score = scores[minor_dist_idx]
    scores.sort()
    print 'Major idx score: ' + str(major_score) + ' is ranked ' + str(scores.index(major_score)+1)
    print 'Minor idx score: ' + str(minor_score) + ' is ranked ' + str(scores.index(minor_score)+1)
    print 'Scores are: ' + str(scores)

def rank_straw_scores_batch():
  base_dir = '/home/thermans/Dropbox/Data/start_loc_learning/point_push/major_minor_axis_point_data/'
  classes = ['camcorder3', 'food_box3', 'large_brush3', 'small_brush3','soap_box3', 'toothpaste3']
  for c in classes:
      file_path = base_dir + 'straw_scores_' + c + '.txt'
      print 'Ranks for: ', c
      rank_straw_scores(file_path)
      print '\n'

if __name__ == '__main__':
    compare_predicted_and_observed_batch()
    # read_and_score_raw_files()
    # extract_shape_features_batch()
    # rank_straw_scores_batch()
