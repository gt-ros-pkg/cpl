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
from geometry_msgs.msg import PoseStamped, TwistStamped, Twist
from tabletop_pushing.srv import *
from tabletop_pushing.msg import *
from math import copysign, pi
import svmutil
import numpy as np
import scipy.optimize as opt

def pushMPCObjectiveFunction(q, H, n, m, x0, xD, xtra, lambda_dynamics, dyn_model):
    '''
    q - decision variables of the form (X[1],...,X[H], U[0],...,U[H-1])
    H - time horizon
    n - system state dimension
    m - control dimension
    x0 - inital state
    xD - desired trajectory
    xtra - other features for dynamics prediction
    lambda_dynamics - weight for penalizing deviation from dynamics model
    dyn_model - model simulating system dynamics of the form x[k+1] = dyn_model.predict(x[k], u[k], xtra)
    '''
    # Pull out x and u from q (Can remove for speed up later if necessary)
    x = [x0]
    u = []
    step = n+m
    for k in xrange(H):
        x_start = k*step
        x_stop = x_start+n
        u_start = x_stop
        u_stop = u_start+m
        x.append(q[x_start:x_stop])
        u.append(q[u_start:u_stop])

    # Evaluate f_dyn(x[k], u[k], xtra) for all pairs and append to x_hat
    x_hat = [x0] # Prepend x_hat with known x0
    for k in xrange(H):
        x_hat.append(dyn_model.predict(x_hat[k], u[k], xtra))

    # Dynamics constraints
    score = 0.0
    for k in xrange(H):
        score += sum(abs(x[k+1] - x_hat[k+1]))
    score *= lambda_dynamics # Scale dynamics constraints

    # Goal trajectory constraints
    for k in xrange(H):
        score += sum(abs(x_desired[k+1] - x_hat[k+1]))
    return score

def pushMPCObjectiveGradient(q, H, n, m, x0, xD, xtra, lambda_dynamics, f_dyn):
    return 0.0

class ModelPredictiveController:
    def __init__(self, model, H=5, u_max=1.0, lambda_dynamics=1.0):
        '''
        model - prediction function for use inside the optimizer
        H - the lookahead horizon for MPC
        u_max - maximum allowed velocity
        lambda_dynamics - weight for soft dynamics constraints
        '''
        self.dyn_model = model
        self.H = H # Time horizon
        self.N = 20 # Non control parameter feature space dimension
        self.n = 3 # Predicted state space dimension
        self.m = 2 # Control space dimension
        bounds_k = []
        for i in xrange(self.n):
            bounds_k.append((None, None))
        for i in xrange(self.m):
            bounds_k.append((-u_max, u_max))
        self.opt_bounds = []
        for i in xrange(self.H):
            self.opt_bounds.extend(bounds_k)
        self.lambda_dynamics = lambda_dynamics

    def feedbackControl(self, cur_state, ee_pose, x_desired):
        # TODO: Get initial guess(es)
        q0 = np.array([0.0])
        x0 = np.array([0.0]) # TODO: Get inital state
        # Ensure x_desired[0] == x0
        # Perform optimization
        xtra = []
        opt_args = (self.H, self.n, self.m, x0, x_desired, xtra, lambda_dynamics, self.dyn_model)
        q_star, opt_val, d_info = opt.fmin_l_bfgs_b(func = pushMPCObjectiveFunction,
                                                    x0 = q0,
                                                    fprime = pushMPCObjectiveGradient,
                                                    args = opt_args,
                                                    bounds = self.opt_bounds)

        # TODO: Convert optimization result to twist
        u_star = q_star[self.m:self.m+self.n]
        u = TwistStamped
        u.header.frame_id = 'torso_lift_link'
        u.header.stamp = rospy.Time.now()
        u.twist.linear.z = 0.0
        u.twist.angular.x = 0.0
        u.twist.angular.y = 0.0
        u.twist.angular.z = 0.0
        u.twist.linear.x = u_star[0]
        u.twist.linear.x = u_star[1]
        return u

class NaiveInputModel:
    def __init__(self, detla_t):
        self.delta_t = delta_t

    def predict_state(self, cur_state, ee_pose, u):
        '''
        Predict the next state given current state estimates and control input
        cur_sate - current state estimate of form VisFeedbackPushTrackingFeedback()
        ee_pose - current end effector pose estimate of type PoseStamped()
        u - control to evaluate of type TwistStamped()
        '''
        next_state = VisFeedbackPushTrackingFeedback()
        next_state.x.x = cur_state.x.x + u.linear.x*self.delta_t
        next_state.x.y = cur_state.x.y + u.linear.y*self.delta_t
        next_state.x.theta = cur_state.x.theta
        next_state.x_dot.x = u.linear.x
        next_state.x_dot.theta = u.linear.y
        next_state.x_dot.theta = cur_state.x_dot.theta
        return next_state

class SVMPushModel:
    def __init__(self, svm_file_names=None, epsilons=None, kernel_type=None, m=3):
        '''
        svm_files - list of svm model files to read from disk
        epsilons - list of epislon values for the epislon insensitive loss function (training only)
        kernel_type - type of kernel to use for traning, can be any of the self.KERNEL_TYPES keys (training only)
        m - number of output dimensions in the model (training only)
        '''
        self.svm_models = []
        if svm_file_names is not None:
            for svm_file_name in svm_file_names:
                # print 'Loading file', svm_file_name
                self.svm_models.append(svmutil.svm_load_model(svm_file_name))

        if epsilons is not None:
            self.epsilons = epsilons
        else:
            self.epsilons = []
            for i in xrange(m):
                self.epsilons.append(1e-6)

        self.KERNEL_TYPES = {'linear': 0, 'polynomial': 1, 'RBF': 2, 'sigmoid': 3, 'precomputed':4}
        if kernel_type is not None:
            self.kernel_type = kernel_type
        else:
            self.kernel_type = 'linear'

    def predict(self, x_k, u_k, xtra=[]):
        '''
        Predict the next state given current state estimates and control input
        x_k - current state estimate (list)
        u_k - current control to evaluate
        xtra - other features for SVM
        '''
        x = x_k[:]
        x.extend(u_k)
        x.extend(xtra)
        Y_hat = []
        y = [0]
        for svm_model in self.svm_models:
            [y_hat, _, _] = svmutil.svm_predict(y, x, svm_model)
            Y_hat.append(y_hat[0])
        return y_hat

    def predict_state(self, cur_state, ee_pose, u, xtra=[]):
        '''
        Predict the next state given current state estimates and control input
        cur_sate - current state estimate of form VisFeedbackPushTrackingFeedback()
        ee_pose - current end effector pose estimate of type PoseStamped()
        u - control to evaluate of type TwistStamped()
        xtra - other features for SVM
        '''
        # Get feature vector
        x = [self.transform_state_data_to_feat_vector(cur_state, ee_pose, u)]
        # Perform predictions
        Y_hat = []
        y = [0]
        for svm_model in self.svm_models:
            [y_hat, _, _] = svmutil.svm_predict(y, x, svm_model)
            Y_hat.append(y_hat[0])
        # print 'Y_hat = ', Y_hat
        # Transform predictions to correct state / class
        next_state = VisFeedbackPushTrackingFeedback()
        next_state.x.x = cur_state.x.x + Y_hat[0]
        next_state.x.y = cur_state.x.y + Y_hat[1]
        next_state.x.theta = cur_state.x.theta + Y_hat[2]
        return next_state

    def transform_state_data_to_feat_vector(self, cur_state, ee_pose, u):
        '''
        Get SVM feature vector from current state information.
        Needs to be updated whenever transform_trial-data_to_feat_vectors() is updated
        cur_sate - current state estimate of form VisFeedbackPushTrackingFeedback()
        ee_pose - current end effector pose estimate of type PoseStamped()
        u - control to evaluate of type TwistStamped()
        '''
        return [cur_state.x.x, cur_state.x.y, cur_state.x.theta,
                ee_pose.pose.position.x, ee_pose.pose.position.y,
                u.twist.linear.x, u.twist.linear.y]

    def transform_trial_data_to_feat_vectors(self, trajectory):
        '''
        Get SVM feature vector from push trial trajectory state information.
        Needs to be updated whenever transform_state_data_to_feat_vector() is updated
        trajectory - list of ControlTimeStep() defined in push_learning.py
        '''
        X = []
        Y = []
        for i in xrange(len(trajectory)-1):
            cts_t0 = trajectory[i]
            cts_t1 = trajectory[i+1]
            x_t = [cts_t0.x.x, cts_t0.x.y, cts_t0.x.theta,
                   cts_t0.ee.position.x, cts_t0.ee.position.y, cts_t0.u.linear.x,
                   cts_t0.u.linear.y]
            y_t = [cts_t1.x.x - cts_t0.x.x, cts_t1.x.y - cts_t0.x.y, cts_t1.x.theta - cts_t0.x.theta]
            X.append(x_t)
            Y.append(y_t)
        return (X, Y)

    def learn_model(self, learn_data):
        '''
        Learns SVMs to predict dynamics given the input data
        learn_data - List of PushCtrlTrial() defined in push_learning.py
        '''
        X = []
        Y = []
        # print 'len(learn_data)',len(learn_data)
        for trial in learn_data:
            (x, y) = self.transform_trial_data_to_feat_vectors(trial.trial_trajectory)
            X.extend(x)
            Y.extend(y)
        # print 'len(Y)', len(Y)
        # print 'len(X)', len(X)
        for i in xrange(len(Y[0])):
            Y_i = []
            for y in Y:
                Y_i.append(y[i])
            param_string = '-s 3 -t ' + str(self.KERNEL_TYPES[self.kernel_type]) + ' -p ' + str(self.epsilons[i])
            # TODO: Kernel specific options
            svm_model = svmutil.svm_train(Y_i, X, param_string)
            self.svm_models.append(svm_model)

    def save_models(self, output_file_names):
        '''
        Write svm models to disk
        output_file_names - list of file paths for saving to, assumes length = len(self.svm_models)
        '''
        if len(self.svm_models) > 0:
            for file_name, model in zip(output_file_names, self.svm_models):
                svmutil.svm_save_model(file_name, model)

#
# Offline Testing
#
import sys
import push_learning
if __name__ == '__main__':
    aff_file_name  = sys.argv[1]
    plio = push_learning.CombinedPushLearnControlIO()
    plio.read_in_data_file(aff_file_name)
    svm_dynamics = SVMPushModel()
    svm_dynamics.learn_model(plio.push_trials)
    base_path = '/u/thermans/data/svm_dyn/'
    output_paths = []
    output_paths.append(base_path+'delta_x_dyn.model')
    output_paths.append(base_path+'delta_y_dyn.model')
    output_paths.append(base_path+'delta_theta_dyn.model')
    svm_dynamics.save_models(output_paths)
    svm_dynamics2 = SVMPushModel(svm_file_names=output_paths)

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

    next_state = svm_dynamics2.predict(test_pose, test_ee, test_u)

    print 'test_state.x: ', test_pose.x
    print 'next_state.x: ', next_state.x
