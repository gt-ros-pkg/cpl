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
import scipy.optimize as opt

def pushMPCObjectiveFunction(q, H, n, m, x0, x_d, xtra, lambda_dynamics, dyn_model):
    '''
    q - decision variable vector of the form (U[0], X[1], U[1],..., U[H-1], X[H])
    H - time horizon
    n - system state dimension
    m - control dimension
    x0 - inital state
    x_d - desired trajectory
    xtra - other features for dynamics prediction
    lambda_dynamics - weight for penalizing deviation from dynamics model
    dyn_model - model simulating system dynamics of the form x[k+1] = dyn_model.predict(x[k], u[k], xtra)
    '''
    # Pull out x and u from q (Can remove for speed up later if necessary)
    x = [x0]
    u = []
    step = m+n
    for k in xrange(H):
        u_start = k*step
        u_stop = u_start+m
        x_start = u_stop
        x_stop = x_start+n
        u.append(np.asarray(q[u_start:u_stop]))
        x.append(np.asarray(q[x_start:x_stop]))

    # Evaluate f_dyn(x[k], u[k], xtra) for all pairs and append to x_hat
    x_hat = [x0] # Prepend x_hat with known x0
    for k in xrange(H):
        # Should this be x_hat or x?
        # NOTE: This could have serious consequences wrt the solver
        y_hat = dyn_model.predict(x[k], u[k], xtra)
        x_hat.append(y_hat[:])

    print 'x_hat =', np.asarray(x_hat)
    print 'x =', np.asarray(x)
    # Dynamics constraints
    score = 0.0
    for k in xrange(H):
        sub_score = sum(abs(x[k+1] - x_hat[k+1]))
        # print 'sum(abs(x[k+1] - x_hat[k+1])) = ', sub_score
        score += sub_score
    score *= lambda_dynamics # Scale dynamics constraints
    # print 'Score constraints = ', score
    # Goal trajectory constraints
    x_d_length = len(x_d[0])
    score_d = 0
    for k in xrange(H):
        sub_score = sum(abs(x_d[k+1] - x_hat[k+1][0:x_d_length]))
        # print 'sum(abs(x_d[k+1] - x_hat[k+1])) = ', sub_score
        score_d += sub_score
    score += score_d
    # print 'Score desired = ', score_d
    # print 'Total Score = ', score
    return score

def pushMPCObjectiveGradient(q, H, n, m, x0, x_d, xtra, lambda_dynamics, dyn_model):
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
        self.n = 5 # Predicted state space dimension
        self.m = 2 # Control space dimension
        self.u_max = u_max
        bounds_k = []
        for i in xrange(self.m):
            bounds_k.append((-u_max, u_max))
        for i in xrange(self.n):
            bounds_k.append((None, None))
        self.opt_bounds = []
        for i in xrange(self.H):
            self.opt_bounds.extend(bounds_k)
        self.lambda_dynamics = lambda_dynamics

    def feedbackControl(self, cur_state, ee_pose, x_d, cur_u):
        # TODO: Get initial guess from cur_state and trajectory...
        x0 = np.asarray([cur_state.x.x, cur_state.x.y, cur_state.x.theta,
                         ee_pose.pose.position.x, ee_pose.pose.position.y])
        q0 = []
        for i in xrange(self.H):
            q0.extend([self.u_max, self.u_max,
                       x_d[i+1][0], x_d[i+1][1], x_d[i+1][2], x_d[i+1][0] - 0.2, x_d[i+1][1] - 0.2])
        print 'x0 = ', np.asarray(x0)
        print 'q0 = ', np.asarray(q0)
        # print 'self.opt_bounds = ', np.asarray(self.opt_bounds)
        # print 'len(x0): ', len(x0)
        # print 'len(q0): ', len(q0)
        # print 'len(self.opt_bounds): ', len(self.opt_bounds)
        # Perform optimization
        xtra = []
        opt_args = (self.H, self.n, self.m, x0, x_d, xtra, self.lambda_dynamics, self.dyn_model)
        q_star, opt_val, d_info = opt.fmin_l_bfgs_b(func = pushMPCObjectiveFunction,
                                                    x0 = q0,
                                                    approx_grad = True,
                                                    # fprime = pushMPCObjectiveGradient,
                                                    args = opt_args,
                                                    bounds = self.opt_bounds)

        # TODO: Pull this into a new function, convert q vector to state
        u_star = q_star[self.m:self.m+self.n]
        print 'q_star =', q_star
        print 'opt_val =', opt_val
        u = TwistStamped()
        u.header.frame_id = 'torso_lift_link'
        # u.header.stamp = rospy.Time.now()
        u.twist.linear.z = 0.0
        u.twist.angular.x = 0.0
        u.twist.angular.y = 0.0
        u.twist.angular.z = 0.0
        u.twist.linear.x = u_star[0]
        u.twist.linear.x = u_star[1]
        return u

    def transform_state_to_vector(self, cur_state, ee_pose, u=None):
        q = [cur_state.x.x, cur_state.x.y, cur_state.x.theta,
             ee_pose.pose.position.x, ee_pose.pose.position.y]
        if u is not None:
            q.extend([u.twist.linear.x, u.twist.linear.y])
        return q

    def transform_vector_to_state(self, q):
        x = Pose2D()
        x.x = q[0]
        x.y = q[1]
        x.theta = q[2]
        ee = Pose()
        u = TwistStamped
        u.header.frame_id = 'torso_lift_link'
        u.header.stamp = rospy.Time.now()
        u.twist.linear.z = 0.0
        u.twist.angular.x = 0.0
        u.twist.angular.y = 0.0
        u.twist.angular.z = 0.0
        u.twist.linear.x = q[self.m]
        u.twist.linear.x = q[self.m + 1]
        return (x, ee, u)


class NaiveInputDynamics:
    def __init__(self, delta_t, n, m):
        self.delta_t = delta_t
        self.A = np.eye(n)
        self.B = np.zeros((n, m))
        self.B[0:m,0:m] = np.eye(m)*delta_t
        self.B[3:3+m,0:m] = np.eye(m)*delta_t
        print 'A=',self.A
        print 'B=',self.B

    def predict(self, x_k, u_k, xtra=[]):
        '''
        Predict the next state given current state estimates and control input
        x_k - current state estimate (list)
        u_k - current control to evaluate
        xtra - other features for SVM
        '''
        # Convert to simple linear format
        delta_x = u_k[0]*self.delta_t
        delta_y = u_k[1]*self.delta_t
        y_hat = x_k[:]
        # Update object pose
        y_hat[0] = x_k[0] + delta_x
        y_hat[1] = x_k[1] + delta_y
        y_hat[2] = x_k[2] # (no rotation)
        # Update hand pose
        # TODO: Add transformation into object frame
        y_hat[3] = x_k[3] + delta_x
        y_hat[4] = x_k[4] + delta_y
        return y_hat

class SVRPushDynamics:
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

def test_svm_stuff():
    aff_file_name  = sys.argv[1]
    plio = push_learning.CombinedPushLearnControlIO()
    plio.read_in_data_file(aff_file_name)
    svm_dynamics = SVRPushDynamics()
    svm_dynamics.learn_model(plio.push_trials)
    base_path = '/u/thermans/data/svm_dyn/'
    output_paths = []
    output_paths.append(base_path+'delta_x_dyn.model')
    output_paths.append(base_path+'delta_y_dyn.model')
    output_paths.append(base_path+'delta_theta_dyn.model')
    svm_dynamics.save_models(output_paths)
    svm_dynamics2 = SVRPushDynamics(svm_file_names=output_paths)

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

import push_trajectory_generator as ptg

def test_mpc():
    delta_t = 2.0
    H=5
    n = 5
    m = 2
    u_max=0.5
    lambda_dynamics=100.0

    cur_state = VisFeedbackPushTrackingFeedback()
    cur_state.x.x = 0.2
    cur_state.x.y = 0.0
    cur_state.x.theta = pi*0.5
    ee_pose = PoseStamped()
    ee_pose.pose.position.x = cur_state.x.x - 0.2
    ee_pose.pose.position.y = cur_state.x.y - 0.2
    cur_u = TwistStamped()
    cur_u.twist.linear.x = u_max
    cur_u.twist.linear.y = 0.0

    goal_loc = Pose2D()
    goal_loc.x = 2.0
    goal_loc.y = 0.0

    trajectory_generator = ptg.StraightLineTrajectoryGenerator()
    x_d = trajectory_generator.generate_trajectory(H, cur_state.x, goal_loc)
    dyn_model = NaiveInputDynamics(delta_t, n, m)

    # TODO: Generate feasible solution using known model
    print 'H = ', H
    print 'delta_t = ', delta_t
    print 'u_max = ', u_max
    print 'max displacement = ', delta_t*u_max
    print 'Total max displacement = ', delta_t*u_max*H
    print 'x_d = ', np.asarray(x_d)
    # TODO: Get initial guess from cur_state and trajectory...
    # TODO: Remove inital pose from decision vector
    x0 = np.asarray([cur_state.x.x, cur_state.x.y, cur_state.x.theta,
                     ee_pose.pose.position.x, ee_pose.pose.position.y])
    q0 = []
    for i in xrange(H):
        q0.extend([u_max, u_max, x_d[i+1][0], x_d[i+1][1], x_d[i+1][2], x_d[i+1][0] - 0.2, x_d[i+1][1] - 0.2])
    print 'x0 = ', x0
    print 'q0 = ', np.asarray(q0)
    xtra = []
    pushMPCObjectiveFunction(q0, H, n, m, x0, x_d, xtra, lambda_dynamics, dyn_model)

    # Construct and run MPC
    # mpc =  ModelPredictiveController(dyn_model, H, u_max, lambda_dynamics)
    # u_star = mpc.feedbackControl(cur_state, ee_pose, x_d, cur_u)

if __name__ == '__main__':
    # test_svm_stuff()
    test_mpc()
