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

def get_x_u_from_q(q, x0, H, n, m):
    x = [x0]
    u = []
    step = m+n
    score = 0
    for k in xrange(H):
        u_start = k*step
        u_stop = u_start + m
        x_start = u_stop
        x_stop = x_start + n
        u.append(np.asarray(q[u_start:u_stop]))
        x.append(np.asarray(q[x_start:x_stop]))
    return (x,u)

def pushMPCObjectiveFunction(q, H, n, m, x0, x_d, xtra, dyn_model):
    '''
    q - decision variable vector of the form (U[0], X[1], U[1],..., U[H-1], X[H])
    H - time horizon
    n - system state dimension
    m - control dimension
    x0 - inital state
    x_d - desired trajectory
    xtra - other features for dynamics prediction
    dyn_model - model simulating system dynamics of the form x[k+1] = dyn_model.predict(x[k], u[k], xtra)
    '''
    step = m+n
    x_d_length = len(x_d[0])
    cost = 0
    for k in xrange(H):
        x_start = m+k*step
        x_stop = x_start+x_d_length
        # Pull out x from q
        x_k_plus_1 = np.asarray(q[x_start:x_stop])
        # Compute sum of squared error on current trajectory time step
        cost = sum((x_k_plus_1 - x_d[k+1])**2)
    return cost

def pushMPCObjectiveGradient(q, H, n, m, x0, x_d, xtra, dyn_model):
    gradient = np.zeros(len(q))
    step = m+n
    x_d_length = len(x_d[0])
    score = 0
    for k in xrange(H):
        x_i_start = m+k*step
        for j, i in enumerate(range(x_i_start, x_i_start+x_d_length)):
            gradient[i] = 2.0*(q[i]-x_d[k+1][j])
    return gradient

def pushMPCConstraints(q, H, n, m, x0, x_d, xtra, dyn_model):
    x, u = get_x_u_from_q(q, x0, H, n, m)
    G = []
    for k in xrange(H):
        G.extend(dyn_model.predict(x[k], u[k], xtra) - x[k+1])
    return np.array(G)

def pushMPCConstraintsGradients(q, H, n, m, x0, x_d, xtra, dyn_model):
    x, u = get_x_u_from_q(q, x0, H, n, m)

    # Build Jacobian of the constraints
    num_constraints = H*n
    J = np.zeros((num_constraints, len(q)))

    # Ignore partials for x0 on the initial time step, since this is fixed
    J_f_k0 = dyn_model.jacobian(x[0], u[0])
    k_plus_1_constraint = np.eye(n)*-1.0
    J[0:n, 0:m] = J_f_k0[:,n:n+m]
    J[0:n, m:m+n] = k_plus_1_constraint

    # Setup partials for x[1], u[1], ..., x[H-1], u[H-1], x[H]
    for k in range(1,H):
        row_start = k*n
        row_stop = row_start+n
        col_start = m+(n+m)*(k-1)
        col_stop = col_start+m+n
        # Each of the H row blocks have the form f(x[k], u[k]) - x[k+1] (n dimensions)
        # Get jacobian for f(x[k], u[k])
        J_f_k = dyn_model.jacobian(x[k], u[k])
        J[row_start:row_stop, col_start:col_stop] = J_f_k
        # TODO: Set derivative for x[k+1]
        col_start = col_stop
        col_stop = col_start + n
        J[row_start:row_stop, col_start:col_stop] = k_plus_1_constraint

    return J

class ModelPredictiveController:
    def __init__(self, model, H=5, u_max=1.0, delta_t=1.0):
        '''
        model - prediction function for use inside the optimizer
        H - the lookahead horizon for MPC
        u_max - maximum allowed velocity
        '''
        self.init_from_previous = False
        self.dyn_model = model
        self.H = H # Time horizon
        self.n = 5 # Predicted state space dimension
        self.m = 2 # Control space dimension
        self.u_max = u_max
        self.delta_t = delta_t
        self.max_iter = 1000 # Max number of iterations
        self.ftol = 1.0E-5 # Accuracy of answer
        self.epsilon = sqrt(finfo(float).eps)
        self.opt_options = {'iter':self.max_iter,
                            'acc':self.ftol,
                            'iprint':1,
                            'epsilon':self.epsilon,
                            'full_output':True}
        self.regenerate_bounds()

    def regenerate_bounds(self):
        bounds_k = []
        for i in xrange(self.m):
            bounds_k.append((-self.u_max, self.u_max))
        for i in xrange(self.n):
            bounds_k.append((-1.0E12,1.0E12))
        self.opt_bounds = []
        for i in xrange(self.H):
            self.opt_bounds.extend(bounds_k)
        self.opt_options['bounds'] = self.opt_bounds

    def feedbackControl(self, x0, x_d, xtra = []):
        if self.init_from_previous:
            q0 = self.init_q0_from_previous(x_d, xtra)
        else:
            U_init = self.get_U_init(x0, x_d)
            q0 = self.get_q0(x0, U_init, xtra)

        opt_args = (self.H, self.n, self.m, x0, x_d, xtra, self.dyn_model)
        # Perform optimization
        res = opt.fmin_slsqp(pushMPCObjectiveFunction, q0, fprime = pushMPCObjectiveGradient,
                             f_eqcons = pushMPCConstraints, fprime_eqcons = pushMPCConstraintsGradients,
                             args = opt_args, **self.opt_options)
        q_star = res[0]
        self.q_star_prev = q_star[:]
        opt_val = res[1]

        return q_star

    def get_U_init(self, x0, x_d):
        # TODO: Get initial guess at U from cur_state and trajectory (or at least goal)
        U_init = []
        for k in xrange(self.H):
            if k % 2 == 0:
                u_x = self.u_max*0.01
            else:
                u_x = self.u_max*0.4
            if k / 3 == 0:
                u_y = self.u_max*0.2
            elif k / 3 == 1:
                u_y = -self.u_max*0.6
            else:
                u_y = self.u_max*0.5
            U_init.append(np.array([u_x, u_y]))
        return U_init

    def get_q0(self, x0, U, xtra):
        q0 = []
        x_k = x0
        for k in xrange(self.H):
            u_k = U[k]
            x_k_plus_1 = self.dyn_model.predict(x_k, u_k, xtra)
            for u_i in u_k:
                q0.append(u_i)
            for x_i in x_k_plus_1:
                q0.append(x_i)
            x_k = x_k_plus_1
        return np.array(q0)

    def init_q0_from_previous(self, x_d, xtra):
        # Remove initial control and loc from previous solution
        N = self.n+self.m
        q0 = self.q_star_prev[N:]
        k0 = len(q0)/N
        N_to_add = self.H - k0
        # Add the necessary number of more controls and locs to get to H tuples
        if N_to_add > 0:
            for k in xrange(N_to_add):
                x_k = q0[-self.n:]
                # Initialize next control assuming straight line motion between via points
                deltaX = x_d[k0+k+1] - x_d[k0+k]
                next_u = np.array([deltaX[0]/self.delta_t, deltaX[1]/self.delta_t])
                q0 = np.concatenate([q0, next_u])
                # Use dynamics to add next location
                q0 = np.concatenate([q0, self.dyn_model.predict(x_k, next_u, xtra)])
        else:
            # Remove the necessary number of more controls and locs to get to H tuples
            q0 = q0[:N*self.H]
        return np.array(q0)

class NaiveInputDynamics:
    def __init__(self, delta_t, n, m):
        self.delta_t = delta_t
        self.A = np.eye(n)
        self.B = np.zeros((n, m))
        self.B[0:m,0:m] = np.eye(m)*delta_t
        self.B[3:3+m,0:m] = np.eye(m)*delta_t
        self.J = np.concatenate((self.A, self.B), axis=1)

    def predict(self, x_k, u_k, xtra=[]):
        '''
        Predict the next state given current state estimates and control input
        x_k - current state estimate (ndarray)
        u_k - current control to evaluate (ndarray)
        xtra - other features for SVM
        '''
        x_k_plus_1 = np.array(self.A*np.matrix(x_k).T+self.B*np.matrix(u_k).T).ravel()
        return x_k_plus_1

    def jacobian(self, x_k, u_k, xtra=[]):
        '''
        Compute the Jacobian of the prediciton function
        x_k - current state estimate (ndarray)
        u_k - current control to evaluate (ndarray)
        xtra - other features for SVM
        returns Jacobian with columns ordered [x, u]
        '''
        return self.J

class StochasticNaiveInputDynamics:
    def __init__(self, delta_t, n, m, sigma):
        self.delta_t = delta_t
        self.A = np.eye(n)
        self.B = np.zeros((n, m))
        self.B[0:m,0:m] = np.eye(m)*delta_t
        self.B[3:3+m,0:m] = np.eye(m)*delta_t
        self.J = np.concatenate((self.A, self.B), axis=1)
        self.sigma = sigma
        self.n = n
        self.m = m

    def predict(self, x_k, u_k, xtra=[]):
        '''
        Predict the next state given current state estimates and control input
        x_k - current state estimate (ndarray)
        u_k - current control to evaluate (ndarray)
        xtra - other features for SVM
        '''
        C = np.random.normal(0.0, self.sigma, self.m)
        u_s = u_k+C
        x_k_plus_1 = np.array(self.A*np.matrix(x_k).T+self.B*np.matrix(u_s).T).ravel()
        return x_k_plus_1

    def jacobian(self, x_k, u_k, xtra=[]):
        '''
        Compute the Jacobian of the prediciton function
        x_k - current state estimate (ndarray)
        u_k - current control to evaluate (ndarray)
        xtra - other features for SVM
        returns Jacobian with columns ordered [x, u]
        '''
        return self.J

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

def plot_desired_vs_controlled(q_star, X_d, x0, n, m, show_plot=True, suffix='', t=0, out_path=''):
    H = len(q_star)/(n+m)
    X,U =  get_x_u_from_q(q_star, x0, H, n, m)
    plotter.figure()
    # Plot desired
    x_d = [X_d_k[0] for X_d_k in X_d]
    y_d = [X_d_k[1] for X_d_k in X_d]
    theta_d = [X_d_k[2] for X_d_k in X_d]
    plotter.plot(x_d, y_d, 'r')
    plotter.plot(x_d, y_d, 'ro')

    # Plot predicted
    x_hat = [X_k[0] for X_k in X[t:]]
    y_hat = [X_k[1] for X_k in X[t:]]
    theta_hat = [X_k[1] for X_k in X[t:]]
    plotter.plot(x_hat, y_hat,'b')
    plotter.plot(x_hat, y_hat,'b+')

    # Plot observed / GT
    x_gt = [X_k[0] for X_k in X[:t+1]]
    y_gt = [X_k[1] for X_k in X[:t+1]]
    theta_gt = [X_k[1] for X_k in X[:t+1]]
    plotter.plot(x_gt, y_gt,'g')
    plotter.plot(x_gt, y_gt,'g+')

    plot_title = 'MPC_Push_Control_Trajectory'+suffix
    plotter.title(plot_title)
    plotter.xlabel('x (meters)')
    plotter.ylabel('y (meters)')
    if len(out_path) > 0:
        plotter.savefig(out_path+plot_title+'.png')
    if show_plot:
        plotter.show()

def plot_controls(q_star, X_d, x0, n, m, u_max, show_plot=True, suffix='', t=0, out_path=''):
    H = len(q_star)/(n+m)
    X,U =  get_x_u_from_q(q_star, x0, H, n, m)

    Ux = [u_k[0] for u_k in U]
    Uy = [u_k[1] for u_k in U]

    plotter.figure()
    plotter.plot(Ux, 'r')
    plotter.plot(Ux, 'ro')

    plotter.plot(Uy, 'b')
    plotter.plot(Uy, 'b+')
    plotter.plot(np.ones(H)*u_max, 'g')
    plotter.plot(np.ones(H)*-u_max, 'g')

    plot_title = 'MPC_Push_Control_Inputs'+suffix
    plotter.title(plot_title)
    ax = plotter.gca()
    ax.set_ylim(-1.1*u_max, 1.1*u_max)
    plotter.xlabel('Time Step')
    plotter.ylabel('U (meters/sec)')
    if len(out_path) > 0:
        plotter.savefig(out_path+plot_title+'.png')
    if show_plot:
        plotter.show()

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
