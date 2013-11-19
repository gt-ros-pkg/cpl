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


def plot_desired_vs_controlled(q_star, X_d, x0, H, n, m, show_plot=True, suffix=''):
    X,U =  get_x_u_from_q(q_star, x0, H, n, m)
    plotter.figure()
    # Plot desired
    x_d = [X_d_k[0] for X_d_k in X_d]
    y_d = [X_d_k[1] for X_d_k in X_d]
    theta_d = [X_d_k[2] for X_d_k in X_d]
    plotter.plot(x_d, y_d, 'r')
    plotter.plot(x_d, y_d, 'ro')

    # Plot predicted
    x_hat = [X_k[0] for X_k in X]
    y_hat = [X_k[1] for X_k in X]
    theta_hat = [X_k[1] for X_k in X]
    plotter.plot(x_hat, y_hat,'b')
    plotter.plot(x_hat, y_hat,'b+')

    ax = plotter.gca()
    ax.set_xlim(0.0, 2.5)
    ax.set_ylim(-2.5, 2.5)
    plotter.title('Desired (Red) and Predicted (Blue) Trajectories '+suffix)
    plotter.xlabel('x (meters)')
    plotter.ylabel('y (meters)')
    if show_plot:
        plotter.show()

class ModelPredictiveController:
    def __init__(self, model, H=5, u_max=1.0):
        '''
        model - prediction function for use inside the optimizer
        H - the lookahead horizon for MPC
        u_max - maximum allowed velocity
        '''
        self.dyn_model = model
        self.H = H # Time horizon
        self.n = 5 # Predicted state space dimension
        self.m = 2 # Control space dimension
        self.u_max = u_max
        self.max_iter = 1000 # Max number of iterations
        self.ftol = 1.0E-6 # Accuracy of answer
        self.epsilon = sqrt(finfo(float).eps)
        bounds_k = []
        for i in xrange(self.m):
            bounds_k.append((-u_max, u_max))
        for i in xrange(self.n):
            bounds_k.append((-1.0E12,1.0E12))
        self.opt_bounds = []
        for i in xrange(self.H):
            self.opt_bounds.extend(bounds_k)
        self.opt_options = {'iter':self.max_iter,
                            'acc':self.ftol,
                            'iprint':1,
                            'disp':True,
                            'epsilon':self.epsilon,
                            'bounds':self.opt_bounds,
                            'full_output':True}

    def feedbackControl(self, cur_state, ee_pose, x_d, cur_u):
        x0 = np.asarray([cur_state.x.x, cur_state.x.y, cur_state.x.theta,
                         ee_pose.pose.position.x, ee_pose.pose.position.y])
        # TODO: Get initial guess at U from cur_state and trajectory (or at least goal)
        U_init = self.get_U_init(x0, x_d)
        xtra = []
        q0 = self.get_q0(x0, U_init, xtra)
        print 'x0 = ', np.asarray(x0)
        print 'q0 = ', np.asarray(q0)
        # TODO: Move as much of this as possible to the constructor, only updated what's needed at each callback
        opt_args = (self.H, self.n, self.m, x0, x_d, xtra, self.dyn_model)
        # Perform optimization
        res = opt.fmin_slsqp(pushMPCObjectiveFunction, q0, fprime = pushMPCObjectiveGradient,
                             f_eqcons = pushMPCConstraints, fprime_eqcons = pushMPCConstraintsGradients,
                             args = opt_args, **self.opt_options)
        q_star = res[0]
        opt_val = res[1]

        print 'opt_val =', opt_val,'\n'
        print 'q_star =', q_star

        plot_desired_vs_controlled(q0, x_d, x0, self.H, self.n, self.m, show_plot=False, suffix='q0')
        plot_desired_vs_controlled(q_star, x_d, x0, self.H, self.n, self.m, suffix='q*')
        return self.q_result_to_control_command(q_star)

    def q_result_to_control_command(self, q_star):
        u = TwistStamped()
        u.header.frame_id = 'torso_lift_link'
        # u.header.stamp = rospy.Time.now()
        u.twist.linear.z = 0.0
        u.twist.angular.x = 0.0
        u.twist.angular.y = 0.0
        u.twist.angular.z = 0.0
        u.twist.linear.x = q_star[0]
        u.twist.linear.x = q_star[1]
        return u

    def get_U_init(self, x0, x_d):
        U_init = []
        for k in xrange(self.H):
            if k % 2 == 0:
                u_x = 0.0
            else:
                u_x = self.u_max
            if k / 3 == 0:
                u_y = 0.0
            elif k / 3 == 1:
                u_y = -self.u_max
            else:
                u_y = self.u_max
            # u_x = 0.9*self.u_max
            # u_y = 0.0
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
    H = 10
    n = 5
    m = 2
    u_max = 0.5

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

    mpc =  ModelPredictiveController(dyn_model, H, u_max)
    u_star = mpc.feedbackControl(cur_state, ee_pose, x_d, cur_u)

    # print 'H = ', H
    # print 'delta_t = ', delta_t
    # print 'u_max = ', u_max
    # print 'max displacement = ', delta_t*u_max
    # print 'Total max displacement = ', delta_t*u_max*H
    # print 'x_d = ', np.array(x_d)
    # x0 = np.array([cur_state.x.x, cur_state.x.y, cur_state.x.theta,
    #                ee_pose.pose.position.x, ee_pose.pose.position.y])
    # xtra = []

    # Get initial guess from cur_state and control trajectory...
    # U_init = []
    # for k in xrange(H):
    #     U_init.append(np.array([0.9*u_max, -0.9*u_max]))
    # q0 = mpc.get_q0(x0, U_init, xtra)
    # print 'x0 = ', x0
    # print 'q0 = ', np.asarray(q0)

    # cost = pushMPCObjectiveFunction(q0, H, n, m, x0, x_d, xtra, dyn_model)
    # print 'Cost =', cost
    # cost_grad = pushMPCObjectiveGradient(q0, H, n, m, x0, x_d, xtra, dyn_model)
    # print 'Cost gradient =', cost_grad
    # constraints = pushMPCConstraints(q0, H, n, m, x0, x_d, xtra, dyn_model)
    # print 'Constraints =', constraints
    # constraint_jacobian = pushMPCConstraintsGradients(q0, H, n, m, x0, x_d, xtra, dyn_model)
    # print 'Constraint Jacobian =', constraint_jacobian
    # return constraint_jacobian

if __name__ == '__main__':
    # test_svm_stuff()
    test_mpc()
