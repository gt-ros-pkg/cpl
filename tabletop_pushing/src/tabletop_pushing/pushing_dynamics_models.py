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

import svmutil
import numpy as np
from math import sin, cos

# TODO: Put dict here to easily get state name to index

class NaiveInputDynamics:
    def __init__(self, delta_t, n, m):
        '''
        delta_t - the (average) time between time steps
        n - number of state space dimensions
        m - number of control space dimensions
        '''
        self.delta_t = delta_t
        self.A = np.eye(n)
        self.B = np.zeros((n, m))
        self.B[0:m,0:m] = np.eye(m)*delta_t
        self.B[3:3+m,0:m] = np.eye(m)*delta_t
        self.J = np.concatenate((self.A, self.B), axis=1)
        self.n = n
        self.m = m

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
        '''
        delta_t - the (average) time between time steps
        n - number of state space dimensions
        m - number of control space dimensions
        sigma - Gaussian noise standard dev to be added on control inputs
        '''
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
    def __init__(self, delta_t, n, m, svm_file_names=None, epsilons=None, kernel_type=None, learned_out_dims=3,
                 object_frame_feats=False):
        '''
        delta_t - the (average) time between time steps
        n - number of state space dimensions
        m - number of control space dimensions
        svm_file_names - list of svm model file names to read from disk
        epsilons - list of epislon values for the epislon insensitive loss function (training only)
        kernel_type - type of kernel to use for traning, can be any of the self.KERNEL_TYPES keys (training only)
        o - number of output dimensions in the model (training only)
        '''
        self.delta_t = delta_t
        self.n = n
        self.m = m

        if epsilons is not None:
            self.epsilons = epsilons
        else:
            self.epsilons = []
            for i in xrange(learned_out_dims):
                self.epsilons.append(1e-6)

        self.KERNEL_TYPES = {'linear': 0, 'polynomial': 1, 'RBF': 2, 'sigmoid': 3, 'precomputed':4}
        if kernel_type is not None:
            self.kernel_type = kernel_type
        else:
            self.kernel_type = 'linear'

        # NOTE: Make switches to change based on preferences here
        self.predict = self.predict_linear_hand
        self.jacobian = self.jacobian_linear_hand
        self.build_jacobian = self.build_jacobian_linear_hand
        self.transform_opt_vector_to_feat_vector = self.opt_vector_to_feats_state_control
        self.test_batch_data = self.test_batch_data_linear_hand

        if object_frame_feats:
            self.transform_opt_vector_to_feat_vector = self.opt_vector_to_feats_object_frame
            self.jacobian = self.jacobian_linear_hand_object_frame
            self.build_jacobian = self.build_jacobian_linear_hand_object_frame
            # TODO: Implement this method
            self.test_batch_data = self.test_batch_data_linear_hand_object_frame
            self.p = 4 # Num feature vector elements

        self.svm_models = []
        if svm_file_names is not None:
            for svm_file_name in svm_file_names:
                # print 'Loading file', svm_file_name
                self.svm_models.append(svmutil.svm_load_model(svm_file_name))
            self.build_jacobian()

    def learn_model(self, learn_data):
        '''
        Learns SVMs to predict dynamics given the input data
        learn_data - List of PushCtrlTrial() defined in push_learning.py
        '''
        X = []
        Y = []
        for trial in learn_data:
            (x, y) = self.transform_trial_data_to_feat_vectors(trial.trial_trajectory)
            X.extend(x)
            Y.extend(y)

        for i in xrange(len(Y[0])):
            Y_i = []
            for y in Y:
                Y_i.append(y[i])
            param_string = '-s 3 -t ' + str(self.KERNEL_TYPES[self.kernel_type]) + ' -p ' + str(self.epsilons[i])
            # TODO: Kernel specific options
            svm_model = svmutil.svm_train(Y_i, X, param_string)
            self.svm_models.append(svm_model)
        self.build_jacobian()

    def save_models(self, output_file_names):
        '''
        Write svm models to disk
        output_file_names - list of file paths for saving to, assumes length = len(self.svm_models)
        '''
        if len(self.svm_models) > 0:
            for file_name, model in zip(output_file_names, self.svm_models):
                svmutil.svm_save_model(file_name, model)

    def test_batch_data_linear_hand(self, test_data):
        X = []
        Y = []
        W = []
        for trial in test_data:
            (x, y) = self.transform_trial_data_to_feat_vectors(trial.trial_trajectory, True)
            X.extend(x)
            Y.extend(y)
        Y_hat = []
        Y_out = []
        for i, svm_model in enumerate(self.svm_models):
            Y_i = []
            for y in Y:
                Y_i.append(y[i])
            [Y_hat_i, _, _] = svmutil.svm_predict(Y_i, X, svm_model, '-q')
            Y_hat.append(Y_hat_i)
            Y_out.append(Y_i)
        # Get EE ground truth
        for i in range(3,5):
            Y_i = []
            for y in Y:
                Y_i.append(y[i])
            Y_out.append(Y_i)

        # Add EE predictions
        Y_hat_Xee = []
        Y_hat_Yee = []
        for x in X:
            Y_hat_Xee.append(self.delta_t*x[3])
            Y_hat_Yee.append(self.delta_t*x[4])

        Y_hat.append(Y_hat_Xee)
        Y_hat.append(Y_hat_Yee)

        return (Y_hat, Y_out, X)

    def transform_trial_data_to_feat_vectors(self, trajectory, EE_deltas = False):
        '''
        Get SVM feature vector from push trial trajectory state information.
        Needs to be updated whenever transform_state_data_to_feat_vector() is updated
        trajectory - list of ControlTimeStep() defined in push_learning.py
        '''
        Z = []
        Y = []
        W = []
        # TODO: Have ability to extract xtra from the data
        xtra = []
        for i in xrange(len(trajectory)-1):
            cts_t0 = trajectory[i]
            cts_t1 = trajectory[i+1]
            x_t = [cts_t0.x.x, cts_t0.x.y, cts_t0.x.theta,
                   cts_t0.ee.position.x, cts_t0.ee.position.y]
            u_t = [cts_t0.u.linear.x, cts_t0.u.linear.y]
            z_t = self.transform_opt_vector_to_feat_vector(x_t, u_t, xtra)
            y_t = [cts_t1.x.x - cts_t0.x.x, cts_t1.x.y - cts_t0.x.y, cts_t1.x.theta - cts_t0.x.theta]
            if EE_deltas:
                y_t.extend([cts_t1.ee.position.x - cts_t0.ee.position.x,
                            cts_t1.ee.position.y - cts_t0.ee.position.y])
            Z.extend(z_t)
            Y.append(y_t)
        return (Z, Y)

    #
    # Prediction and jacobian methods
    #
    def predict_linear_hand(self, x_k, u_k, xtra=[]):
        '''
        Predict the next state given current state estimates and control input
        x_k - current state estimate (ndarray)
        u_k - current control to evaluate (ndarray)
        xtra - other features for SVM
        '''
        z = self.transform_opt_vector_to_feat_vector(x_k, u_k, xtra)
        x_k_plus_1 = []
        y = [0]
        for i, svm_model in enumerate(self.svm_models):
            [delta_i, _, _] = svmutil.svm_predict(y, z, svm_model, '-q')
            x_k_plus_1.append(x_k[i]+delta_i[0])

        # X ee
        x_k_plus_1.append(x_k[3] + self.delta_t*u_k[0])
        # Y ee
        x_k_plus_1.append(x_k[4] + self.delta_t*u_k[1])

        return np.array(x_k_plus_1)

    def jacobian_linear_hand(self, x_k, u_k, xtra=[]):
        '''
        Return the matrix of partial derivatives of the dynamics model w.r.t. the current state and control
        x_k - current state estimate (ndarray)
        u_k - current control to evaluate (ndarray)
        xtra - other features for SVM
        '''
        return self.J

    def build_jacobian_linear_hand(self):
        self.J = np.eye(self.n, self.n+self.m)

        # Setup partials for hand position change, currently using linear model of applied velocity
        self.J[3:5, 5: ] = np.eye(self.m)*self.delta_t

        # Setup partials w.r.t. SVM model parameters
        for i, svm_model in enumerate(self.svm_models):
            alphas = svm_model.get_sv_coef()
            svs = svm_model.get_SV()
            # Partial derivative is the sum of the product of the SV elements and coefficients
            # \sum_{i=1}^{l} alpha_i*z_i^j
            for alpha, sv in zip(alphas, svs):
                for j in xrange(self.m+self.n):
                    self.J[i, j] += alpha[0]*sv[j+1]

    def jacobian_linear_hand_object_frame(self, x_k, u_k, xtra=[]):
        '''
        Return the matrix of partial derivatives of the dynamics model w.r.t. the current state and control
        x_k - current state estimate (ndarray)
        u_k - current control to evaluate (ndarray)
        xtra - other features for SVM
        '''
        x_ee_demeaned = x_k[3] - x_k[0]
        y_ee_demeaned = x_k[4] - x_k[1]
        st = sin(x_k[2])
        ct = cos(x_k[2])
        # Partial derivatives of the transformed features
        J_feats = np.matrix(np.zeros((self.p, self.n+self.m)))

        J_feats[0, 0] = -ct # d/dx[x_ee]
        J_feats[0, 1] = -st # d/dy[x_ee]
        J_feats[0, 2] =  st*x_ee_demeaned - ct*y_ee_demeaned # d/dtheta[x_ee]
        J_feats[0, 3] =  ct # d/dx_ee[x_ee]
        J_feats[0, 4] =  st # d/dy_ee[x_ee]

        J_feats[1, 0] =  st # d/dx[y_ee]
        J_feats[1, 1] = -ct # d/dy[y_ee]
        J_feats[1, 2] =  ct*x_ee_demeaned + st*y_ee_demeaned # d/dtheta[y_ee]
        J_feats[1, 3] = -st # d/dx_ee[y_ee]
        J_feats[1, 4] =  ct # d/dy_ee[y_ee]

        J_feats[2, 2] = st*u_k[0] - ct*u_k[1] # d/dtheta[u_x]
        J_feats[2, 5] = ct # d/du_x[u_x]
        J_feats[2, 6] = st # d/du_y[u_x]

        J_feats[3, 2] =  ct*u_k[0] + st*u_k[1] # d/dtheta[u_y]
        J_feats[3, 5] = -st # d/du_x[u_y]
        J_feats[3, 6] =  ct # d/du_y[u_y]

        # Do chain rule here
        J_update = np.zeros(self.J_base.shape)
        J_update[:len(self.svm_models),:] = self.sv_coeffs*J_feats
        J = self.J_base + J_update
        return J

    def build_jacobian_linear_hand_object_frame(self):
        self.J_base = np.eye(self.n, self.n+self.m)

        # Setup partials for hand position change, currently using linear model of applied velocity
        self.J_base[3:5, 5: ] = np.eye(self.m)*self.delta_t

        self.sv_coeffs = np.matrix(np.zeros((len(self.svm_models), self.p)))

        # Setup partials w.r.t. SVM model parameters
        for i, svm_model in enumerate(self.svm_models):
            alphas = svm_model.get_sv_coef()
            svs = svm_model.get_SV()
            # Partial derivative is the sum of the product of the SV elements and coefficients
            # \sum_{i=1}^{l} alpha_i*z_i^j
            for alpha, sv in zip(alphas, svs):
                for j in xrange(self.p):
                    self.sv_coeffs[i, j] += alpha[0]*sv[j+1]

    #
    # Features transforms
    #
    def opt_vector_to_feats_state_control(self, x_k, u_k, xtra):
        '''
        Does the simplest form of features, just passing on the decision variables
        '''
        z = np.concatenate([x_k, u_k, xtra])
        # SVM expects a list of lists, but we're only doing one instance at a time
        z = [z.tolist()]
        return z

    def opt_vector_to_feats_object_frame(self, x_k, u_k, xtra):
        '''
        Projects features into hand frame, ignores absolute coordinates
        '''
        # Remove mean
        x_ee_demeaned = np.matrix([[x_k[3]-x_k[0]],
                                   [x_k[4]-x_k[1]]])
        # Rotate into frame
        st = sin(x_k[2])
        ct = cos(x_k[2])
        R = np.matrix([[ct, st],
                       [-st, ct]])
        x_ee_obj = np.array(R*x_ee_demeaned).T.ravel()

        # Rotate u_k into frame
        u_k_obj = np.array(R*np.matrix(u_k).T).ravel()

        z = np.concatenate([x_ee_obj, u_k_obj, xtra])
        # SVM expects a list of lists, but we're only doing one instance at a time
        z = [z.tolist()]
        return z
