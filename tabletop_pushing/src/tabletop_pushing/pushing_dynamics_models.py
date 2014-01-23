#!/usr/bin/env python
# Software License Agreement (BSD License)
#
#  Copyright (c) 2014, Georgia Institute of Technology
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
from math import sin, cos, exp, tanh
import dynamics_learning
from util import subPIAngle
from sklearn.gaussian_process import GaussianProcess

_PARAM_FILE_SUFFIX = '_params.txt'
_PARAM_FEATURE_HEADER = 'FEATURE_NAMES'
_PARAM_TARGET_HEADER = 'TARGET_NAMES'

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
    # Setup constants
    _KERNEL_TYPES = {'LINEAR': 0, 'POLYNOMIAL': 1, 'RBF': 2, 'SIGMOID': 3, 'PRECOMPUTED':4}
    _KERNEL_NAMES = dict((v,k) for k,v in _KERNEL_TYPES.items())

    def __init__(self, delta_t = 1.0, n = 6, m = 3,
                 param_file_name = '', epsilons = None, kernel_type = None,
                 feature_names = [], target_names = [], xtra_names = []):
        '''
        delta_t - the (average) time between time steps
        n - number of state space dimensions
        m - number of control space dimensions
        param_file_name - base path and file name for parameter file
        epsilons - list of epislon values for the epislon insensitive loss function (training only)
        kernel_type - type of kernel to use for traning, can be any of the self.KERNEL_TYPES keys (training only)
        '''
        # Class paramters
        self.delta_t = delta_t
        self.n = n
        self.m = m
        self.svm_models = []
        self.kernel_types = []
        self.feature_names = []
        self.target_names = []
        self.jacobian_needs_updates = False

        # Get settings from disk if a file base is specified
        if len(param_file_name) > 0:
            self.load_models(param_file_name)
        else:
            # Custom initialization otherwise
            self.feature_names = feature_names
            self.target_names = target_names
            learned_out_dims = len(target_names)

            # Set kernel types
            # TODO: Make this variable between different dimensions
            if kernel_type is not None:
                for i in xrange(learned_out_dims):
                    self.kernel_types.append(kernel_type)
            else:
                for i in xrange(learned_out_dims):
                    self.kernel_types.append('LINEAR')

            # Setup loss function paramters
            if epsilons is not None:
                self.epsilons = epsilons
            else:
                self.epsilons = []
                for i in xrange(learned_out_dims):
                    self.epsilons.append(1e-6)

        # Set secondary parameters
        self.p = len(self.feature_names)
        self.obj_frame_ee_feats = (dynamics_learning._EE_X_OBJ in self.feature_names or
                                   dynamics_learning._EE_Y_OBJ in self.feature_names)
        self.obj_frame_u_feats  = (dynamics_learning._U_X_OBJ in self.feature_names or
                                   dynamics_learning._U_Y_OBJ in self.feature_names)
        self.use_naive_ee_model = not (dynamics_learning._DELTA_EE_X_WORLD in self.target_names or
                                       dynamics_learning._DELTA_EE_Y_WORLD in self.target_names or
                                       dynamics_learning._DELTA_EE_X_OBJ in self.target_names or
                                       dynamics_learning._DELTA_EE_Y_OBJ in self.target_names or
                                       dynamics_learning._DELTA_EE_PHI_WORLD in self.target_names)

        self.obj_frame_obj_targets = (dynamics_learning._DELTA_OBJ_X_OBJ in self.target_names or
                                      dynamics_learning._DELTA_OBJ_Y_OBJ in self.target_names)
        self.obj_frame_ee_targets = (dynamics_learning._DELTA_EE_X_OBJ in self.target_names or
                                     dynamics_learning._DELTA_EE_Y_OBJ in self.target_names)

        # Optimization vector variables [Fixed outside of scope here]
        self.obj_x_opt_idx = 0
        self.obj_y_opt_idx = 1
        self.obj_theta_opt_idx = 2
        self.ee_x_opt_idx = 3
        self.ee_y_opt_idx = 4
        self.ee_phi_opt_idx = 5
        self.u_phi_opt_idx = 8
        self.u_x_opt_idx = 6
        self.u_y_opt_idx = 7

        # Set all indices for Jacobian junk
        if self.obj_frame_obj_targets:
            self.obj_x_target_idx = self.target_names.index(dynamics_learning._DELTA_OBJ_X_OBJ)
            self.obj_y_target_idx = self.target_names.index(dynamics_learning._DELTA_OBJ_Y_OBJ)
        else:
            self.obj_x_target_idx = self.target_names.index(dynamics_learning._DELTA_OBJ_X_WORLD)
            self.obj_y_target_idx = self.target_names.index(dynamics_learning._DELTA_OBJ_Y_WORLD)

        self.obj_theta_target_idx = self.target_names.index(dynamics_learning._DELTA_OBJ_THETA_WORLD)

        if self.obj_frame_ee_targets:
            self.ee_x_target_idx = self.target_names.index(dynamics_learning._DELTA_EE_X_OBJ)
            self.ee_y_target_idx = self.target_names.index(dynamics_learning._DELTA_EE_Y_OBJ)
            self.ee_phi_target_idx = self.target_names.index(dynamics_learning._DELTA_EE_PHI_WORLD)
        elif self.use_naive_ee_model:
            self.ee_x_target_idx = 3
            self.ee_y_target_idx = 4
            self.ee_phi_target_idx = 5
        else:
            self.ee_x_target_idx = self.target_names.index(dynamics_learning._DELTA_EE_X_WORLD)
            self.ee_y_target_idx = self.target_names.index(dynamics_learning._DELTA_EE_Y_WORLD)
            self.ee_phi_target_idx = self.target_names.index(dynamics_learning._DELTA_EE_PHI_WORLD)


        if self.obj_frame_ee_feats:
            self.ee_x_feat_idx = self.feature_names.index(dynamics_learning._EE_X_OBJ)
            self.ee_y_feat_idx = self.feature_names.index(dynamics_learning._EE_Y_OBJ)
            self.ee_phi_feat_idx = self.feature_names.index(dynamics_learning._EE_PHI_OBJ)
        else:
            self.ee_x_feat_idx = self.feature_names.index(dynamics_learning._EE_X_WORLD)
            self.ee_y_feat_idx = self.feature_names.index(dynamics_learning._EE_Y_WORLD)
            self.ee_phi_feat_idx = self.feature_names.index(dynamics_learning._EE_PHI_WORLD)

        if self.obj_frame_u_feats:
            self.u_x_feat_idx = self.feature_names.index(dynamics_learning._U_X_OBJ)
            self.u_y_feat_idx = self.feature_names.index(dynamics_learning._U_Y_OBJ)
        else:
            self.u_x_feat_idx = self.feature_names.index(dynamics_learning._U_X_WORLD)
            self.u_y_feat_idx = self.feature_names.index(dynamics_learning._U_Y_WORLD)

        self.u_phi_feat_idx = self.feature_names.index(dynamics_learning._U_PHI_WORLD)

        self.build_jacobian()

    def learn_model(self, X_all, Y_all, kernel_params = {}):
        '''
        Learns SVMs to predict dynamics given the input data
        X_all - batch list of features
        Y_all - batch list of targets
        kernel_params - user specified dictionary of kernel specific parameters
        '''
        (X, Y) = self.select_feature_data_and_targets(X_all, Y_all)

        for i, Y_i in enumerate(Y):
            print 'Learning for target:', self.target_names[i]
            param_string = '-s 3 -t ' + str(self._KERNEL_TYPES[self.kernel_types[i]]) + ' -p ' + str(self.epsilons[i])
            if i in kernel_params:
                param_string += ' ' + kernel_params[i]
            # print 'Parameters', param_string
            svm_model = svmutil.svm_train(Y_i, X, param_string)
            self.svm_models.append(svm_model)
        self.build_jacobian()

    def test_batch_data(self, X_all, Y_all):
        (X, Y_out) = self.select_feature_data_and_targets(X_all, Y_all)
        Y_hat = []
        for Y_i, svm_model in zip(Y_out, self.svm_models):
            [Y_hat_i, _, _] = svmutil.svm_predict(Y_i, X, svm_model, '-q')
            Y_hat.append(Y_hat_i)
        return (Y_hat, Y_out, X)

    #
    # Prediction and jacobian methods
    #
    def predict(self, x_k, u_k, xtra=[]):
        '''
        Predict the next state given current state estimates and control input
        x_k - current state estimate (ndarray)
        u_k - current control to evaluate (ndarray)
        xtra - other features for SVM
        '''
        z = [self.transform_opt_vector_to_feat_vector(x_k, u_k, xtra)]
        y = [0]

        deltas = []

        for i, svm_model in enumerate(self.svm_models):
            [delta_i, _, _] = svmutil.svm_predict(y, z, svm_model, '-q')
            deltas.append(delta_i[0])

        return np.array(self.transform_svm_results_to_opt_vector(x_k, u_k, deltas))

    def jacobian(self, x_k, u_k, xtra=[]):
        '''
        Return the matrix of partial derivatives of the dynamics model w.r.t. the current state and control
        x_k - current state estimate (ndarray)
        u_k - current control to evaluate (ndarray)
        xtra - other features for SVM
        '''
        if self.jacobian_needs_updates:
            return self.update_jacobian(x_k, u_k, xtra)
        return self.J

    def build_jacobian(self):
        '''
        Setup the basic structure of the jacobian for faster computation at each iteration
        '''
        self.J_base = np.matrix(np.eye(self.n, self.n+self.m))
        self.J_delta_d_targets = np.matrix(np.zeros((self.n, self.n)))
        self.jacobian_needs_updates = False

        # Setup partials for ee position change, currently using linear model of applied velocity
        if self.use_naive_ee_model:
            self.J_base[3:self.n, self.n: ] = np.eye(self.m)*self.delta_t

        if (self.obj_frame_ee_targets or self.obj_frame_obj_targets or
            self.obj_frame_ee_feats or self.obj_frame_u_feats):
            self.jacobian_needs_updates = True

        # Setup static deltas d targets (currently none)
        J_delta_d_targets = np.matrix(np.eye(self.n, self.n))

        # Setup jacobain of learned targets w.r.t. features
        self.J_targets_d_feats = np.matrix(np.zeros((self.n, self.p)))
        self.rbf_deriv_prefixes = {}
        self.full_SVs = {}
        # Setup partials w.r.t. SVM model parameters
        for i, svm_model in enumerate(self.svm_models):
            alphas = svm_model.get_sv_coef()
            svs_sparse = svm_model.get_SV()
            svs = self.make_svs_dense(svs_sparse)
            self.full_SVs[i] = svs
            # print 'svm[',i, '] has kernel type', self._KERNEL_NAMES[svm_model.param.kernel_type]
            if self.kernel_types[i] == 'LINEAR':
                # Partial derivative is the sum of the product of the SV elements and coefficients
                # \sum_{i=1}^{l} alpha_i*z_i^j
                for alpha, sv in zip(alphas, self.full_SVs[i]):
                    for j in xrange(self.p):
                        self.J_targets_d_feats[i, j] += alpha[0]*sv[j]
            elif self.kernel_types[i] == 'RBF':
                gamma = svm_model.param.gamma
                self.rbf_deriv_prefixes[i] = []
                for alpha in alphas:
                    self.rbf_deriv_prefixes[i].append(-2.*alpha[0]*gamma)

        # Setup static feature d opts (currently none)
        J_feats_d_opts = np.matrix(np.eye(self.p, self.n+self.m))

        # print 'self.J_base:\n', self.J_base
        # print 'J_delta_d_targets:\n',J_delta_d_targets
        # print 'self.J_targets_d_feats:\n',self.J_targets_d_feats
        # print 'J_feats_d_opts:\n', J_feats_d_opts

        self.J = self.J_base + J_delta_d_targets*self.J_targets_d_feats*J_feats_d_opts

    def update_jacobian(self, x_k, u_k, xtra=[]):
        '''
        Return the matrix of partial derivatives of the dynamics model w.r.t. the current state and control
        x_k - current state estimate (ndarray)
        u_k - current control to evaluate (ndarray)
        xtra - other features for SVM
        '''
        # NOTE: Being lazy and assuming a specific order of features here...
        st = sin(x_k[self.obj_theta_opt_idx])
        ct = cos(x_k[self.obj_theta_opt_idx])

        # Update derivative of opt deltas w.r.t. learned targets
        J_delta_d_targets = np.matrix(np.eye(self.n, self.n))
        if self.obj_frame_obj_targets:
            J_delta_d_targets[self.obj_x_opt_idx, self.obj_x_target_idx] = ct # d/do_x^o[o_x^w]
            J_delta_d_targets[self.obj_x_opt_idx, self.obj_y_target_idx] = -st # d/do_y^o[o_x^w]
            J_delta_d_targets[self.obj_y_opt_idx, self.obj_x_target_idx] = st # d/do_x^o[o_y^w]
            J_delta_d_targets[self.obj_y_opt_idx, self.obj_y_target_idx] = ct # d/do_y^o[o_y^w]

        if self.obj_frame_ee_targets:
            J_delta_d_targets[self.ee_x_opt_idx, self.ee_x_target_idx] = ct # d/dee_x^o[ee_x^w]
            J_delta_d_targets[self.ee_x_opt_idx, self.ee_y_target_idx] = -st # d/dee_y^o[ee_x^w]
            J_delta_d_targets[self.ee_y_opt_idx, self.ee_x_target_idx] = st # d/dee_x^o[ee_y^w]
            J_delta_d_targets[self.ee_y_opt_idx, self.ee_y_target_idx] = ct # d/dee_y^o[ee_y^w]

        # Update derivatives of features w.r.t. decision variables
        J_feats_d_opts = np.matrix(np.zeros((self.p, self.n+self.m)))
        if self.obj_frame_ee_feats:
            x_ee_demeaned = x_k[self.ee_x_opt_idx] - x_k[self.obj_x_opt_idx]
            y_ee_demeaned = x_k[self.ee_y_opt_idx] - x_k[self.obj_y_opt_idx]

            J_feats_d_opts[self.ee_x_feat_idx, self.obj_x_opt_idx] = -ct # d/dx[ee_x^o]
            J_feats_d_opts[self.ee_x_feat_idx, self.obj_y_opt_idx] = -st # d/dy[ee_x^o]
            J_feats_d_opts[self.ee_x_feat_idx,
                           self.obj_theta_opt_idx] =  st*x_ee_demeaned - ct*y_ee_demeaned # d/dtheta[ee_x^o]
            J_feats_d_opts[self.ee_x_feat_idx, self.ee_x_opt_idx] =  ct # d/dee_x[ee_x^o]
            J_feats_d_opts[self.ee_x_feat_idx, self.ee_y_opt_idx] =  st # d/dee_y[ee_x^o]

            J_feats_d_opts[self.ee_y_feat_idx, self.obj_x_opt_idx] =  st # d/dx[ee_y^o]
            J_feats_d_opts[self.ee_y_feat_idx, self.obj_y_opt_idx] = -ct # d/dy[ee_y^o]
            J_feats_d_opts[self.ee_y_feat_idx,
                           self.obj_theta_opt_idx] =  ct*x_ee_demeaned + st*y_ee_demeaned # d/dtheta[ee_y^o]
            J_feats_d_opts[self.ee_y_feat_idx, self.ee_x_opt_idx] = -st # d/dx_ee[ee_y^o]
            J_feats_d_opts[self.ee_y_feat_idx, self.ee_y_opt_idx] =  ct # d/dy_ee[ee_y^o]

            # Setup ee_phi partials
            J_feats_d_opts[self.ee_phi_feat_idx, self.ee_phi_opt_idx] = 1.0 # d/dee_phi^w[ee_phi^o]
            J_feats_d_opts[self.ee_phi_feat_idx, self.obj_theta_opt_idx] =  -1.0 # d/dtheta[ee_phi^o]
        else:
            # Identity matrix
            J_feats_d_opts[self.ee_x_feat_idx, self.ee_x_opt_idx] = 1.
            J_feats_d_opts[self.ee_y_feat_idx, self.ee_y_opt_idx] = 1.
            J_feats_d_opts[self.ee_phi_feat_idx, self.ee_phi_opt_idx] = 1.

        if self.obj_frame_u_feats:
            J_feats_d_opts[self.u_x_feat_idx, self.obj_theta_opt_idx] = st*u_k[0] - ct*u_k[1] # d/dtheta[u_x]
            J_feats_d_opts[self.u_x_feat_idx, self.u_x_opt_idx] = ct # d/du_x[u_x]
            J_feats_d_opts[self.u_x_feat_idx, self.u_y_opt_idx] = st # d/du_y[u_x]

            J_feats_d_opts[self.u_y_feat_idx, self.obj_theta_opt_idx] =  ct*u_k[0] + st*u_k[1] # d/dtheta[u_y]
            J_feats_d_opts[self.u_y_feat_idx, self.u_x_opt_idx] = -st # d/du_x[u_y]
            J_feats_d_opts[self.u_y_feat_idx, self.u_y_opt_idx] =  ct # d/du_y[u_y]
        else:
            # Identity matrix
            J_feats_d_opts[self.u_x_feat_idx, self.u_x_opt_idx] = 1.
            J_feats_d_opts[self.u_y_feat_idx, self.u_y_opt_idx] = 1.
        # Setup derivates of u_phi
        J_feats_d_opts[self.u_phi_feat_idx, self.u_phi_opt_idx] = 1.

        # Needed for kernel derivative evaluation
        z_k = self.transform_opt_vector_to_feat_vector(x_k, u_k, xtra)

        J_targets_d_feats = np.matrix(np.zeros((self.n, self.p)))
        # Update derivatives of targets w.r.t feature variables
        for i, svm_model in enumerate(self.svm_models):
            if self.kernel_types[i] == 'LINEAR':
                J_targets_d_feats[i,:] = self.J_targets_d_feats[i,:]

            elif self.kernel_types[i] == 'RBF':
                svs = self.full_SVs[i]
                gamma = svm_model.param.gamma
                for l, v in enumerate(svs):
                    k_sv = self.rbf_kernel(z_k, v, gamma)
                    alpha_gamma = self.rbf_deriv_prefixes[i][l]
                    for j in xrange(self.p):
                        J_targets_d_feats[i, j] += alpha_gamma*k_sv*(z_k[j] - v[j])

            elif self.kernel_types[i] == 'POLYNOMIAL':
                k = svm_model.param.coef0
                svs = self.full_SVs[i]
                c0 = svm_model.param.coef0
                gamma = svm_model.param.gamma
                alphas = svm_model.get_sv_coef()
                for l, v in enumerate(svs):
                    alpha = alphas[l][0]
                    core = alpha*k*gamma*(gamma*np.dot(z_k, v)+c0)**(k-1)
                    for j in xrange(self.p):
                        J_targets_d_feats[i, j] += core*v[j]

            elif self.kernel_types[i] == 'SIGMOID':
                # (1-tanh(gamma*np.dot(z,v)+c0))*gamma*v[j]
                c0 = svm_model.param.coef0
                svs = self.full_SVs[i]
                gamma = svm_model.param.gamma
                alphas = svm_model.get_sv_coef()
                for l, v in enumerate(svs):
                    alpha = alphas[l][0]
                    for j in xrange(self.p):
                        J_targets_d_feats[i, j] += (1-tanh(gamma*np.dot(z_k, v)+c0))*gamma*v[j]
            elif self.kernel_types[i] == 'PRECOMPUTED':
                # TODO: User needs to supply a function to be called here
                pass

        # Do chain rule here
        # print 'J_base:\n', self.J_base
        # print 'J_delta_d_targets:\n', J_delta_d_targets
        # print 'J_targets_d_feats:\n', J_targets_d_feats
        # print 'J_feats_d_opts:\n', J_feats_d_opts
        J = self.J_base + J_delta_d_targets*J_targets_d_feats*J_feats_d_opts
        return J

    #
    # Features transforms
    #
    def select_feature_data_and_targets(self, X_all, Y_all):
        '''
        X_all - full feature vector for all examples (list of lists)
        Y_all - dictionary of all targets, keys are _TARGET_NAMES.keys()
        '''
        X = []
        for x_raw in X_all:
            x_subset = []
            for feat in self.feature_names:
                # TODO: Correctly deal with shape features heres
                feat_idx = dynamics_learning._FEAT_INDICES[feat]
                x_subset.append(x_raw[feat_idx])
            # TODO: Append XTRA Features
            X.append(x_subset)

        Y = []

        # Select target sets based on requested target names
        for name in self.target_names:
            Y.append(Y_all[name])
        return (X, Y)

    def transform_opt_vector_to_feat_vector(self, x_k, u_k, xtra=[]):
        z = []
        # Setup rotation matrix for local feats
        if self.obj_frame_ee_feats or self.obj_frame_u_feats:
            st = sin(x_k[2])
            ct = cos(x_k[2])
            R = np.matrix([[ct, st],
                           [-st, ct]])
            # Transfrom ee to object frame
            if self.obj_frame_ee_feats:
                ee_demeaned = np.matrix([[x_k[self.ee_x_opt_idx] - x_k[self.obj_x_opt_idx]],
                                         [x_k[self.ee_y_opt_idx] - x_k[self.obj_y_opt_idx]]])
                ee_obj = np.array(R*ee_demeaned).T.ravel()
                ee_phi_obj = subPIAngle(x_k[self.ee_phi_opt_idx] - x_k[self.obj_theta_opt_idx])
            if self.obj_frame_u_feats:
                # transfrom u to object frame
                u_obj = np.array(R*np.matrix(u_k[:2]).T).ravel()

        for feature_name in self.feature_names:
            if feature_name == dynamics_learning._OBJ_X_WORLD:
                z.append(x_k[self.obj_x_opt_idx])
            elif feature_name == dynamics_learning._OBJ_Y_WORLD:
                z.append(x_k[self.obj_y_opt_idx])
            elif feature_name == dynamics_learning._OBJ_THETA_WORLD:
                z.append(x_k[self.obj_theta_opt_idx])
            elif feature_name == dynamics_learning._EE_X_WORLD:
                z.append(x_k[self.ee_x_opt_idx])
            elif feature_name == dynamics_learning._EE_Y_WORLD:
                z.append(x_k[self.ee_y_opt_idx])
            elif feature_name == dynamics_learning._EE_PHI_WORLD:
                z.append(x_k[self.ee_phi_opt_idx])
            elif feature_name == dynamics_learning._U_X_WORLD:
                z.append(u_k[0])
            elif feature_name == dynamics_learning._U_Y_WORLD:
                z.append(u_k[1])
            elif feature_name == dynamics_learning._U_PHI_WORLD:
                z.append(u_k[2])
            elif feature_name == dynamics_learning._EE_X_OBJ:
                z.append(ee_obj[0])
            elif feature_name == dynamics_learning._EE_Y_OBJ:
                z.append(ee_obj[1])
            elif feature_name == dynamics_learning._EE_PHI_OBJ:
                z.append(ee_phi_obj)
            elif feature_name == dynamics_learning._U_X_OBJ:
                z.append(u_obj[0])
            elif feature_name == dynamics_learning._U_Y_OBJ:
                z.append(u_obj[1])
            # TODO: Implement below if desired
            # elif feature_name == dynamics_learning._EE_Z_WORLD:
            #     z.append(0.0)
            # elif feature_name == dynamics_learning._U_Z_WORLD:
            #     z.append(0.0)
            # elif feature_name == dynamics_learning._U_PHI_WORLD:
            #     z.append(0.0)
            # elif feature_name == dynamics_learning._EE_Z_OBJ:
            #     z.append(0.0)
            # elif feature_name == dynamics_learning._SHAPE_LOCAL:
            #     z.append(0.0)
            # elif feature_name == dynamics_learning._SHAPE_GLOBAL:
            #     z.append(0.0)
        # Add auxilarty features
        z.extend(xtra)
        return z

    def transform_svm_results_to_opt_vector(self, x_k, u_k, deltas):
        obj_x_val = 0.0
        obj_y_val = 0.0
        obj_theta_val = 0.0
        ee_x_val = 0.0
        ee_y_val = 0.0
        ee_phi_val = 0.0

        if self.obj_frame_ee_targets or self.obj_frame_obj_targets:
            st = sin(x_k[2])
            ct = cos(x_k[2])
            R = np.matrix([[ct, -st],
                           [st, ct]])

            if self.obj_frame_obj_targets:
                delta_obj_world = np.array(R*np.matrix([[ deltas[self.obj_x_target_idx] ],
                                                        [ deltas[self.obj_y_target_idx] ]])).ravel()
            if self.obj_frame_ee_targets:
                delta_ee_world = np.array(R*np.matrix([[ deltas[self.ee_x_target_idx] ],
                                                       [ deltas[self.ee_y_target_idx] ]])).ravel()

        for i, target_name in enumerate(self.target_names):
            if target_name == dynamics_learning._DELTA_OBJ_X_WORLD:
                obj_x_val = x_k[0] + deltas[i]
            elif target_name == dynamics_learning._DELTA_OBJ_Y_WORLD:
                obj_y_val = x_k[1] + deltas[i]
            elif target_name == dynamics_learning._DELTA_OBJ_THETA_WORLD:
                obj_theta_val = subPIAngle(x_k[2] + deltas[i])
            elif target_name == dynamics_learning._DELTA_EE_X_WORLD:
                ee_x_val = x_k[3] + deltas[i]
            elif target_name == dynamics_learning._DELTA_EE_Y_WORLD:
                ee_y_val = x_k[4] + deltas[i]
            elif target_name == dynamics_learning._DELTA_EE_PHI_WORLD:
                ee_phi_val = subPIAngle(x_k[5] + deltas[i])
            elif target_name == dynamics_learning._DELTA_OBJ_X_OBJ:
                obj_x_val = x_k[0] + delta_obj_world[0]
            elif target_name == dynamics_learning._DELTA_OBJ_Y_OBJ:
                obj_y_val = x_k[1] + delta_obj_world[1]
            elif target_name == dynamics_learning._DELTA_EE_X_OBJ:
                ee_x_val = x_k[3] + delta_ee_world[0]
            elif target_name == dynamics_learning._DELTA_EE_Y_OBJ:
                ee_y_val = x_k[4] + delta_ee_world[1]
            # TODO: Setup these if desired
            # elif target_name == dynamics_learning._DELTA_EE_Z_WORLD:
            #     pass
            # elif target_name == dynamics_learning._DELTA_T:
            #     pass

        if self.use_naive_ee_model:
            # X ee
            ee_x_val = x_k[3] + self.delta_t*u_k[0]
            # Y ee
            ee_y_val = x_k[4] + self.delta_t*u_k[1]
            # Phi ee
            ee_phi_val = x_k[5] + self.delta_t*u_k[2]

        return [obj_x_val, obj_y_val, obj_theta_val, ee_x_val, ee_y_val, ee_phi_val]

    #
    # Kernel Functions
    #
    def rbf_kernel(self, z, v, gamma):
        d = z - v
        return exp(-gamma*sum(d*d))

    #
    # I/O Functions
    #
    def save_models(self, output_file_base_string):
        '''
        Write svm models to disk
        output_file_base_string - base path and naming prefix for saving learning model and parameters
        '''
        param_file_name = output_file_base_string+_PARAM_FILE_SUFFIX
        print 'Writing param_file:', param_file_name
        self.write_param_file(param_file_name)

        for model, target_name in zip(self.svm_models, self.target_names):
            file_name = output_file_base_string + '_' + target_name + '.model'
            print 'Saving svm model file:', file_name
            svmutil.svm_save_model(file_name, model)

    def load_models(self, param_file_name):
        '''
        Load parameters and previously learned SVM models
        '''
        input_base_string = param_file_name[:-len(_PARAM_FILE_SUFFIX)]
        self.parse_param_file(param_file_name)
        for target_name in self.target_names:
            svm_file_name = input_base_string + '_' + target_name + '.model'
            print 'Loading file:', svm_file_name
            model = svmutil.svm_load_model(svm_file_name)
            self.svm_models.append(model)
            self.kernel_types.append(self._KERNEL_NAMES[model.param.kernel_type])

    def write_param_file(self, param_file_name):
        '''
        Write necessary parameters not stored in the SVM model file
        '''
        param_file = file(param_file_name, 'w')
        param_file.write(str(self.delta_t)+'\n')
        param_file.write(str(self.n)+'\n')
        param_file.write(str(self.m)+'\n')

        param_file.write(_PARAM_FEATURE_HEADER+'\n')
        for feature_name in self.feature_names:
            param_file.write(feature_name+'\n')
        param_file.write(_PARAM_TARGET_HEADER+'\n')
        for target_name in self.target_names:
            param_file.write(target_name+'\n')
        # NOTE: Add any other necessary parameters here
        param_file.close()

    def parse_param_file(self, param_file_name):
        '''
        Parse necessary parameters not stored in the SVM model file
        '''
        print 'Loading SVRDynamics parameter file:', param_file_name
        param_file = file(param_file_name, 'r')
        lines = param_file.readlines()

        parsing_features = False
        parsing_targets = False
        param_file.close()

        self.feature_names = []
        self.target_names = []

        self.delta_t = float(lines.pop(0).rstrip())
        self.n = int(lines.pop(0).rstrip())
        self.m = int(lines.pop(0).rstrip())

        for l in lines:
            if l.startswith(_PARAM_FEATURE_HEADER):
                parsing_features = True
                parsing_targets = False
            elif l.startswith(_PARAM_TARGET_HEADER):
                parsing_features = False
                parsing_targets = True
            elif parsing_features:
                self.feature_names.append(l.rstrip())
            elif parsing_targets:
                self.target_names.append(l.rstrip())
            # NOTE: Add any other necessary parameters here

    def make_svs_dense(self, svs):
        full_SVs = []
        for v in svs:
            v_full = np.zeros(self.p)
            for key, val in v.items():
                if key > 0:
                    v_full[key-1] = val
            full_SVs.append(v_full)
        return full_SVs


class GPPushDynamics:
    def __init__(self, delta_t = 1.0, n = 6, m = 3,
                 param_file_name = '', feature_names = [], target_names = [], xtra_names = []):
        '''
        delta_t - the (average) time between time steps
        n - number of state space dimensions
        m - number of control space dimensions
        param_file_name - base path and file name for parameter file
        '''
        # Class paramters
        self.delta_t = delta_t
        self.n = n
        self.m = m
        self.GPs = []
        self.feature_names = []
        self.target_names = []
        self.jacobian_needs_updates = False

        # Get settings from disk if a file base is specified
        if len(param_file_name) > 0:
            self.load_models(param_file_name)
        else:
            # Custom initialization otherwise
            self.feature_names = feature_names
            self.target_names = target_names
            learned_out_dims = len(target_names)


            # Setup loss function paramters
            if epsilons is not None:
                self.epsilons = epsilons
            else:
                self.epsilons = []
                for i in xrange(learned_out_dims):
                    self.epsilons.append(1e-6)

        # Set secondary parameters
        self.p = len(self.feature_names)
        self.obj_frame_ee_feats = (dynamics_learning._EE_X_OBJ in self.feature_names or
                                   dynamics_learning._EE_Y_OBJ in self.feature_names)
        self.obj_frame_u_feats  = (dynamics_learning._U_X_OBJ in self.feature_names or
                                   dynamics_learning._U_Y_OBJ in self.feature_names)
        self.use_naive_ee_model = not (dynamics_learning._DELTA_EE_X_WORLD in self.target_names or
                                       dynamics_learning._DELTA_EE_Y_WORLD in self.target_names or
                                       dynamics_learning._DELTA_EE_X_OBJ in self.target_names or
                                       dynamics_learning._DELTA_EE_Y_OBJ in self.target_names or
                                       dynamics_learning._DELTA_EE_PHI_WORLD in self.target_names)

        self.obj_frame_obj_targets = (dynamics_learning._DELTA_OBJ_X_OBJ in self.target_names or
                                      dynamics_learning._DELTA_OBJ_Y_OBJ in self.target_names)
        self.obj_frame_ee_targets = (dynamics_learning._DELTA_EE_X_OBJ in self.target_names or
                                     dynamics_learning._DELTA_EE_Y_OBJ in self.target_names)

        # Optimization vector variables [Fixed outside of scope here]
        self.obj_x_opt_idx = 0
        self.obj_y_opt_idx = 1
        self.obj_theta_opt_idx = 2
        self.ee_x_opt_idx = 3
        self.ee_y_opt_idx = 4
        self.ee_phi_opt_idx = 5
        self.u_phi_opt_idx = 8
        self.u_x_opt_idx = 6
        self.u_y_opt_idx = 7

        # Set all indices for Jacobian junk
        if self.obj_frame_obj_targets:
            self.obj_x_target_idx = self.target_names.index(dynamics_learning._DELTA_OBJ_X_OBJ)
            self.obj_y_target_idx = self.target_names.index(dynamics_learning._DELTA_OBJ_Y_OBJ)
        else:
            self.obj_x_target_idx = self.target_names.index(dynamics_learning._DELTA_OBJ_X_WORLD)
            self.obj_y_target_idx = self.target_names.index(dynamics_learning._DELTA_OBJ_Y_WORLD)

        self.obj_theta_target_idx = self.target_names.index(dynamics_learning._DELTA_OBJ_THETA_WORLD)

        if self.obj_frame_ee_targets:
            self.ee_x_target_idx = self.target_names.index(dynamics_learning._DELTA_EE_X_OBJ)
            self.ee_y_target_idx = self.target_names.index(dynamics_learning._DELTA_EE_Y_OBJ)
            self.ee_phi_target_idx = self.target_names.index(dynamics_learning._DELTA_EE_PHI_WORLD)
        elif self.use_naive_ee_model:
            self.ee_x_target_idx = 3
            self.ee_y_target_idx = 4
            self.ee_phi_target_idx = 5
        else:
            self.ee_x_target_idx = self.target_names.index(dynamics_learning._DELTA_EE_X_WORLD)
            self.ee_y_target_idx = self.target_names.index(dynamics_learning._DELTA_EE_Y_WORLD)
            self.ee_phi_target_idx = self.target_names.index(dynamics_learning._DELTA_EE_PHI_WORLD)


        if self.obj_frame_ee_feats:
            self.ee_x_feat_idx = self.feature_names.index(dynamics_learning._EE_X_OBJ)
            self.ee_y_feat_idx = self.feature_names.index(dynamics_learning._EE_Y_OBJ)
            self.ee_phi_feat_idx = self.feature_names.index(dynamics_learning._EE_PHI_OBJ)
        else:
            self.ee_x_feat_idx = self.feature_names.index(dynamics_learning._EE_X_WORLD)
            self.ee_y_feat_idx = self.feature_names.index(dynamics_learning._EE_Y_WORLD)
            self.ee_phi_feat_idx = self.feature_names.index(dynamics_learning._EE_PHI_WORLD)

        if self.obj_frame_u_feats:
            self.u_x_feat_idx = self.feature_names.index(dynamics_learning._U_X_OBJ)
            self.u_y_feat_idx = self.feature_names.index(dynamics_learning._U_Y_OBJ)
        else:
            self.u_x_feat_idx = self.feature_names.index(dynamics_learning._U_X_WORLD)
            self.u_y_feat_idx = self.feature_names.index(dynamics_learning._U_Y_WORLD)

        self.u_phi_feat_idx = self.feature_names.index(dynamics_learning._U_PHI_WORLD)

        self.build_jacobian()

    def learn_model(self, X_all, Y_all, kernel_params = {}):
        '''
        Learns SVMs to predict dynamics given the input data
        X_all - batch list of features
        Y_all - batch list of targets
        kernel_params - user specified dictionary of kernel specific parameters
        '''
        (X, Y) = self.select_feature_data_and_targets(X_all, Y_all)

        for i, Y_i in enumerate(Y):
            print 'Learning for target:', self.target_names[i]
            param_string = '-s 3 -t ' + str(self._KERNEL_TYPES[self.kernel_types[i]]) + ' -p ' + str(self.epsilons[i])
            if i in kernel_params:
                param_string += ' ' + kernel_params[i]
            # print 'Parameters', param_string
            svm_model = svmutil.svm_train(Y_i, X, param_string)
            self.svm_models.append(svm_model)
        self.build_jacobian()

    def test_batch_data(self, X_all, Y_all):
        (X, Y_out) = self.select_feature_data_and_targets(X_all, Y_all)
        Y_hat = []
        for Y_i, svm_model in zip(Y_out, self.svm_models):
            [Y_hat_i, _, _] = svmutil.svm_predict(Y_i, X, svm_model, '-q')
            Y_hat.append(Y_hat_i)
        return (Y_hat, Y_out, X)

    #
    # Prediction and jacobian methods
    #
    def predict(self, x_k, u_k, xtra=[]):
        '''
        Predict the next state given current state estimates and control input
        x_k - current state estimate (ndarray)
        u_k - current control to evaluate (ndarray)
        xtra - other features for SVM
        '''
        z = [self.transform_opt_vector_to_feat_vector(x_k, u_k, xtra)]
        y = [0]

        deltas = []

        for i, gp in enumerate(self.GPs):
            delta_i = gp.predict(z)
            deltas.append(delta_i[0])

        return np.array(self.transform_svm_results_to_opt_vector(x_k, u_k, deltas))

    def jacobian(self, x_k, u_k, xtra=[]):
        '''
        Return the matrix of partial derivatives of the dynamics model w.r.t. the current state and control
        x_k - current state estimate (ndarray)
        u_k - current control to evaluate (ndarray)
        xtra - other features for SVM
        '''
        if self.jacobian_needs_updates:
            return self.update_jacobian(x_k, u_k, xtra)
        return self.J

    def build_jacobian(self):
        '''
        Setup the basic structure of the jacobian for faster computation at each iteration
        '''
        self.J_base = np.matrix(np.eye(self.n, self.n+self.m))
        self.J_delta_d_targets = np.matrix(np.zeros((self.n, self.n)))
        self.jacobian_needs_updates = False

        # Setup partials for ee position change, currently using linear model of applied velocity
        if self.use_naive_ee_model:
            self.J_base[3:self.n, self.n: ] = np.eye(self.m)*self.delta_t

        if (self.obj_frame_ee_targets or self.obj_frame_obj_targets or
            self.obj_frame_ee_feats or self.obj_frame_u_feats):
            self.jacobian_needs_updates = True

        # Setup static deltas d targets (currently none)
        J_delta_d_targets = np.matrix(np.eye(self.n, self.n))

        # TODO: Setup jacobain of learned targets w.r.t. features

        # Setup static feature d opts (currently none)
        J_feats_d_opts = np.matrix(np.eye(self.p, self.n+self.m))

        # print 'self.J_base:\n', self.J_base
        # print 'J_delta_d_targets:\n',J_delta_d_targets
        # print 'self.J_targets_d_feats:\n',self.J_targets_d_feats
        # print 'J_feats_d_opts:\n', J_feats_d_opts

        self.J = self.J_base + J_delta_d_targets*self.J_targets_d_feats*J_feats_d_opts

    def update_jacobian(self, x_k, u_k, xtra=[]):
        '''
        Return the matrix of partial derivatives of the dynamics model w.r.t. the current state and control
        x_k - current state estimate (ndarray)
        u_k - current control to evaluate (ndarray)
        xtra - other features for SVM
        '''
        # NOTE: Being lazy and assuming a specific order of features here...
        st = sin(x_k[self.obj_theta_opt_idx])
        ct = cos(x_k[self.obj_theta_opt_idx])

        # Update derivative of opt deltas w.r.t. learned targets
        J_delta_d_targets = np.matrix(np.eye(self.n, self.n))
        if self.obj_frame_obj_targets:
            J_delta_d_targets[self.obj_x_opt_idx, self.obj_x_target_idx] = ct # d/do_x^o[o_x^w]
            J_delta_d_targets[self.obj_x_opt_idx, self.obj_y_target_idx] = -st # d/do_y^o[o_x^w]
            J_delta_d_targets[self.obj_y_opt_idx, self.obj_x_target_idx] = st # d/do_x^o[o_y^w]
            J_delta_d_targets[self.obj_y_opt_idx, self.obj_y_target_idx] = ct # d/do_y^o[o_y^w]

        if self.obj_frame_ee_targets:
            J_delta_d_targets[self.ee_x_opt_idx, self.ee_x_target_idx] = ct # d/dee_x^o[ee_x^w]
            J_delta_d_targets[self.ee_x_opt_idx, self.ee_y_target_idx] = -st # d/dee_y^o[ee_x^w]
            J_delta_d_targets[self.ee_y_opt_idx, self.ee_x_target_idx] = st # d/dee_x^o[ee_y^w]
            J_delta_d_targets[self.ee_y_opt_idx, self.ee_y_target_idx] = ct # d/dee_y^o[ee_y^w]

        # Update derivatives of features w.r.t. decision variables
        J_feats_d_opts = np.matrix(np.zeros((self.p, self.n+self.m)))
        if self.obj_frame_ee_feats:
            x_ee_demeaned = x_k[self.ee_x_opt_idx] - x_k[self.obj_x_opt_idx]
            y_ee_demeaned = x_k[self.ee_y_opt_idx] - x_k[self.obj_y_opt_idx]

            J_feats_d_opts[self.ee_x_feat_idx, self.obj_x_opt_idx] = -ct # d/dx[ee_x^o]
            J_feats_d_opts[self.ee_x_feat_idx, self.obj_y_opt_idx] = -st # d/dy[ee_x^o]
            J_feats_d_opts[self.ee_x_feat_idx,
                           self.obj_theta_opt_idx] =  st*x_ee_demeaned - ct*y_ee_demeaned # d/dtheta[ee_x^o]
            J_feats_d_opts[self.ee_x_feat_idx, self.ee_x_opt_idx] =  ct # d/dee_x[ee_x^o]
            J_feats_d_opts[self.ee_x_feat_idx, self.ee_y_opt_idx] =  st # d/dee_y[ee_x^o]

            J_feats_d_opts[self.ee_y_feat_idx, self.obj_x_opt_idx] =  st # d/dx[ee_y^o]
            J_feats_d_opts[self.ee_y_feat_idx, self.obj_y_opt_idx] = -ct # d/dy[ee_y^o]
            J_feats_d_opts[self.ee_y_feat_idx,
                           self.obj_theta_opt_idx] =  ct*x_ee_demeaned + st*y_ee_demeaned # d/dtheta[ee_y^o]
            J_feats_d_opts[self.ee_y_feat_idx, self.ee_x_opt_idx] = -st # d/dx_ee[ee_y^o]
            J_feats_d_opts[self.ee_y_feat_idx, self.ee_y_opt_idx] =  ct # d/dy_ee[ee_y^o]

            # Setup ee_phi partials
            J_feats_d_opts[self.ee_phi_feat_idx, self.ee_phi_opt_idx] = 1.0 # d/dee_phi^w[ee_phi^o]
            J_feats_d_opts[self.ee_phi_feat_idx, self.obj_theta_opt_idx] =  -1.0 # d/dtheta[ee_phi^o]
        else:
            # Identity matrix
            J_feats_d_opts[self.ee_x_feat_idx, self.ee_x_opt_idx] = 1.
            J_feats_d_opts[self.ee_y_feat_idx, self.ee_y_opt_idx] = 1.
            J_feats_d_opts[self.ee_phi_feat_idx, self.ee_phi_opt_idx] = 1.

        if self.obj_frame_u_feats:
            J_feats_d_opts[self.u_x_feat_idx, self.obj_theta_opt_idx] = st*u_k[0] - ct*u_k[1] # d/dtheta[u_x]
            J_feats_d_opts[self.u_x_feat_idx, self.u_x_opt_idx] = ct # d/du_x[u_x]
            J_feats_d_opts[self.u_x_feat_idx, self.u_y_opt_idx] = st # d/du_y[u_x]

            J_feats_d_opts[self.u_y_feat_idx, self.obj_theta_opt_idx] =  ct*u_k[0] + st*u_k[1] # d/dtheta[u_y]
            J_feats_d_opts[self.u_y_feat_idx, self.u_x_opt_idx] = -st # d/du_x[u_y]
            J_feats_d_opts[self.u_y_feat_idx, self.u_y_opt_idx] =  ct # d/du_y[u_y]
        else:
            # Identity matrix
            J_feats_d_opts[self.u_x_feat_idx, self.u_x_opt_idx] = 1.
            J_feats_d_opts[self.u_y_feat_idx, self.u_y_opt_idx] = 1.
        # Setup derivates of u_phi
        J_feats_d_opts[self.u_phi_feat_idx, self.u_phi_opt_idx] = 1.

        # Needed for kernel derivative evaluation
        z_k = self.transform_opt_vector_to_feat_vector(x_k, u_k, xtra)

        J_targets_d_feats = np.matrix(np.zeros((self.n, self.p)))
        # TODO: Update derivatives of targets w.r.t feature variables

        # Do chain rule here
        # print 'J_base:\n', self.J_base
        # print 'J_delta_d_targets:\n', J_delta_d_targets
        # print 'J_targets_d_feats:\n', J_targets_d_feats
        # print 'J_feats_d_opts:\n', J_feats_d_opts
        J = self.J_base + J_delta_d_targets*J_targets_d_feats*J_feats_d_opts
        return J

    #
    # Features transforms
    #
    def select_feature_data_and_targets(self, X_all, Y_all):
        '''
        X_all - full feature vector for all examples (list of lists)
        Y_all - dictionary of all targets, keys are _TARGET_NAMES.keys()
        '''
        X = []
        for x_raw in X_all:
            x_subset = []
            for feat in self.feature_names:
                # TODO: Correctly deal with shape features heres
                feat_idx = dynamics_learning._FEAT_INDICES[feat]
                x_subset.append(x_raw[feat_idx])
            # TODO: Append XTRA Features
            X.append(x_subset)

        Y = []

        # Select target sets based on requested target names
        for name in self.target_names:
            Y.append(Y_all[name])
        return (X, Y)

    def transform_opt_vector_to_feat_vector(self, x_k, u_k, xtra=[]):
        z = []
        # Setup rotation matrix for local feats
        if self.obj_frame_ee_feats or self.obj_frame_u_feats:
            st = sin(x_k[2])
            ct = cos(x_k[2])
            R = np.matrix([[ct, st],
                           [-st, ct]])
            # Transfrom ee to object frame
            if self.obj_frame_ee_feats:
                ee_demeaned = np.matrix([[x_k[self.ee_x_opt_idx] - x_k[self.obj_x_opt_idx]],
                                         [x_k[self.ee_y_opt_idx] - x_k[self.obj_y_opt_idx]]])
                ee_obj = np.array(R*ee_demeaned).T.ravel()
                ee_phi_obj = subPIAngle(x_k[self.ee_phi_opt_idx] - x_k[self.obj_theta_opt_idx])
            if self.obj_frame_u_feats:
                # transfrom u to object frame
                u_obj = np.array(R*np.matrix(u_k[:2]).T).ravel()

        for feature_name in self.feature_names:
            if feature_name == dynamics_learning._OBJ_X_WORLD:
                z.append(x_k[self.obj_x_opt_idx])
            elif feature_name == dynamics_learning._OBJ_Y_WORLD:
                z.append(x_k[self.obj_y_opt_idx])
            elif feature_name == dynamics_learning._OBJ_THETA_WORLD:
                z.append(x_k[self.obj_theta_opt_idx])
            elif feature_name == dynamics_learning._EE_X_WORLD:
                z.append(x_k[self.ee_x_opt_idx])
            elif feature_name == dynamics_learning._EE_Y_WORLD:
                z.append(x_k[self.ee_y_opt_idx])
            elif feature_name == dynamics_learning._EE_PHI_WORLD:
                z.append(x_k[self.ee_phi_opt_idx])
            elif feature_name == dynamics_learning._U_X_WORLD:
                z.append(u_k[0])
            elif feature_name == dynamics_learning._U_Y_WORLD:
                z.append(u_k[1])
            elif feature_name == dynamics_learning._U_PHI_WORLD:
                z.append(u_k[2])
            elif feature_name == dynamics_learning._EE_X_OBJ:
                z.append(ee_obj[0])
            elif feature_name == dynamics_learning._EE_Y_OBJ:
                z.append(ee_obj[1])
            elif feature_name == dynamics_learning._EE_PHI_OBJ:
                z.append(ee_phi_obj)
            elif feature_name == dynamics_learning._U_X_OBJ:
                z.append(u_obj[0])
            elif feature_name == dynamics_learning._U_Y_OBJ:
                z.append(u_obj[1])
            # TODO: Implement below if desired
            # elif feature_name == dynamics_learning._EE_Z_WORLD:
            #     z.append(0.0)
            # elif feature_name == dynamics_learning._U_Z_WORLD:
            #     z.append(0.0)
            # elif feature_name == dynamics_learning._U_PHI_WORLD:
            #     z.append(0.0)
            # elif feature_name == dynamics_learning._EE_Z_OBJ:
            #     z.append(0.0)
            # elif feature_name == dynamics_learning._SHAPE_LOCAL:
            #     z.append(0.0)
            # elif feature_name == dynamics_learning._SHAPE_GLOBAL:
            #     z.append(0.0)
        # Add auxilarty features
        z.extend(xtra)
        return z

    def transform_svm_results_to_opt_vector(self, x_k, u_k, deltas):
        obj_x_val = 0.0
        obj_y_val = 0.0
        obj_theta_val = 0.0
        ee_x_val = 0.0
        ee_y_val = 0.0
        ee_phi_val = 0.0

        if self.obj_frame_ee_targets or self.obj_frame_obj_targets:
            st = sin(x_k[2])
            ct = cos(x_k[2])
            R = np.matrix([[ct, -st],
                           [st, ct]])

            if self.obj_frame_obj_targets:
                delta_obj_world = np.array(R*np.matrix([[ deltas[self.obj_x_target_idx] ],
                                                        [ deltas[self.obj_y_target_idx] ]])).ravel()
            if self.obj_frame_ee_targets:
                delta_ee_world = np.array(R*np.matrix([[ deltas[self.ee_x_target_idx] ],
                                                       [ deltas[self.ee_y_target_idx] ]])).ravel()

        for i, target_name in enumerate(self.target_names):
            if target_name == dynamics_learning._DELTA_OBJ_X_WORLD:
                obj_x_val = x_k[0] + deltas[i]
            elif target_name == dynamics_learning._DELTA_OBJ_Y_WORLD:
                obj_y_val = x_k[1] + deltas[i]
            elif target_name == dynamics_learning._DELTA_OBJ_THETA_WORLD:
                obj_theta_val = subPIAngle(x_k[2] + deltas[i])
            elif target_name == dynamics_learning._DELTA_EE_X_WORLD:
                ee_x_val = x_k[3] + deltas[i]
            elif target_name == dynamics_learning._DELTA_EE_Y_WORLD:
                ee_y_val = x_k[4] + deltas[i]
            elif target_name == dynamics_learning._DELTA_EE_PHI_WORLD:
                ee_phi_val = subPIAngle(x_k[5] + deltas[i])
            elif target_name == dynamics_learning._DELTA_OBJ_X_OBJ:
                obj_x_val = x_k[0] + delta_obj_world[0]
            elif target_name == dynamics_learning._DELTA_OBJ_Y_OBJ:
                obj_y_val = x_k[1] + delta_obj_world[1]
            elif target_name == dynamics_learning._DELTA_EE_X_OBJ:
                ee_x_val = x_k[3] + delta_ee_world[0]
            elif target_name == dynamics_learning._DELTA_EE_Y_OBJ:
                ee_y_val = x_k[4] + delta_ee_world[1]
            # TODO: Setup these if desired
            # elif target_name == dynamics_learning._DELTA_EE_Z_WORLD:
            #     pass
            # elif target_name == dynamics_learning._DELTA_T:
            #     pass

        if self.use_naive_ee_model:
            # X ee
            ee_x_val = x_k[3] + self.delta_t*u_k[0]
            # Y ee
            ee_y_val = x_k[4] + self.delta_t*u_k[1]
            # Phi ee
            ee_phi_val = x_k[5] + self.delta_t*u_k[2]

        return [obj_x_val, obj_y_val, obj_theta_val, ee_x_val, ee_y_val, ee_phi_val]

    #
    # Kernel Functions
    #
    def rbf_kernel(self, z, v, gamma):
        d = z - v
        return exp(-gamma*sum(d*d))

    #
    # I/O Functions
    #
    def save_models(self, output_file_base_string):
        '''
        Write svm models to disk
        output_file_base_string - base path and naming prefix for saving learning model and parameters
        '''
        param_file_name = output_file_base_string+_PARAM_FILE_SUFFIX
        print 'Writing param_file:', param_file_name
        self.write_param_file(param_file_name)

        for model, target_name in zip(self.svm_models, self.target_names):
            file_name = output_file_base_string + '_' + target_name + '.model'
            print 'Saving svm model file:', file_name
            svmutil.svm_save_model(file_name, model)

    def load_models(self, param_file_name):
        '''
        Load parameters and previously learned SVM models
        '''
        input_base_string = param_file_name[:-len(_PARAM_FILE_SUFFIX)]
        self.parse_param_file(param_file_name)
        for target_name in self.target_names:
            svm_file_name = input_base_string + '_' + target_name + '.model'
            print 'Loading file:', svm_file_name
            model = svmutil.svm_load_model(svm_file_name)
            self.svm_models.append(model)
            self.kernel_types.append(self._KERNEL_NAMES[model.param.kernel_type])

    def write_param_file(self, param_file_name):
        '''
        Write necessary parameters not stored in the SVM model file
        '''
        param_file = file(param_file_name, 'w')
        param_file.write(str(self.delta_t)+'\n')
        param_file.write(str(self.n)+'\n')
        param_file.write(str(self.m)+'\n')

        param_file.write(_PARAM_FEATURE_HEADER+'\n')
        for feature_name in self.feature_names:
            param_file.write(feature_name+'\n')
        param_file.write(_PARAM_TARGET_HEADER+'\n')
        for target_name in self.target_names:
            param_file.write(target_name+'\n')
        # NOTE: Add any other necessary parameters here
        param_file.close()

    def parse_param_file(self, param_file_name):
        '''
        Parse necessary parameters not stored in the SVM model file
        '''
        print 'Loading SVRDynamics parameter file:', param_file_name
        param_file = file(param_file_name, 'r')
        lines = param_file.readlines()

        parsing_features = False
        parsing_targets = False
        param_file.close()

        self.feature_names = []
        self.target_names = []

        self.delta_t = float(lines.pop(0).rstrip())
        self.n = int(lines.pop(0).rstrip())
        self.m = int(lines.pop(0).rstrip())

        for l in lines:
            if l.startswith(_PARAM_FEATURE_HEADER):
                parsing_features = True
                parsing_targets = False
            elif l.startswith(_PARAM_TARGET_HEADER):
                parsing_features = False
                parsing_targets = True
            elif parsing_features:
                self.feature_names.append(l.rstrip())
            elif parsing_targets:
                self.target_names.append(l.rstrip())
            # NOTE: Add any other necessary parameters here
