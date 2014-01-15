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
import dynamics_learning

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

    def __init__(self, delta_t = 1.0, n = 5, m = 2,
                 param_file_name = '', epsilons = None, kernel_type = None,
                 feature_names = [], target_names = []):
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
            # TODO: Make this variable between different dimension
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
        self.obj_frame_ee_feats = (dynamics_learning._EE_X_OBJ in self.feature_names or
                                   dynamics_learning._EE_Y_OBJ in self.feature_names)
        self.obj_frame_u_feats  = (dynamics_learning._U_X_OBJ in self.feature_names or
                                   dynamics_learning._U_Y_OBJ in self.feature_names)
        self.use_naive_ee_model = not (dynamics_learning._DELTA_EE_X_WORLD in self.target_names or
                                       dynamics_learning._DELTA_EE_Y_WORLD in self.target_names or
                                       dynamics_learning._DELTA_EE_X_OBJ in self.target_names or
                                       dynamics_learning._DELTA_EE_Y_OBJ in self.target_names)

        self.obj_frame_obj_targets = (dynamics_learning._DELTA_OBJ_X_OBJ in self.target_names or
                                      dynamics_learning._DELTA_OBJ_Y_OBJ in self.target_names)
        self.obj_frame_ee_targets = (dynamics_learning._DELTA_EE_X_OBJ in self.target_names or
                                     dynamics_learning._DELTA_EE_Y_OBJ in self.target_names)

        if self.obj_frame_obj_targets:
            self.obj_frame_obj_x_idx = self.target_names.index(dynamics_learning._DELTA_OBJ_X_OBJ);
            self.obj_frame_obj_y_idx = self.target_names.index(dynamics_learning._DELTA_OBJ_Y_OBJ);
        if self.obj_frame_ee_targets:
            self.obj_frame_ee_x_idx = self.target_names.index(dynamics_learning._DELTA_EE_X_OBJ);
            self.obj_frame_ee_y_idx = self.target_names.index(dynamics_learning._DELTA_EE_Y_OBJ);

        self.p = len(self.feature_names)
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
            # TODO: test this out
            if i in kernel_params:
                param_string += kernel_params[i]
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
        # TODO: Set this up correctly...
        self.J = np.eye(self.n, self.n+self.m)
        self.jacobian_needs_updates = False

    def build_jacobian_linear_ee(self):
        self.J = np.eye(self.n, self.n+self.m)

        # Setup partials for ee position change, currently using linear model of applied velocity
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

    def update_jacobian(self, x_k, u_k, xtra=[]):
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

    def build_jacobian_linear_ee_object_frame(self):
        self.J_base = np.eye(self.n, self.n+self.m)

        # Setup partials for ee position change, currently using linear model of applied velocity
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
            X.append(x_subset)

        Y = []

        # Select target sets based on requested target names
        for name in self.target_names:
            Y.append(Y_all[name])
        return (X, Y)

    def transform_opt_vector_to_feat_vector(self, x_k, u_k, xtra):
        z = []
        # Setup rotation matrix for local feats
        if self.obj_frame_ee_feats or self.obj_frame_u_feats:
            st = sin(x_k[2])
            ct = cos(x_k[2])
            R = np.matrix([[ct, st],
                           [-st, ct]])
            # Transfrom ee to object frame
            if self.obj_frame_ee_feats:
                ee_demeaned = np.matrix([[x_k[3]-x_k[0]],
                                         [x_k[4]-x_k[1]]])
                ee_obj = np.array(R*ee_demeaned).T.ravel()
            if self.obj_frame_u_feats:
                # transfrom u to object frame
                u_obj = np.array(R*np.matrix(u_k).T).ravel()

        for feature_name in self.feature_names:
            if feature_name == dynamics_learning._OBJ_X_WORLD:
                feat = x_k[0]
            elif feature_name == dynamics_learning._OBJ_Y_WORLD:
                feat = x_k[1]
            elif feature_name == dynamics_learning._OBJ_THETA_WORLD:
                feat = x_k[2]
            elif feature_name == dynamics_learning._EE_X_WORLD:
                feat = x_k[3]
            elif feature_name == dynamics_learning._EE_Y_WORLD:
                feat = x_k[4]
            elif feature_name == dynamics_learning._U_X_WORLD:
                feat = u_k[0]
            elif feature_name == dynamics_learning._U_Y_WORLD:
                feat = u_k[1]
            elif feature_name == dynamics_learning._EE_X_OBJ:
                feat = ee_obj[0]
            elif feature_name == dynamics_learning._EE_Y_OBJ:
                feat = ee_obj[1]
            elif feature_name == dynamics_learning._U_X_OBJ:
                feat = u_obj[0]
            elif feature_name == dynamics_learning._U_Y_OBJ:
                feat = u_obj[1]
            # TODO: Implement below if desired
            # elif feature_name == dynamics_learning._EE_Z_WORLD:
            #     feat = 0.0
            # elif feature_name == dynamics_learning._EE_PHI_WORLD:
            #     feat = 0.0
            # elif feature_name == dynamics_learning._U_Z_WORLD:
            #     feat = 0.0
            # elif feature_name == dynamics_learning._U_PHI_WORLD:
            #     feat = 0.0
            # elif feature_name == dynamics_learning._EE_Z_OBJ:
            #     feat = 0.0
            # elif feature_name == dynamics_learning._EE_PHI_OBJ:
            #     feat = 0.0
            # elif feature_name == dynamics_learning._SHAPE_LOCAL:
            #     feat = 0.0
            # elif feature_name == dynamics_learning._SHAPE_GLOBAL:
            #     feat = 0.0
            z.append(feat)
        return z

    def transform_svm_results_to_opt_vector(self, x_k, u_k, deltas):
        obj_x_val = 0.0
        obj_y_val = 0.0
        obj_theta_val = 0.0
        ee_x_val = 0.0
        ee_y_val = 0.0

        if self.obj_frame_ee_targets or self.obj_frame_obj_targets:
            st = sin(x_k[2])
            ct = cos(x_k[2])
            R = np.matrix([[ct, -st],
                           [st, ct]])

            if self.obj_frame_obj_targets:
                delta_obj_world = np.array(R*np.matrix([[ deltas[self.obj_frame_obj_x_idx] ],
                                                        [ deltas[self.obj_frame_obj_y_idx] ]])).ravel()
            if self.obj_frame_ee_targets:
                delta_ee_world = np.array(R*np.matrix([[ deltas[self.obj_frame_ee_x_idx] ],
                                                       [ deltas[self.obj_frame_ee_y_idx] ]])).ravel()

        for i, target_name in enumerate(self.target_names):
            if target_name == dynamics_learning._DELTA_OBJ_X_WORLD:
                obj_x_val = x_k[0] + deltas[i]
            elif target_name == dynamics_learning._DELTA_OBJ_Y_WORLD:
                obj_y_val = x_k[1] + deltas[i]
            elif target_name == dynamics_learning._DELTA_OBJ_THETA_WORLD:
                obj_theta_val = x_k[2] + deltas[i]
            elif target_name == dynamics_learning._DELTA_EE_X_WORLD:
                ee_x_val = x_k[3] + deltas[i]
            elif target_name == dynamics_learning._DELTA_EE_Y_WORLD:
                ee_y_val = x_k[4] + deltas[i]
            elif target_name == dynamics_learning._DELTA_OBJ_X_OBJ:
                obj_x_val = x_k[0] + delta_obj_world[0]
            elif target_name == dynamics_learning._DELTA_OBJ_Y_OBJ:
                obj_y_val = x_k[1] + delta_obj_world[1]
            elif target_name == dynamics_learning._DELTA_EE_X_OBJ:
                ee_x_val = x_k[3] + delta_ee_world[0]
            elif target_name == dynamics_learning._DELTA_EE_Y_OBJ:
                ee_x_val = x_k[4] + delta_ee_world[1]
            # TODO: Setup these if desired
            # elif target_name == dynamics_learning._DELTA_EE_Z_WORLD:
            #     pass
            # elif target_name == dynamics_learning._DELTA_EE_PHI_WORLD:
            #     pass
            # elif target_name == dynamics_learning._DELTA_T:
            #     pass

        if self.use_naive_ee_model:
            # X ee
            ee_x_val = x_k[3] + self.delta_t*u_k[0]
            # Y ee
            ee_y_val = x_k[4] + self.delta_t*u_k[1]

        return [obj_x_val, obj_y_val, obj_theta_val, ee_x_val, ee_y_val]

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
