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

import roslib; roslib.load_manifest('tabletop_pushing')
import push_learning
from push_learning import subPIAngle
import numpy as np
import os
import svmutil
from math import sin, cos
from tf.transformations import euler_from_quaternion


# Setup dictionaries to index into labels and features
_DELTA_X_OBJ_WORLD = 'DELTA_X_OBJ_WORLD'
_DELTA_Y_OBJ_WORLD = 'DELTA_Y_OBJ_WORLD'
_DELTA_THETA_OBJ_WORLD = 'DELTA_THETA_OBJ_WORLD'
_DELTA_X_EE_WORLD = 'DELTA_X_EE_WORLD'
_DELTA_Y_EE_WORLD = 'DELTA_Y_EE_WORLD'
_DELTA_Z_EE_WORLD = 'DELTA_Z_EE_WORLD'
_DELTA_PHI_EE_WORLD = 'DELTA_PHI_EE_WORLD'
_DELTA_T = 'DELTA_T'
_DELTA_X_OBJ_OBJ = 'DELTA_X_OBJ_OBJ'
_DELTA_Y_OBJ_OBJ = 'DELTA_Y_OBJ_OBJ'
_DELTA_X_EE_OBJ = 'DELTA_X_EE_OBJ'
_DELTA_Y_EE_OBJ = 'DELTA_Y_EE_OBJ'

_X_OBJ_WORLD = 'X_OBJ_WORLD'
_Y_OBJ_WORLD = 'Y_OBJ_WORLD'
_THETA_OBJ_WORLD = 'THETA_OBJ_WORLD'
_X_EE_WORLD = 'X_EE_WORLD'
_Y_EE_WORLD = 'Y_EE_WORLD'
_Z_EE_WORLD = 'Z_EE_WORLD'
_PHI_EE_WORLD = 'PHI_EE_WORLD'
_U_X_WORLD = 'U_X_WORLD'
_U_Y_WORLD = 'U_Y_WORLD'
_U_Z_WORLD = 'U_Z_WORLD'
_U_PHI_WORLD = 'U_PHI_WORLD'
_X_EE_OBJ = 'X_EE_OBJ'
_Y_EE_OBJ = 'Y_EE_OBJ'
_Z_EE_OBJ = 'Z_EE_OBJ'
_PHI_EE_OBJ = 'PHI_EE_OBJ'
_U_X_OBJ = 'U_X_OBJ'
_U_Y_OBJ = 'U_Y_OBJ'
_SHAPE_LOCAL = 'SHAPE_LOCAL'
_SHAPE_GLOBAL = 'SHAPE_GLOBAL'

_TARGET_INDICES = {_DELTA_X_OBJ_WORLD:0,
                   _DELTA_Y_OBJ_WORLD:1,
                   _DELTA_THETA_OBJ_WORLD:2,
                   _DELTA_X_EE_WORLD:3,
                   _DELTA_Y_EE_WORLD:4,
                   _DELTA_Z_EE_WORLD:5,
                   _DELTA_PHI_EE_WORLD:6,
                   _DELTA_T:7,
                   _DELTA_X_OBJ_OBJ:8,
                   _DELTA_Y_OBJ_OBJ:9,
                   _DELTA_X_EE_OBJ:10,
                   _DELTA_Y_EE_OBJ:11}

_FEAT_INDICES = {_X_OBJ_WORLD:0,
                 _Y_OBJ_WORLD:1,
                 _THETA_OBJ_WORLD:2,
                 _X_EE_WORLD:3,
                 _Y_EE_WORLD:4,
                 _Z_EE_WORLD:5,
                 _PHI_EE_WORLD:6,
                 _U_X_WORLD:7,
                 _U_Y_WORLD:8,
                 _U_Z_WORLD:9,
                 _U_PHI_WORLD:10,
                 _X_EE_OBJ:11,
                 _Y_EE_OBJ:12,
                 _Z_EE_OBJ:13,
                 _PHI_EE_OBJ:14,
                 _U_X_OBJ:15,
                 _U_Y_OBJ:16,
                 # TODO: Fix the shape stuff
                 _SHAPE_LOCAL:17,
                 _SHAPE_GLOBAL:-1}

_FEAT_NAMES = dict((v,k) for k,v in _FEAT_INDICES.items())
_TARGET_NAMES = dict((v,k) for k,v in _TARGET_INDICES.items())

# TODO: Need to test with synthetic data to confirm
def get_object_frame_features(cts, ee_phi):
    # Demean EE coordinates into object frame
    X_ee_demeaned = np.matrix([[cts.ee.position.x - cts.x.x],
                               [cts.ee.position.y - cts.x.y]])
    z_ee_obj = cts.ee.position.z - cts.z
    # Rotate x,y into object orientated frame
    st = sin(cts.x.theta)
    ct = cos(cts.x.theta)
    R = np.matrix([[ct, st],
                   [-st, ct]])
    X_ee_obj = np.array(R*X_ee_demeaned).T.ravel()
    # Rotate push vector into object frame
    U_obj = np.array(R*np.matrix([[cts.u.linear.x],
                                  [cts.u.linear.y]])).ravel()
    # Rotate EE orientation into object frame
    phi_ee_obj = subPIAngle(ee_phi - cts.x.theta)

    return [X_ee_obj[0], X_ee_obj[1], z_ee_obj,
            phi_ee_obj,
            U_obj[0], U_obj[1]]

# TODO: Need to test with synthetic data to confirm
def get_object_frame_targets(cts_t0, cts_t1):
    # Delta x and y of the object are just the object locations at the next time step
    # in the current object's frame
    X_t1_demeaned = np.matrix([[cts_t1.x.x - cts_t0.x.x],
                               [cts_t1.x.y - cts_t0.x.y]])
    st = sin(cts_t0.x.theta)
    ct = cos(cts_t1.x.theta)
    R = np.matrix([[ct, st],
                   [-st, ct]])
    X_t1_obj = np.array(R*X_t1_demeaned).T.ravel()

    delta_x_obj_obj = X_t1_obj[0]
    delta_y_obj_obj = X_t1_obj[1]

    # Convert both ee positions into t0 object frame and get diff
    ee_t0_demeaned = np.matrix([[cts_t0.ee.position.x - cts_t0.x.x],
                                [cts_t0.ee.position.y - cts_t0.x.y]])
    ee_t1_demeaned = np.matrix([[cts_t1.ee.position.x - cts_t0.x.x],
                                [cts_t1.ee.position.y - cts_t0.x.y]])

    ee_t0_obj = np.array(R*ee_t0_demeaned).T.ravel()
    ee_t1_obj = np.array(R*ee_t1_demeaned).T.ravel()
    ee_delta_X = ee_t1_obj - ee_t0_obj

    delta_x_ee_obj = ee_delta_X[0]
    delta_y_ee_obj = ee_delta_X[1]

    return [delta_x_obj_obj,
            delta_y_obj_obj,
            delta_x_ee_obj,
            delta_y_ee_obj]

# TODO: Need to test with synthetic data to confirm
def convert_push_trial_to_feat_vectors(trial):
    Z = []
    Y = []
    for i in xrange(len(trial.trial_trajectory)-1):
        cts_t0 = trial.trial_trajectory[i]
        cts_t1 = trial.trial_trajectory[i+1]
        [_, _, ee_phi_t0] = euler_from_quaternion([cts_t0.ee.orientation.x,
                                                   cts_t0.ee.orientation.y,
                                                   cts_t0.ee.orientation.z,
                                                   cts_t0.ee.orientation.w])
        [_, _, ee_phi_t1] = euler_from_quaternion([cts_t1.ee.orientation.x,
                                                   cts_t1.ee.orientation.y,
                                                   cts_t1.ee.orientation.z,
                                                   cts_t1.ee.orientation.w])
        # TODO: This will depend on the primitive used
        u_phi_world = cts_t0.u.angular.x

        z_t = [cts_t0.x.x,
               cts_t0.x.y,
               cts_t0.x.theta,
               cts_t0.ee.position.x,
               cts_t0.ee.position.y,
               cts_t0.ee.position.z,
               ee_phi_t0,
               cts_t0.u.linear.x,
               cts_t0.u.linear.y,
               cts_t0.u.linear.z,
               u_phi_world]

        # Convert world frame feats into object frame
        z_t_obj_frame = get_object_frame_features(cts_t0, ee_phi_t0)
        # TODO: Convert shape into frame for each instance?
        # TODO: Need to extract this first...
        z_shape = trial.trial_start.shape_descriptor
        z_t.extend(z_t_obj_frame)
        z_t.extend(z_shape)

        # World frame
        y_t = [cts_t1.x.x - cts_t0.x.x,
               cts_t1.x.y - cts_t0.x.y,
               cts_t1.x.theta - cts_t0.x.theta,
               cts_t1.ee.position.x - cts_t0.ee.position.x,
               cts_t1.ee.position.y - cts_t0.ee.position.y,
               cts_t1.ee.position.z - cts_t0.ee.position.z,
               ee_phi_t1 - ee_phi_t0,
               cts_t1.t - cts_t0.t]

        # Transform targets into object frame here...
        y_t_obj_frame = get_object_frame_targets(cts_t0, cts_t1)
        y_t.extend(y_t_obj_frame)

        Z.append(z_t)
        Y.append(y_t)
    return (Z, Y)

def write_dynamics_learning_trial_file(X, Y, out_file_base_name):
    for i in xrange(len(Y[0])):
        Y_i = []
        for y in Y:
            Y_i.append(y[i])
        target_name = _TARGET_NAMES[i]
        out_file_name = out_file_base_name + '_' + target_name + '.txt'
        print 'Writing:', out_file_name
        push_learning.write_example_file(out_file_name, X, Y_i)

def create_object_class_svm_files(directory_list, base_out_dir):
    '''
    directory_list - list of directories to get aff files from
    '''
    plio = push_learning.CombinedPushLearnControlIO()
    print 'Current number of trials:', len(plio.push_trials)
    for directory in directory_list:
        # Get list of directory contents, load all aff learning files
        all_files = os.listdir(directory)
        for f in all_files:
            if not (f.startswith('aff_learn_out') and not '-' in f):
                continue
            aff_file_name = directory + '/' + f
            plio.read_in_data_file(aff_file_name, True)
            print 'Current number of trials:', len(plio.push_trials)

    object_classes = {}
    for i, trial in enumerate(plio.push_trials):
        obj_id = trial.trial_start.object_id
        if obj_id in object_classes:
            object_classes[obj_id].append(i)
        else:
            object_classes[obj_id] = [i]

    print 'Object classes:', object_classes.keys()

    for obj_id in object_classes:
        out_dir = base_out_dir + obj_id+'/'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        print 'object_classes[',obj_id,'] =', object_classes[obj_id]
        for i, t_idx in enumerate(object_classes[obj_id]):
            out_file_base_name = out_dir + str(i)
            print 'out_file_base_name:', out_file_base_name
            (X, Y) = convert_push_trial_to_feat_vectors(plio.push_trials[t_idx])
            write_dynamics_learning_trial_file(X, Y, out_file_base_name)

def create_train_and_validate_splits(in_dir, out_dir):
    # TODO: Create function to take in header to link different files
    # together in train and validate sets
    pass
