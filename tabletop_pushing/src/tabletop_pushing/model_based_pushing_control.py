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
from math import copysign
import svmutil

class ModelPredictiveController:
    def __init__(self, model):
        self.f = model

class NaiveInputModel:
    def __init__(self, detla_t):
        self.delta_t = delta_t

    def predict(self, cur_state, ee_pose, u):
        next_state = VisFeedbackPushTrackingFeedback()
        next_state.x.x = cur_state.x.x + u.linear.x*self.delta_t
        next_state.x.y = cur_state.x.y + u.linear.y*self.delta_t
        next_state.x.theta = cur_state.x.theta
        next_state.x_dot.x = u.linear.x
        next_state.x_dot.theta = u.linear.y
        next_state.x_dot.theta = cur_state.x_dot.theta
        return next_state

class SVMPushModel:
    def __init__(self, svm_files=None):
        if svm_files is not None:
            for svm_file in svm_files:
                self.svm_models.append(svm_load_model(svm_file))
        else:
            self.svm_models = []
        self.epsilon = 1e-6

    def predict(self, cur_state, ee_pose, u):
        # TODO: get feats
        y = 0.0
        x = []
        # TODO: perform prediction
        svmutil.svm_predict(y, x, self.svm_model)
        # TODO: transform prediction to correct state / class
        next_state = VisFeedbackPushTrackingFeedback()
        return next_state

    def transform_state_data_to_feat_vector(self, cur_state, ee_pose, u, next_state):
        pass

    def transform_trial_data_to_feat_vector(self, trajectory):
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
        X = []
        Y = []
        # print 'len(learn_data)',len(learn_data)
        for trial in learn_data:
            (x, y) = self.transform_trial_data_to_feat_vector(trial.trial_trajectory)
            X.extend(x)
            Y.extend(y)
        # print 'len(Y)', len(Y)
        # print 'len(X)', len(X)
        for i in xrange(len(Y[0])):
            Y_i = []
            for y in Y:
                Y_i.append(y[i])
            svm_model = svmutil.svm_train(Y_i, X, '-s 3 -p '+str(self.epsilon))
            self.svm_models.append(svm_model)

    def save_model(self, output_paths):
        if len(self.svm_models) > 0:
            for path, model in zip(output_paths, self.svm_models):
                svmutil.svm_save_model(path, model)

# import sys
# import push_learning
# if __name__ == '__main__':
#     aff_file_name  = sys.argv[1]
#     plio = push_learning.CombinedPushLearnControlIO()
#     plio.read_in_data_file(aff_file_name)
#     svm_dynamics = SVMPushModel()
#     svm_dynamics.learn_model(plio.push_trials)
