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

class SVMModel:
    def __init__(self, svm_file=None):
        if svm_file is not None:
            self.svm_model = svm_load_model(svm_file)
        else:
            self.svm_model = None

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

    def learn_model(self, learn_data):
        X = []
        Y = []
        self.svm_model = svmutil.svm_train()

    def save_model(self, output_path):
        if self.svm_model is not None:
            svmutil.svm_save_model(output_path, self.svm_model)
