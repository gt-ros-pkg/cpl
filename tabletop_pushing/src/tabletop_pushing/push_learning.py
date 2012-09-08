#!/usr/bin/env python
# Software License Agreement (BSD License)
#
#  Copyright (c) 2012, Georgia Institute of Technology
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
from geometry_msgs.msg import Point, Pose2D
from math import sin, cos, pi, sqrt, fabs, atan2, hypot
import cv2
import tf.transformations as tr
import numpy as np
import sys
import rospy
from push_primitives import *

_VERSION_LINE = '# v0.3'
_HEADER_LINE = '# object_id init_x init_y init_z init_theta final_x final_y final_z final_theta goal_x goal_y goal_theta behavior_primitive controller proxy which_arm precondition_method push_time'

def get_attr(instance, attribute):
    '''
    A wrapper for the builtin python getattr which handles recursive attributes of attributes
    '''
    attr_list = attribute.split('.')
    def get_nested_attr(instance, attr_list):
        if len(attr_list) == 1:
            return getattr(instance, attr_list[0])
        return get_nested_attr(getattr(instance, attr_list[0]), attr_list[1:])
    return get_nested_attr(instance, attr_list)

class PushTrial:
    def __init__(self):
        self.object_id = ''
        self.init_centroid = Point()
        self.init_orientation = 0.0
        self.final_centroid = Point()
        self.final_orientation = 0.0
        self.goal_pose = Pose2D()
        self.behavior_primitive = ''
        self.controller = ''
        self.proxy = ''
        self.which_arm = ''
        self.push_time = 0.0
        self.precondition_method = ''
        # NOTE: Everything below not saved to disk, just computed for convenience
        self.push_angle = 0.0
        self.push_dist = 0.0

    def __str__(self):
        # TODO: Print more info here
        return (self.object_id +
                ' (' + self.proxy + ', ' + self.controller + ', ' + self.behavior_primitive + ', ' +
                self.which_arm + '):\n' +
                'init_centroid:\n' + str(self.init_centroid) + '\n'
                'init_orientation: ' + str(self.init_orientation))

class PushLearningIO:
    def __init__(self):
        self.data_out = None
        self.data_in = None

    def write_line(self, init_centroid, init_orientation, final_centroid,
                   final_orientation, goal_pose, behavior_primitive,
                   controller, proxy, which_arm, push_time, object_id,
                   precondition_method='centroid_push'):
        if self.data_out is None:
            rospy.logerr('Attempting to write to file that has not been opened.')
            return
        rospy.logdebug('Writing output line.\n')
        data_line = object_id+' '+str(init_centroid.x)+' '+str(init_centroid.y)+' '+str(init_centroid.z)+' '+\
            str(init_orientation)+' '+str(final_centroid.x)+' '+str(final_centroid.y)+' '+\
            str(final_centroid.z)+' '+str(final_orientation)+' '+\
            str(goal_pose.x)+' '+str(goal_pose.y)+' '+str(goal_pose.theta)+' '+\
            behavior_primitive+' '+controller+' '+proxy+' '+which_arm+' '+str(push_time)+' '+precondition_method+'\n'
        self.data_out.write(data_line)
        self.data_out.flush()

    def parse_line(self, line):
        if line.startswith('#'):
            return None
        l  = line.split()
        l.reverse()
        num_objs = len(l)
        push = PushTrial()
        push.object_id = l.pop()
        push.init_centroid.x = float(l.pop())
        push.init_centroid.y = float(l.pop())
        push.init_centroid.z = float(l.pop())
        push.init_orientation = float(l.pop())
        push.final_centroid.x = float(l.pop())
        push.final_centroid.y = float(l.pop())
        push.final_centroid.z = float(l.pop())
        push.final_orientation = float(l.pop())
        push.goal_pose.x = float(l.pop())
        push.goal_pose.y = float(l.pop())
        push.goal_pose.theta = float(l.pop())
        push.behavior_primitive = l.pop()
        push.controller = l.pop()
        push.proxy = l.pop()
        push.which_arm = l.pop()
        push.push_time = float(l.pop())
        if len(l) > 0:
            push.precondition_method = l.pop()
        else:
            push.precondition_method = 'push_centroid'
        push.push_angle = atan2(push.goal_pose.y-push.init_centroid.y,
                                push.goal_pose.x-push.init_centroid.x)
        push.push_dist = hypot(push.goal_pose.y-push.init_centroid.y,
                               push.goal_pose.x-push.init_centroid.x)
        return push

    def read_in_data_file(self, file_name):
        data_in = file(file_name, 'r')
        x = [self.parse_line(l) for l in data_in.readlines()]
        data_in.close()
        return filter(None, x)

    def open_out_file(self, file_name):
        self.data_out = file(file_name, 'a')
        self.data_out.write(_VERSION_LINE+'\n')
        self.data_out.write(_HEADER_LINE+'\n')
        self.data_out.flush()

    def close_out_file(self):
        self.data_out.close()

class PushLearningAnalysis:

    def __init__(self):
        self.raw_data = None
        self.io = PushLearningIO()
        self.compute_push_score = self.compute_push_error_xy
        self.xy_hash_precision = 20.0 # bins/meter
        self.num_angle_bins = 8
    #
    # Methods for computing marginals
    #
    def workspace_distribution(self):
        '''
        Method to find the best performing (on average) behavior_primitive as a function of (x,y,push_angle)
        '''
        self.score_push_trials(self.all_trials)
        loc_groups = self.group_trials_by_xy_and_push_angle(self.all_trials)

        # Group multiple trials of same push at a given location
        groups = []
        for i, group_key in enumerate(loc_groups):
            group = loc_groups[group_key]
            behavior_primitive_dict = {}
            for j,t in enumerate(group):
                opt_key = t.behavior_primitive
                try:
                    behavior_primitive_dict[opt_key].append(t)
                except KeyError:
                    behavior_primitive_dict[opt_key] = [t]
            # Average errors for each push key
            mean_group = []
            for opt_key in behavior_primitive_dict:
                mean_score = 0
                for push in behavior_primitive_dict[opt_key]:
                    mean_score += push.score
                mean_score = mean_score / float(len(behavior_primitive_dict[opt_key]))
                # Make a fake mean push
                mean_push = PushTrial()
                mean_push.score = mean_score
                mean_push.c_x = Point(0,0,0)
                mean_push.init_centroid.x, mean_push.init_centroid.y = self.hash_xy(
                    behavior_primitive_dict[opt_key][0].init_centroid.x,
                    behavior_primitive_dict[opt_key][0].init_centroid.y)
                mean_push.push_angle = self.hash_angle(
                    behavior_primitive_dict[opt_key][0].push_angle)
                mean_push.behavior_primitive = opt_key
                mean_group.append(mean_push)
            groups.append(mean_group)

        # Choose best push for each (angle, centroid) group
        best_pushes = []
        for i, group in enumerate(groups):
            min_score = 200.0 # Meters
            min_score_push = None
            for j, t in enumerate(group):
                if t.score < min_score:
                    min_score = t.score
                    min_score_push = t
            best_pushes.append(min_score_push)
        return best_pushes

    def object_distribution(self):
        '''
        Method to find the best performing (on average) behavior_primitive as a function of object_id
        '''
        return self.conditional_max_likelihood(self.group_trials_by_object_id,
                                               self.group_trials_by_behavior_primitive,
                                               self.get_mean_objcet_id_push)

    def angle_distribution(self):
        '''
        Method to find the best performing (on average) behavior_primitive as a function of object_id
        '''
        return self.conditional_max_likelihood(self.group_trials_by_push_angle,
                                               self.group_trials_by_behavior_primitive,
                                               self.get_mean_angle_push)

    def conditional_max_likelihood(self, trial_grouping_function,
                                   likelihood_arg_grouping_function,
                                   mean_push_function):
        '''
        '''
        self.score_push_trials(self.all_trials)
        trial_groups = trial_grouping_function(self.all_trials)

        # Group multiple trials of same type
        mean_groups = []
        for i, group_key in enumerate(trial_groups):
            trial_group = trial_groups[group_key]
            likelihood_arg_dict = likelihood_arg_grouping_function(trial_group)
            # Average errors for each push key
            mean_group = []
            for arg_key in likelihood_arg_dict:
                mean_score = 0
                for push in likelihood_arg_dict[arg_key]:
                    mean_score += push.score
                mean_score = mean_score / float(len(likelihood_arg_dict[arg_key]))
                mean_push = mean_push_function(likelihood_arg_dict[arg_key], arg_key, mean_score)
                mean_group.append(mean_push)
            mean_groups.append(mean_group)

        # Choose best push for each conditional group
        best_pushes = []
        for i, group in enumerate(mean_groups):
            min_score = 200000.0 # Meters
            min_score_push = None
            for j, t in enumerate(group):
                if t.score < min_score:
                    min_score = t.score
                    min_score_push = t
            best_pushes.append(min_score_push)
        return best_pushes

    def get_mean_objcet_id_push(self, arg_list, arg_key, mean_score):
        mean_push = PushTrial()
        mean_push.score = mean_score
        mean_push.c_x = Point(0,0,0)
        mean_push.object_id = arg_list[0].object_id
        mean_push.init_centroid.x, mean_push.init_centroid.y = self.hash_xy(
            arg_list[0].init_centroid.x, arg_list[0].init_centroid.y)
        mean_push.push_angle = self.hash_angle(arg_list[0].push_angle)
        mean_push.behavior_primitive = arg_key
        return mean_push

    def get_mean_angle_push(self, arg_list, arg_key, mean_score):
        mean_push = PushTrial()
        mean_push.score = mean_score
        mean_push.c_x = Point(0,0,0)
        mean_push.push_angle = self.hash_angle(arg_list[0].push_angle)
        mean_push.behavior_primitive = arg_key
        return mean_push

    #
    # Grouping methods
    #

    # TODO: Make a generic function where we just pass in the hash functions
    def group_trials_by_xy_and_push_angle(self, all_trials):
        '''
        Method to group all trials by initial table location and push angle
        '''
        groups = {}
        # Group scored pushes by push angle
        for t in all_trials:
            angle_key = self.hash_angle(t.push_angle)
            x_key, y_key = self.hash_xy(t.init_centroid.x, t.init_centroid.y)
            group_key = (x_key, y_key, angle_key)
            try:
                groups[group_key].append(t)
            except KeyError:
                groups[group_key] = [t]
        return groups

    def group_trials_by_push_angle(self, all_trials):
        '''
        Method to group all trials by initial table location and push angle
        '''
        groups = {}
        # Group scored pushes by push angle
        for t in all_trials:
            group_key = self.hash_angle(t.push_angle)
            try:
                groups[group_key].append(t)
            except KeyError:
                groups[group_key] = [t]
        return groups

    def group_trials_by_xy(self, all_trials):
        '''
        Method to group all trials by initial table location and push angle
        '''
        xy_dict = {}
        for t in all_trials:
            xy_key = self.hash_xy(t.init_centroid.x, t.init_centroid.y)
            # Check if x_key exists
            try:
                xy_dict[xy_key].append(t)
            except KeyError:
                xy_dict[xy_key] = [t]
        return xy_dict

    def group_trials_by_object_id(self, all_trials):
        '''
        Method to group all trials by object being pushed
        '''
        object_dict = {}
        # Group scored pushes by push angle
        for t in all_trials:
            object_key = t.object_id
            try:
                object_dict[object_key].append(t)
            except KeyError:
                object_dict[object_key] = [t]
        return object_dict

    def group_trials_by_behavior_primitive(self, trials):
        likelihood_arg_dict = {}
        for j,t in enumerate(trials):
            opt_key = t.behavior_primitive
            try:
                likelihood_arg_dict[opt_key].append(t)
            except KeyError:
                likelihood_arg_dict[opt_key] = [t]
        return likelihood_arg_dict
    #
    # Visualization functions
    #
    def visualize_push_choices(self, choices):
        # load in image from dropbox folder
        # TODO: Get new image of the table we use now
        disp_img = cv2.imread('/u/thermans/Dropbox/Data/choose_push/use_for_display.png')

        xy_groups = self.group_trials_by_xy(choices)

        # TODO: Draw all choices for a specific (x,y) together to get better matting of shadows
        for group_key in xy_groups:
            disp_img = self.draw_push_choices_on_image(xy_groups[group_key], disp_img)
        cv2.imshow('Chosen pushes', disp_img)
        cv2.imwrite('/u/thermans/Desktop/push_learn_out.png', disp_img)
        cv2.waitKey()

    def draw_push_choices_on_image(self, choices, img):
        # TODO: Double check where the bin centroid is
        # c.init_centroid.x += 0.5/self.xy_hash_precision
        # c.init_centroid.y += 0.5/self.xy_hash_precision

        # TODO: load in transform and camera parameters from saved info file
        K = np.matrix([[525, 0, 319.5, 0.0],
                       [0, 525, 239.5, 0.0],
                       [0, 0, 1, 0.0]])
        tl = np.asarray([-0.0115423, 0.441939, 0.263569])
        q = np.asarray([0.693274, -0.685285, 0.157732, 0.157719])
        num_downsamples = 0
        # TODO: Save table height in trial data
        table_height = -0.3
        P_w = np.matrix([[choices[0].init_centroid.x], [choices[0].init_centroid.y],
                         [table_height], [1.0]])
        # Transform choice location into camera frame
        T = (np.matrix(tr.translation_matrix(tl)) *
             np.matrix(tr.quaternion_matrix(q)))
        P_c = T*P_w
        # Transform camera point into image frame
        P_i = K*P_c
        P_i = P_i / P_i[2]
        u = P_i[0]/pow(2,num_downsamples)
        v = P_i[1]/pow(2,num_downsamples)
        # Draw circle for the location
        radius = 15
        cv2.circle(img, (u,v), 3, [0.0,0.0,0.0],3)
        cv2.circle(img, (u,v), 3, [255.0,255.0,255.0], 1)
        # Draw Shadows for all angles
        for c in choices:
            end_point = (u+cos(c.push_angle)*(radius), v+sin(c.push_angle)*(radius))
            cv2.line(img, (u,v), end_point, [0.0,0.0,0.0],3)
        for c in choices:
            # Choose color by push type
            if c.behavior_primitive == GRIPPER_PUSH:
                color = [255.0, 0.0, 0.0] # Blue
            elif c.behavior_primitive == GRIPPER_SWEEP:
                color = [0.0, 255.0, 0.0] # Green
            elif c.behavior_primitive == OVERHEAD_PUSH:
                color = [0.0, 0.0, 255.0] # Red
            elif c.behavior_primitive == GRIPPER_PULL:
                color = [0.0, 255.0, 255.0] # Yellow
            # Draw line depicting the angle
            end_point = (u+cos(c.push_angle)*(radius), v+sin(c.push_angle)*(radius))
            cv2.line(img, (u,v), end_point, color)
        return img

    #
    # IO Functions
    #
    def read_in_push_trials(self, file_name):
        self.all_trials = self.io.read_in_data_file(file_name)

    def output_loc_push_choices(self, choices):
        print "Loc push choices:"
        for c in choices:
            self.output_push_choice(c, ['init_centroid.x','init_centroid.y','push_angle'],
                                    'behavior_primitive')

    def output_obj_push_choices(self, choices):
        print "Object push choices:"
        for c in choices:
            self.output_push_choice(c, 'object_id', 'behavior_primitive')

    def output_angle_push_choices(self, choices):
        print "Angle push choices:"
        for c in choices:
            self.output_push_choice(c,'push_angle', 'behavior_primitive')

    def output_push_choice(self, c, conditional_value, arg_value):
        '''
        Method takes strings [or lists of strings] of what attributes of PushTrail instance c to display
        '''
        push_angle = self.hash_angle(c.push_angle)

        if type(conditional_value) == list:
            conditional_str = ''
            for val in conditional_value:
                conditional_str += str(get_attr(c,val))+', '
            conditional_str = conditional_str[:-2]
        else:
            conditional_str = str(get_attr(c,conditional_value))
        if type(arg_value) == list:
            arg_str = ''
            for val in arg_value:
                arg_str += str(get_attr(c,val))+', '
            arg_str = arg_str[:-2]
        else:
            arg_str = str(get_attr(c,arg_value))
        print 'Choice for (' +  conditional_str + '): '+ arg_str+ ' : ' + str(c.score)

    #
    # Hashing Functions
    #
    def hash_angle(self, theta):
        # Group different pushes by push_angle
        bin = int((theta + pi)/(2.0*pi)*self.num_angle_bins)
        bin = max(min(bin, self.num_angle_bins-1), 0)
        return -pi+(2.0*pi)*float(bin)/self.num_angle_bins

    def hash_xy(self, x,y):
        # Group different pushes by push_angle
        x_prime = round(x*self.xy_hash_precision)/self.xy_hash_precision
        y_prime = round(y*self.xy_hash_precision)/self.xy_hash_precision
        return (x_prime, y_prime)

    def hash_behavior_primitive(self, push):
        return (push.which_arm,push.behavior_primitive)

    def unhash_behavior_primitive_key(self, opt_key):
        return (opt_key[0], opt_key[1])

    #
    # Scoring Functions
    #
    def score_push_trials(self, trials):
        # Get error scores for each push
        for t in trials:
            t.score = self.compute_push_score(t)

    def compute_push_error_xy(self, push):
        err_x = push.goal_pose.x - push.final_centroid.x
        err_y = push.goal_pose.y - push.final_centroid.y
        return err_x*err_x+err_y*err_y

    def compute_push_error_push_dist_diff(self, push):
        desired_x = push.init_centroid.x + cos(push.push_angle)*push.push_dist
        desired_y = push.init_centroid.y + sin(push.push_angle)*push.push_dist
        d_x = push.final_centroid.x - push.init_centroid.x
        d_y = push.final_centroid.y - push.init_centroid.y
        actual_dist = sqrt(d_x*d_x + d_y*d_y)
        return fabs(actual_dist - push.push_dist)

if __name__ == '__main__':
    # TODO: Add more options to command line
    if len(sys.argv) > 1:
        data_path = str(sys.argv[1])
    else:
        print 'Usage:',sys.argv[0],'input_file'
        quit()

    pla = PushLearningAnalysis()
    pla.read_in_push_trials(data_path)

    # TODO: Use command line arguments to choose these
    angle_pushes = pla.angle_distribution()

    workspace_pushes = pla.workspace_distribution()

    object_pushes = pla.object_distribution()

    pla.output_loc_push_choices(workspace_pushes)
    pla.output_obj_push_choices(object_pushes)
    pla.output_angle_push_choices(angle_pushes)

    pla.visualize_push_choices(workspace_pushes)
