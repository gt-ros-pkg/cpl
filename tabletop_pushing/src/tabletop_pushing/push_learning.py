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
        self.controller = ''
        self.behavior_primitive = ''
        self.proxy = ''
        self.which_arm = ''
        self.precondition_method = ''
        self.init_centroid = Point()
        self.init_orientation = 0.0
        self.final_centroid = Point()
        self.final_orientation = 0.0
        self.goal_pose = Pose2D()
        self.push_time = 0.0
        # NOTE: Everything below not saved to disk, just computed for convenience
        self.push_angle = 0.0
        self.push_dist = 0.0
        self.continuation = False

    def __str__(self):
        return (self.object_id +
                ' (' + self.proxy + ', ' + self.controller + ', ' + self.behavior_primitive + ', ' +
                self.which_arm + '_arm'+', ' +self.precondition_method+'):\n' +
                'init_centroid:\n' + str(self.init_centroid) + '\n'+
                'init_orientation: ' + str(self.init_orientation) +'\n'+
                'final_centroid:\n' + str(self.final_centroid) + '\n'+
                'final_orientation: ' + str(self.final_orientation) + '\n'+
                'goal_pose:\n' + str(self.goal_pose) + '\n'+
                'push_time: ' + str(self.push_time))

def compare_pushes(a, b):
    if a.score < b.score:
        return -1
    elif a.score > b.score:
        return 1
    else:
        return 0

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
        x = filter(None, x)
        return self.link_repeat_trials(x)

    def link_repeat_trials(self, trials):
        for i in range(1,len(trials)):
            if (trials[i].goal_pose.y == trials[i-1].goal_pose.y and
                trials[i].goal_pose.x == trials[i-1].goal_pose.x and
                trials[i].behavior_primitive == trials[i-1].behavior_primitive and
                trials[i].proxy == trials[i-1].proxy and
                trials[i].controller == trials[i-1].controller):
                if trials[i].which_arm == trials[i-1].which_arm:
                    # print 'Continuation with different arm'
                    pass
                else:
                    trials[i].continuation = True
        return trials

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
        self.xy_hash_precision = 20.0 # bins/meter
        self.num_angle_bins = 8
    #
    # Methods for computing marginals
    #
    def workspace_ranking(self):
        likelihood_arg=['behavior_primitive','proxy','controller','which_arm']
        # likelihood_arg=['which_arm']
        score_fnc=self.compute_change_in_push_error_xy
        workspace_ranks = self.rank_avg_pushes(trial_hash_fnc=self.hash_push_xy_angle,
                                               likelihood_arg=likelihood_arg,
                                               mean_push_fnc=self.get_mean_workspace_push,
                                               score_fnc=score_fnc)
        print "Workspace push rankings:"
        for ranks in workspace_ranks:
            self.output_push_ranks(ranks,
                                   ['init_centroid.x','init_centroid.y','push_angle'],
                                   likelihood_arg, n=3)
        return workspace_ranks

    def object_ranking(self, use_raw=False):
        '''
        Method to find the best performing (on average) behavior_primitive as a function of object_id
        '''
        cond_arg='object_id'
        likelihood_arg=['behavior_primitive','proxy','controller']
        # score_fnc=self.compute_normalized_push_time
        score_fnc=self.compute_push_error_xy
        score_fnc=self.compute_change_in_push_error_xy
        if use_raw:
            rankings = self.rank_raw_pushes(trial_grouping_arg=cond_arg,
                                            score_fnc=score_fnc)
            print "\Raw object behavior rankings:"
            for ranks in rankings:
                self.output_push_ranks(ranks, cond_arg, likelihood_arg, n=5)
        else:
            rankings = self.rank_avg_pushes(trial_grouping_arg=cond_arg,
                                            likelihood_arg=likelihood_arg,
                                            score_fnc=score_fnc)
            print "\nObject behavior rankings:"
            for ranks in rankings:
                self.output_push_ranks(ranks, cond_arg, likelihood_arg, n=5)
        return rankings

    def object_proxy_ranking(self):
        '''
        Method to find the best performing (on average) behavior_primitive as a function of object_id
        '''
        cond_arg='object_id'
        likelihood_arg='proxy'
        # score_fnc=self.compute_normalized_push_time
        score_fnc=self.compute_push_error_xy
        rankings = self.rank_avg_pushes(trial_grouping_arg=cond_arg,
                                        likelihood_arg=likelihood_arg,
                                        score_fnc=score_fnc)
        print "\nObject proxy rankings:"
        for ranks in rankings:
            self.output_push_ranks(ranks, cond_arg, likelihood_arg, n=3)
        return rankings

    def angle_ranking(self):
        '''
        Method to find the best performing (on average) behavior_primitive as a function of push_angle
        '''
        likelihood_arg=['behavior_primitive','proxy','controller']
        angle_ranks = self.rank_avg_pushes(
            trial_hash_fnc=self.hash_push_angle, likelihood_arg=likelihood_arg,
            mean_push_fnc=self.get_mean_angle_push)
        print "Angle push rankings:"
        for rank in angle_ranks:
            self.output_push_ranks(rank, 'push_angle', likelihood_arg, n=3)
        return angle_ranks

    def rank_avg_pushes(self, trial_hash_fnc=None, trial_grouping_arg=None,
                        likelihood_arg=None, mean_push_fnc=None,score_fnc=None):
        '''
        '''
        # Compute scores
        if score_fnc is None:
            score_fnc = self.compute_push_error_xy
        for t in self.all_trials:
            t.score = score_fnc(t)
        # Group objects based on conditional variables
        if trial_hash_fnc is not None:
            trial_groups = self.group_trials(self.all_trials, hash_function=trial_hash_fnc)
        else:
            trial_groups = self.group_trials(self.all_trials, trial_grouping_arg)

        # Group multiple trials of same type
        mean_groups = []
        for i, group_key in enumerate(trial_groups):
            trial_group = trial_groups[group_key]
            likelihood_arg_dict = self.group_trials(trial_group, likelihood_arg)
            # Average errors for each push key
            mean_group = []
            for arg_key in likelihood_arg_dict:
                if trial_grouping_arg is not None:
                    mean_push = self.get_mean_push(likelihood_arg_dict[arg_key],
                                                   trial_grouping_arg, likelihood_arg)
                else:
                    mean_push = mean_push_fnc(likelihood_arg_dict[arg_key])
                mean_group.append(mean_push)
            mean_groups.append(mean_group)

        # Choose best push for each conditional group
        ranked_pushes = []
        for i, group in enumerate(mean_groups):
            group.sort(compare_pushes)
            ranked_pushes.append(group)
        return ranked_pushes


    def rank_raw_pushes(self, trial_hash_fnc=None, trial_grouping_arg=None, score_fnc=None):
        '''
        '''
        # Compute scores
        if score_fnc is None:
            score_fnc = self.compute_push_error_xy
        for t in self.all_trials:
            t.score = score_fnc(t)
        # Group objects based on conditional variables
        if trial_hash_fnc is not None:
            trial_groups = self.group_trials(self.all_trials, hash_function=trial_hash_fnc)
        else:
            trial_groups = self.group_trials(self.all_trials, trial_grouping_arg)

        # Group multiple trials of same type
        ranked_pushes = []
        for i, group_key in enumerate(trial_groups):
            trial_group = trial_groups[group_key]
            trial_group.sort(compare_pushes)
            ranked_pushes.append(trial_group)
        return ranked_pushes

    def get_mean_push(self, arg_list, trial_grouping_arg, likelihood_arg):
        mean_push = PushTrial()
        mean_score, score_var = self.mean_and_variance(arg_list)
        mean_push.score = mean_score
        mean_push.var = score_var
        if type(trial_grouping_arg) is not list:
            trial_grouping_arg = [trial_grouping_arg]
        if type(likelihood_arg) is not list:
            likelihood_arg = [likelihood_arg]
        for arg in trial_grouping_arg:
            setattr(mean_push,arg, get_attr(arg_list[0],arg))
        for arg in likelihood_arg:
            setattr(mean_push,arg, get_attr(arg_list[0],arg))
        return mean_push

    def get_mean_angle_push(self, arg_list):
        mean_push = PushTrial()
        mean_score, score_var = self.mean_and_variance(arg_list)
        mean_push.score = mean_score
        mean_push.var = score_var
        mean_push.push_angle = self.hash_angle(arg_list[0].push_angle)
        mean_push.behavior_primitive = arg_list[0].behavior_primitive
        mean_push.controller = arg_list[0].controller
        mean_push.proxy = arg_list[0].proxy
        return mean_push

    def get_mean_workspace_push(self, arg_list):
        mean_push = PushTrial()
        mean_score, score_var = self.mean_and_variance(arg_list)
        mean_push.score = mean_score
        mean_push.var = score_var
        mean_push.init_centroid.x, mean_push.init_centroid.y = self.hash_xy(
            arg_list[0].init_centroid.x, arg_list[0].init_centroid.y)
        mean_push.push_angle = self.hash_angle(arg_list[0].push_angle)
        mean_push.behavior_primitive = arg_list[0].behavior_primitive
        mean_push.controller = arg_list[0].controller
        mean_push.proxy = arg_list[0].proxy
        mean_push.which_arm = arg_list[0].which_arm
        return mean_push

    def mean_and_variance(self, pushes):
        n = len(pushes)
        mean_score = 0
        for push in pushes:
            mean_score += push.score
        mean_score = mean_score / n

        if n <= 1:
            return (mean_score, 0)

        var_sum = 0
        for push in pushes:
            var_sum += push.score**2
        score_var = (var_sum-n*mean_score**2)/(n-1)

        return (mean_score, score_var)
    #
    # Grouping methods
    #
    def group_trials(self, all_trials, hash_attribute=None, hash_function=None):
        group_dict = {}
        # Group scored pushes by push angle
        for t in all_trials:
            if hash_function is not None:
                group_key = hash_function(t)
            elif type(hash_attribute) == list and len(hash_attribute) > 1:
                group_key = []
                for attr in hash_attribute:
                    group_key.append(get_attr(t,attr))
                group_key = tuple(group_key)
            elif type(hash_attribute) == list:
                group_key = get_attr(t, hash_attribute[0])
            else:
                group_key = get_attr(t, hash_attribute)
            try:
                group_dict[group_key].append(t)
            except KeyError:
                group_dict[group_key] = [t]
        return group_dict

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
        return self.group_trials(all_trials, hash_function=self.hash_push_angle)

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
    #
    # Visualization functions
    #
    def visualize_push_choices(self, ranks, score_threshold=None):
        choices = []
        for rank in ranks:
            choices.append(rank[0])
        # load in image from dropbox folder
        disp_img = cv2.imread('/u/thermans/Dropbox/Data/choose_push/use_for_display.png')

        xy_groups = self.group_trials_by_xy(choices)

        for group_key in xy_groups:
            disp_img = self.draw_push_choices_on_image(xy_groups[group_key], disp_img,
                                                       score_threshold=score_threshold)
        cv2.imshow('Chosen pushes', disp_img)
        cv2.imwrite('/u/thermans/Desktop/push_learn_out.png', disp_img)
        cv2.waitKey()

    def draw_push_choices_on_image(self, choices, img, score_threshold=None):
        # TODO: Double check that this is all correct
        # TODO: load in transform and camera parameters from saved info file
        K = np.matrix([[525, 0, 319.5, 0.0],
                       [0, 525, 239.5, 0.0],
                       [0, 0, 1, 0.0]])
        tl = np.asarray([-0.0115423, 0.441939, 0.263569])
        q = np.asarray([0.693274, -0.685285, 0.157732, 0.157719])
        num_downsamples = 0
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

        valid_choices = 0
        for c in choices:
            if score_threshold is None:
                valid_choices += 1
            elif c.score < score_threshold:
                valid_choices += 1
        if valid_choices == 0:
            return img
        # Draw circle for the location
        radius = 13
        cv2.circle(img, (u,v), 3, [0.0,0.0,0.0],3)
        cv2.circle(img, (u,v), 3, [255.0,255.0,255.0], 1)
        # Draw Shadows for all angles
        for c in choices:
            if score_threshold is None:
                pass
            elif c.score > score_threshold:
                continue
            end_point = (u-sin(c.push_angle)*(radius), v-cos(c.push_angle)*(radius))
            color = [0.0, 0.0, 0.0] # Black
            cv2.line(img, (u,v), end_point, color,3)
        for c in choices:
            if score_threshold is None:
                pass
            elif c.score > score_threshold:
                continue
            # Choose color by push type
            if c.which_arm == 'l':
                if c.behavior_primitive == GRIPPER_PUSH:
                    color = [255.0, 0.0, 0.0] # Blue
                elif c.behavior_primitive == GRIPPER_SWEEP:
                    color = [0.0, 255.0, 0.0] # Green
                elif c.behavior_primitive == OVERHEAD_PUSH:
                    color = [0.0, 0.0, 255.0] # Red
                elif c.behavior_primitive == GRIPPER_PULL:
                    color = [0.0, 255.0, 255.0] # Yellow
                # color = [0.0, 255.0, 0.0] # Green
            else:
                if c.behavior_primitive == GRIPPER_PUSH:
                    color = [128.0, 0.0, 128.0] # Magenta
                elif c.behavior_primitive == GRIPPER_SWEEP:
                    color = [0.0, 64.0, 129.0] # Brown
                elif c.behavior_primitive == OVERHEAD_PUSH:
                    color = [128.0, 128.0, 0.0] # Cyan
                elif c.behavior_primitive == GRIPPER_PULL:
                    color = [200.0, 200.0, 200.0] # White
                # color = [0.0, 0.0, 255.0] # Red
            # Draw line depicting the angle
            end_point = (u-sin(c.push_angle)*(radius), v-cos(c.push_angle)*(radius))
            cv2.line(img, (u,v), end_point, color)
        return img

    def visualize_angle_push_choices(self, ranks, score_threshold=None):
        choices = []
        for rank in ranks:
            choices.append(rank[0])
        # load in image from dropbox folder
        disp_img = cv2.imread('/u/thermans/Dropbox/Data/choose_push/use_for_display.png')

        # TODO: Create a single group and display for all (x,y) locs

        for x in x_locs:
            for y in y_locs:
                for c in choices:
                    c.init_centroid.x = x
                    c.init_centroid.y = y
                    disp_img = self.draw_push_choices_on_image(choices,
                                                               disp_img,
                                                               score_threshold=score_threshold)
        cv2.imshow('Chosen pushes', disp_img)
        cv2.imwrite('/u/thermans/Desktop/push_learn_angle_out.png', disp_img)
        cv2.waitKey()

    #
    # IO Functions
    #
    def read_in_push_trials(self, file_name):
        self.all_trials = self.io.read_in_data_file(file_name)

    def output_push_choice(self, c, conditional_value, arg_value):
        '''
        Method takes strings [or lists of strings] of what attributes of PushTrail instance c to display
        '''
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
        print 'Choice for (' +  conditional_str + '): ('+ arg_str+ ') : ' + str(c.score) + ', ' + str(c.var)

    def output_push_ranks(self, ranks, conditional_value, arg_value, n=3):
        '''
        Method takes strings [or lists of strings] of what attributes of PushTrail instance c to display
        '''
        if len(ranks) == 0:
            return
        if type(conditional_value) == list:
            conditional_str = ''
            for val in conditional_value:
                conditional_str += str(get_attr(ranks[0], val))+', '
            conditional_str = conditional_str[:-2]
        else:
            conditional_str = str(get_attr(ranks[0], conditional_value))
        print 'Ranks for (' +  conditional_str + '):'
        for i, c in enumerate(ranks):
            if i >= n:
                break
            if type(arg_value) == list:
                arg_str = ''
                for val in arg_value:
                    arg_str += str(get_attr(c,val))+', '
                arg_str = arg_str[:-2]
            else:
                arg_str = str(get_attr(c,arg_value))
            print '\t ('+ arg_str+ ') : ' + str(c.score) + ', ' + str(c.var)

    def output_loc_push_choices(self, choices):
        print "Loc push choices:"
        for c in choices:
            self.output_push_choice(c, ['init_centroid.x','init_centroid.y','push_angle'],
                                    ['behavior_primitive','controller','proxy'])

    #
    # Hashing Functions
    #

    def hash_push_angle(self, push):
        return self.hash_angle(push.push_angle)

    def hash_push_xy_angle(self, push):
        x_prime, y_prime = self.hash_xy(push.init_centroid.x, push.init_centroid.y)
        return (x_prime, y_prime, self.hash_angle(push.push_angle))

    def hash_angle(self, theta):
        # Group different pushes by push_angle
        bin = int((theta + pi)/(2.0*pi)*self.num_angle_bins)
        bin = max(min(bin, self.num_angle_bins-1), 0)
        return -pi+(2.0*pi)*float(bin)/self.num_angle_bins

    def hash_xy(self, x,y):
        # Group different pushes by push_angle
        x_prime = (round(x*self.xy_hash_precision)-1)/self.xy_hash_precision
        y_prime = (round(y*self.xy_hash_precision)-1)/self.xy_hash_precision
        return (x_prime, y_prime)

    #
    # Scoring Functions
    #

    def compute_push_error_xy(self, push):
        err_x = push.goal_pose.x - push.final_centroid.x
        err_y = push.goal_pose.y - push.final_centroid.y
        return err_x*err_x+err_y*err_y

    def compute_change_in_push_error_xy(self, push):
        init_x_error = push.goal_pose.x - push.init_centroid.x
        init_y_error = push.goal_pose.y - push.init_centroid.y
        final_x_error = push.goal_pose.x - push.final_centroid.x
        final_y_error = push.goal_pose.y - push.final_centroid.y

        if push.final_centroid == 0.0 and push.final_centroid.y == 0.0:
            print 'Knocked off object'
            return 100.0
        init_error = hypot(init_x_error, init_y_error)
        final_error = hypot(final_x_error, final_y_error)
        return final_error / init_error

    def compute_push_error_push_dist_diff(self, push):
        desired_x = push.init_centroid.x + cos(push.push_angle)*push.push_dist
        desired_y = push.init_centroid.y + sin(push.push_angle)*push.push_dist
        d_x = push.final_centroid.x - push.init_centroid.x
        d_y = push.final_centroid.y - push.init_centroid.y
        actual_dist = sqrt(d_x*d_x + d_y*d_y)
        return fabs(actual_dist - push.push_dist)

    def compute_normalized_push_time(self, push):
        return push.push_time / push.push_dist

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
    workspace_ranks = pla.workspace_ranking()
    # angle_ranks = pla.angle_ranking()
    pla.object_ranking()
    pla.object_proxy_ranking()
    pla.visualize_push_choices(workspace_ranks, 1.0)
