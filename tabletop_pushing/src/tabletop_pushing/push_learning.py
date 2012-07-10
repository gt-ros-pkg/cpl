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
from geometry_msgs.msg import Point
from math import sin, cos, pi, sqrt, fabs
import cv2
import tf.transformations as tr
import numpy as np
import sys
import rospy

_HEADER_LINE = '# c_x c_y c_z theta push_opt arm c_x\' c_y\' c_z\' push_dist high_init push_time'

class PushTrial:
    def __init__(self):
        self.c_x = None
        self.c_x_prime = None
        self.push_angle = None
        self.push_opt = None
        self.arm = None
        self.push_dist = None
        self.high_init = 0
        self.push_time = 0.0
        self.score = None

    def __str__(self):
        return str((self.c_x, self.push_angle, self.push_opt, self.arm,
                    self.c_x_prime, self.push_dist, self.high_init, self.push_time))

class PushLearningAnalysis:

    def __init__(self):
        self.raw_data = None
        self.io = PushLearningIO()
        self.compute_push_score = self.compute_push_error_xy
        # self.compute_push_score = self.compute_push_error_push_dist_diff
        self.xy_hash_precision = 10.0 # bins/meter
        self.num_angle_bins = 8

    def determine_best_pushes(self, file_name):
        all_trials = self.read_in_push_trials(file_name)
        loc_groups = self.group_trials(all_trials)

        # Group multiple trials of same push at a given location
        groups = []
        for i, group in enumerate(loc_groups):
            push_opt_dict = {}
            for j,t in enumerate(group):
                opt_key = self.hash_push_opt(t)
                try:
                    push_opt_dict[opt_key].append(t)
                except KeyError:
                    push_opt_dict[opt_key] = [t]
            # Average errors for each push key
            mean_group = []
            for opt_key in push_opt_dict:
                mean_score = 0
                for push in push_opt_dict[opt_key]:
                    mean_score += push.score
                mean_score = mean_score / float(len(push_opt_dict[opt_key]))
                # Make a fake mean push
                mean_push = PushTrial()
                mean_push.score = mean_score
                mean_push.c_x = Point(0,0,0)
                mean_push.c_x.x, mean_push.c_x.y = self.hash_xy(
                    push_opt_dict[opt_key][0].c_x.x,
                    push_opt_dict[opt_key][0].c_x.y)
                mean_push.push_angle = self.hash_angle(
                    push_opt_dict[opt_key][0].push_angle)
                mean_push.arm, mean_push.push_opt = self.unhash_opt_key(opt_key)
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

    def group_trials(self, all_trials):
        # Get error scores for each push
        # Group scored pushes by push angle
        angle_dict = {}
        for t in all_trials:
            t.score = self.compute_push_score(t)
            angle_key = self.hash_angle(t.push_angle)
            try:
                angle_dict[angle_key].append(t)
            except KeyError:
                angle_dict[angle_key] = [t]
        # Group different pushes by start centroid
        groups = []
        i = 0
        for angle_key in angle_dict:
            x_dict = {}
            for t in angle_dict[angle_key]:
                x_key, y_key = self.hash_xy(t.c_x.x, t.c_x.y)
                # Check if x_key exists
                try:
                    cur_y_dict = x_dict[x_key]
                    # Check if y_key exists
                    try:
                        x_dict[x_key][y_key].append(t)
                    except KeyError:
                        x_dict[x_key][y_key] = [t]
                except KeyError:
                    y_dict = {y_key:[t]}
                    x_dict[x_key] = y_dict
            # Flatten groups
            for x_key in x_dict:
                for y_key in x_dict[x_key]:
                    groups.append(x_dict[x_key][y_key])
                    i += 1
        return groups

    def visualize_push_choices(self, choices):
        # load in image from dropbox folder
        disp_img = cv2.imread('/u/thermans/Dropbox/Data/choose_push/test_disp.png')
        world_min_x = 0.2
        world_max_x = 1.0
        world_min_y = -0.5
        world_max_y = 0.5
        world_x_dist = world_max_x - world_min_x
        world_y_dist = world_max_y - world_min_y
        world_x_bins = world_x_dist*self.xy_hash_precision
        world_y_bins = world_y_dist*self.xy_hash_precision
        for c in choices:
            start_x, start_y = self.hash_xy(c.c_x.x, c.c_x.y)
            push_angle = self.hash_angle(c.push_angle)
            rospy.loginfo('Choice for (' + str(start_x) + ', ' + str(start_y) +
                          ', ' + str(push_angle) + '): (' + str(c.arm) + ', ' +
                          str(c.push_opt) + ') : ' + str(c.score))
            disp_img = self.draw_push_choice_on_image(c, disp_img)
        cv2.imshow('Chosen pushes', disp_img)
        cv2.imwrite('/u/thermans/Desktop/push_learn_out.png', disp_img)
        cv2.waitKey()

    def draw_push_choice_on_image(self, c, img):
        # TODO: Double check where the bin centroid is
        # c.c_x.x -= 0.5/self.xy_hash_precision
        # c.c_x.y -= 0.5/self.xy_hash_precision

        # TODO: load in transform and camera parameters from saved info file
        K = np.matrix([[525, 0, 319.5, 0.0],
                       [0, 525, 239.5, 0.0],
                       [0, 0, 1, 0.0]])
        tl = np.asarray([-0.0115423, 0.441939, 0.263569])
        q = np.asarray([0.693274, -0.685285, 0.157732, 0.157719])
        num_downsamples = 1
        # TODO: Save table height in trial data
        table_height = -0.3
        P_w = np.matrix([[c.c_x.x], [c.c_x.y], [table_height], [1.0]])
        # Transform choice location into camera frame
        T = (np.matrix(tr.translation_matrix(tl)) *
             np.matrix(tr.quaternion_matrix(q)))
        P_c = T*P_w
        # Transform camera point into image frame
        P_i = K*P_c
        P_i = P_i / P_i[2]
        u = P_i[0]/pow(2,num_downsamples)
        v = P_i[1]/pow(2,num_downsamples)
        # Choose color by push type
        if c.arm == 'l':
            if c.push_opt == 0: # Gripper push
                color = [255.0, 0.0, 0.0] # Blue
            elif c.push_opt == 1: # Sweep
                color = [0.0, 255.0, 0.0] # Green
            else: # Overhead push
                color = [0.0, 0.0, 255.0] # Red
        else:
            if c.push_opt == 0:# Gripper push
                color = [255.0, 0.0, 255.0] # Magenta
            elif c.push_opt == 1: # Sweep
                color = [0.0, 255.0, 255.0] # Yellow
            else: # Overhead push
                color = [255.0, 255.0, 0.0] # Cyan

        # Draw line depicting the angle
        radius = 7
        end_point = (u+cos(c.push_angle)*radius, v+sin(c.push_angle)*radius)
        cv2.line(img, (u,v), end_point, color)
        cv2.circle(img, (u,v), radius, [0.0,0.0,0.0])
        return img

    #
    # IO Functions
    #
    def read_in_push_trials(self, file_name):
        return self.io.read_in_data_file(file_name)

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

    def hash_push_opt(self, push):
        return push.arm+str(push.push_opt)

    def unhash_opt_key(self, opt_key):
        return (opt_key[0], int(opt_key[1]))

    #
    # Scoring Functions
    #
    def compute_push_error_xy(self, push):
        desired_x = push.c_x.x + cos(push.push_angle)*push.push_dist
        desired_y = push.c_x.y + sin(push.push_angle)*push.push_dist
        err_x = desired_x - push.c_x_prime.x
        err_y = desired_y - push.c_x_prime.y
        return err_x*err_x+err_y*err_y

    def compute_push_error_push_dist_diff(self, push):
        desired_x = push.c_x.x + cos(push.push_angle)*push.push_dist
        desired_y = push.c_x.y + sin(push.push_angle)*push.push_dist
        d_x = push.c_x_prime.x - push.c_x.x
        d_y = push.c_x_prime.y - push.c_x.y
        actual_dist = sqrt(d_x*d_x + d_y*d_y)
        return fabs(actual_dist - push.push_dist)

class PushLearningIO:
    def __init__(self):
        self.data_out = None
        self.data_in = None

    def write_line(self, c_x, push_angle, push_opt, arm, c_x_prime, push_dist,
                   high_init=False, push_time=0.0):
        if self.data_out is None:
            rospy.logerr('Attempting to write to file that has not been opened.')
            return
        rospy.loginfo('Writing output line.\n')
        # c_x c_y c_z theta push_opt arm c_x' c_y' c_z' push_dist
        data_line = str(c_x.x)+' '+str(c_x.y)+' '+str(c_x.z)+' '+\
            str(push_angle)+' '+str(push_opt)+' '+str(arm)+' '+\
            str(c_x_prime.x)+' '+str(c_x_prime.y)+' '+str(c_x_prime.z)+' '+\
            str(push_dist)+' '+str(int(high_init))+' '+str(push_time)+'\n'
        self.data_out.write(data_line)
        self.data_out.flush()

    def parse_line(self, line):
        if line.startswith('#'):
            return None
        l  = line.split()
        # c_x c_y c_z theta push_opt arm c_x' c_y' c_z' push_dist
        push = PushTrial()
        push.c_x = Point(float(l[0]), float(l[1]), float(l[2]))
        push.push_angle = float(l[3])
        push.push_opt = int(l[4])
        push.arm = l[5]
        push.c_x_prime = Point(float(l[6]),float(l[7]),float(l[8]))
        push.push_dist = float(l[9])
        if len(l) > 10:
            push.high_init = int(l[10])
        if len(l) > 11:
            push.push_time = float(l[11])
        return push

    def read_in_data_file(self, file_name):
        data_in = file(file_name, 'r')
        x = [self.parse_line(l) for l in data_in.readlines()]
        data_in.close()
        return filter(None, x)

    def open_out_file(self, file_name):
        self.data_out = file(file_name, 'a')
        self.data_out.write(_HEADER_LINE+'\n')
        self.data_out.flush()

    def close_out_file(self):
        self.data_out.close()

if __name__ == '__main__':
    # TODO: Read command line arguments for data file and metric to use
    pla = PushLearningAnalysis()
    if len(sys.argv) > 1:
        data_path = str(sys.argv[1])
    else:
        data_path = '/u/thermans/Dropbox/Data/choose_push/batch_out0.txt'
    best_pushes = pla.determine_best_pushes(data_path)
    pla.visualize_push_choices(best_pushes)
