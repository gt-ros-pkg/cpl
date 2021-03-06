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
import rospy
import actionlib
import hrl_pr2_lib.pr2 as pr2
import hrl_lib.tf_utils as tfu
import tf
import numpy as np
from tabletop_pushing.srv import *
from tabletop_pushing.msg import *
from math import sin, cos, pi, fabs, sqrt, hypot
import sys
from select import select
from push_learning import PushLearningIO
from geometry_msgs.msg import Pose2D
import time
import random
from push_primitives import *

_OFFLINE_WITH_FEEDBACK_CONTROL = False
_OFFLINE = False
_USE_LEARN_IO = True
_TEST_START_POSE = False
_USE_FIXED_GOAL = False

def quit_code(code_in):
    return code_in.lower().strip().startswith('q')


class TabletopExecutive:

    def __init__(self, use_singulation, use_learning):
        self.previous_rand_pose = None

        rospy.init_node('tabletop_executive_node')
        self.min_push_dist = rospy.get_param('~min_push_dist', 0.07)
        self.max_push_dist = rospy.get_param('~mix_push_dist', 0.3)
        self.use_overhead_x_thresh = rospy.get_param('~use_overhead_x_thresh', 0.55)
        self.use_sweep_angle_thresh = rospy.get_param('~use_sweep_angle_thresh', pi*0.4)
        self.use_pull_angle_thresh = rospy.get_param('~use_sweep_angle_thresh', pi*0.525)
        self.use_same_side_y_thresh = rospy.get_param('~use_same_side_y_thresh', 0.3)
        self.use_same_side_x_thresh = rospy.get_param('~use_same_side_x_thresh', 0.8)

        self.gripper_offset_dist = rospy.get_param('~gripper_push_offset_dist', 0.05)
        self.gripper_start_z = rospy.get_param('~gripper_push_start_z', -0.29)

        self.pincher_offset_dist = rospy.get_param('~pincher_push_offset_dist', 0.05)
        self.pincher_start_z = rospy.get_param('~pincher_push_start_z', -0.29)

        self.sweep_face_offset_dist = rospy.get_param('~gripper_sweep_face_offset_dist', 0.03)
        self.sweep_wrist_offset_dist = rospy.get_param('~gripper_sweep_wrist_offset_dist', 0.01)
        self.sweep_start_z = rospy.get_param('~gripper_sweep_start_z', -0.30)

        self.overhead_offset_dist = rospy.get_param('~overhead_push_offset_dist', 0.05)
        self.overhead_start_z = rospy.get_param('~overhead_push_start_z', -0.305)

        self.gripper_pull_offset_dist = rospy.get_param('~gripper_push_offset_dist', 0.05)
        self.gripper_pull_start_z = rospy.get_param('~gripper_push_start_z', -0.29)

        self.max_restart_limit = rospy.get_param('~max_restart_limit', 0)

        self.min_new_pose_dist = rospy.get_param('~min_new_pose_dist', 0.2)
        self.min_workspace_x = rospy.get_param('~min_workspace_x', 0.475)
        self.max_workspace_x = rospy.get_param('~max_workspace_x', 0.75)
        self.max_workspace_y = rospy.get_param('~max_workspace_y', 0.4)
        self.min_workspace_y = -self.max_workspace_y
        self.min_goal_x = rospy.get_param('~min_workspace_x', 0.5)
        self.max_goal_x = rospy.get_param('~max_workspace_x', 0.7)
        self.max_goal_y = rospy.get_param('~max_workspace_y', 0.375)
        self.min_goal_y = -self.max_goal_y

        self.goal_y_base_delta = 0.01
        self.num_start_loc_clusters = 5
        self.start_loc_use_fixed_goal = rospy.get_param('start_loc_use_fixed_goal', False)
        self.servo_head_during_pushing = rospy.get_param('servo_head_during_pushing', False)
        self.learn_file_base = rospy.get_param('push_learn_file_base_path', '/u/thermans/data/new/aff_learn_out_')
        self.learning_dynamics = False
        self.compare_shape_for_dynamics = False
        self.check_model_performance = False

        # Setup service proxies
        if not _OFFLINE:
            # Proxy to setup spine and head
            self.raise_and_look_proxy = rospy.ServiceProxy('raise_and_look', RaiseAndLook)
            if use_singulation:
                self.gripper_push_proxy = rospy.ServiceProxy('gripper_push', GripperPush)
                self.gripper_pre_push_proxy = rospy.ServiceProxy('gripper_pre_push', GripperPush)
                self.gripper_post_push_proxy = rospy.ServiceProxy('gripper_post_push', GripperPush)
                self.gripper_pre_sweep_proxy = rospy.ServiceProxy('gripper_pre_sweep', GripperPush)
                self.gripper_sweep_proxy = rospy.ServiceProxy('gripper_sweep', GripperPush)
                self.gripper_post_sweep_proxy = rospy.ServiceProxy('gripper_post_sweep', GripperPush)
                self.overhead_pre_push_proxy = rospy.ServiceProxy('overhead_pre_push', GripperPush)
                self.overhead_push_proxy = rospy.ServiceProxy('overhead_push', GripperPush)
                self.overhead_post_push_proxy = rospy.ServiceProxy('overhead_post_push', GripperPush)
        if use_learning:
            # New visual feedback proxies
            self.overhead_feedback_push_proxy = rospy.ServiceProxy(
                'overhead_feedback_push', FeedbackPush)
            self.overhead_feedback_post_push_proxy = rospy.ServiceProxy(
                'overhead_feedback_post_push', FeedbackPush)
            self.gripper_feedback_push_proxy = rospy.ServiceProxy(
                'gripper_feedback_push', FeedbackPush)
            self.gripper_feedback_post_push_proxy = rospy.ServiceProxy(
                'gripper_feedback_post_push', FeedbackPush)
            self.gripper_feedback_sweep_proxy = rospy.ServiceProxy(
                'gripper_feedback_sweep', FeedbackPush)
            self.gripper_feedback_post_sweep_proxy = rospy.ServiceProxy(
                'gripper_feedback_post_sweep', FeedbackPush)
            self.overhead_feedback_pre_push_proxy = rospy.ServiceProxy('overhead_pre_push',
                                                                       FeedbackPush)
            self.gripper_feedback_pre_push_proxy = rospy.ServiceProxy('gripper_pre_push',
                                                                      FeedbackPush)
            self.gripper_feedback_pre_sweep_proxy = rospy.ServiceProxy('gripper_pre_sweep',
                                                                       FeedbackPush)
        self.table_proxy = rospy.ServiceProxy('get_table_location', LocateTable)
        self.learn_io = None

        if use_singulation:
            self.use_singulation = True
            self.init_singulation()
        else:
            self.use_singulation = False
        if use_learning:
            self.use_learning = True
            self.init_learning()
        else:
            self.use_learning = False
        rospy.on_shutdown(self.shutdown_hook)

    def init_singulation(self):
        # Singulation Push proxy
        self.singulation_push_vector_proxy = rospy.ServiceProxy(
            'get_singulation_push_vector', SingulationPush)

    def init_learning(self):
        self.learning_push_vector_proxy = rospy.ServiceProxy(
            'get_learning_push_vector', LearnPush)
        # Get table height and raise to that before anything else
        if not _OFFLINE:
            self.raise_and_look()
        # Initialize push pose
        initialized = False
        r = rospy.Rate(0.5)
        while not initialized:
            initialized = self.initialize_learning_push()
            r.sleep()
        rospy.loginfo('Done initializing learning')

    def init_learning_io(self):
        # Start loc learning stuff
        self.push_count = 0
        self.base_trial_id = str(rospy.get_time())

        if _USE_LEARN_IO:
            self.learn_io = PushLearningIO()
            learn_file_name = self.learn_file_base+str(self.base_trial_id)+'.txt'
            self.learn_out_file_name = learn_file_name
            rospy.loginfo('Opening learn file: '+learn_file_name)
            self.learn_io.open_out_file(learn_file_name)

    def finish_learning(self):
        rospy.loginfo('Done with learning pushes and such.')
        if _USE_LEARN_IO and self.learn_io is not None:
            self.learn_io.close_out_file()

    def run_singulation(self, num_pushes=1, use_guided=True):
        # Get table height and raise to that before anything else
        if not _OFFLINE:
            self.raise_and_look()
        # Initialize push pose
        self.initialize_singulation_push_vector();

        # NOTE: Should exit before reaching num_pushes, this is just a backup
        for i in xrange(num_pushes):
            if _OFFLINE:
                code_in = raw_input("Press <Enter> to determine next singulation push: ")
                if quit_code(code_in):
                    break
            pose_res = self.request_singulation_push(use_guided)
            if pose_res is None:
                rospy.logwarn("pose_res is None. Exiting pushing");
                break
            if pose_res.no_push:
                rospy.loginfo("No push. Exiting pushing.");
                break
            rospy.loginfo('Performing push #' + str(i+1))
            # Decide push based on the orientation returned
            rospy.loginfo('Push start_point: (' + str(pose_res.start_point.x) +
                          ', ' + str(pose_res.start_point.y) +
                          ', ' + str(pose_res.start_point.z) + ')')
            rospy.loginfo('Push angle: ' + str(pose_res.push_angle))
            rospy.loginfo('Push dist: ' + str(pose_res.push_dist))

            behavior_primitive = self.choose_singulation_primitive(pose_res)
            # behavior_primitive = GRIPPER_PUSH
            # behavior_primitive = OVERHEAD_PUSH
            # behavior_primitive = GRIPPER_SWEEP

            # Choose arm
            which_arm = self.choose_singulation_arm(pose_res)
            push_dist = pose_res.push_dist
            push_dist = max(min(push_dist, self.max_push_dist),
                            self.min_push_dist)

            if _OFFLINE:
                continue
            if behavior_primitive == GRIPPER_PUSH:
                self.gripper_push_object(push_dist, which_arm, pose_res, True)
            if behavior_primitive == GRIPPER_SWEEP:
                self.sweep_object(push_dist, which_arm, pose_res, True)
            if behavior_primitive == OVERHEAD_PUSH:
                self.overhead_push_object(push_dist, which_arm, pose_res, True)
            rospy.loginfo('Done performing push behavior.\n')

        if not (pose_res is None):
            rospy.loginfo('Singulated objects: ' + str(pose_res.singulated))
            rospy.loginfo('Final estimate of ' + str(pose_res.num_objects) +
                          ' objects')

    def run_start_loc_learning(self, object_id, num_pushes_per_sample, num_sample_locs, start_loc_param_path=''):
        for controller in CONTROLLERS:
            for behavior_primitive in BEHAVIOR_PRIMITIVES[controller]:
                for proxy in PERCEPTUAL_PROXIES[controller]:
                    for arm in ROBOT_ARMS:
                        res = self.explore_push_start_locs(num_pushes_per_sample, num_sample_locs, behavior_primitive,
                                                           controller, proxy, object_id, arm, start_loc_param_path)
                        if res == 'quit':
                            rospy.loginfo('Quiting on user request')
                            return False
        return True

    def run_push_exploration(self, object_id):
        for controller in CONTROLLERS:
            for behavior_primitive in BEHAVIOR_PRIMITIVES[controller]:
                for proxy in PERCEPTUAL_PROXIES[controller]:
                    for arm in ROBOT_ARMS:
                        precondition_method = PRECONDITION_METHODS[behavior_primitive]
                        res = self.explore_push(behavior_primitive, controller, proxy, object_id,
                                                precondition_method, arm)
                        if res == 'quit':
                            rospy.loginfo('Quiting on user request')
                            return False
                        elif res is None:
                            return False
        return True

    def explore_push(self, behavior_primitive, controller_name, proxy_name, object_id,
                     precondition_method=CENTROID_PUSH_PRECONDITION, input_arm=None):
        if input_arm is not None:
            rospy.loginfo('Exploring push behavior: (' + behavior_primitive + ', '
                          + controller_name + ', ' + proxy_name + ', ' + input_arm + ')')
        else:
            rospy.loginfo('Exploring push triple: (' + behavior_primitive + ', '
                          + controller_name + ', ' + proxy_name + ')')
        timeout = 2
        rospy.loginfo("Enter something to pause before pushing: ")
        rlist, _, _ = select([sys.stdin], [], [], timeout)
        if rlist:
            s = sys.stdin.readline()
            code_in = raw_input('Move object and press <Enter> to continue: ')
            if quit_code(code_in):
                return 'quit'
        else:
            rospy.loginfo("No input. Moving on...")

        # NOTE: Get initial object pose here to make sure goal pose is far enough away
        if self.servo_head_during_pushing:
            self.raise_and_look(point_head_only=True)

        init_pose = self.get_feedback_push_initial_obj_pose()
        while not _OFFLINE and self.out_of_workspace(init_pose):
            rospy.loginfo('Object out of workspace at pose: (' + str(init_pose.x) + ', ' +
                          str(init_pose.y) + ')')
            code_in = raw_input('Move object inside workspace and press <Enter> to continue: ')
            if quit_code(code_in):
                return 'quit'
            init_pose = self.get_feedback_push_initial_obj_pose()
        goal_pose = self.generate_random_table_pose(init_pose)

        restart_count = 0
        done_with_push = False
        while not done_with_push:
            start_time = time.time()
            push_vec_res = self.get_feedback_push_start_pose(goal_pose, controller_name,
                                                             proxy_name, behavior_primitive,
                                                             learn_start_loc=False)
            if push_vec_res is None:
                rospy.logerr('Failed to get learning push response!')
                return None
            elif push_vec_res == 'quit':
                return push_vec_res

            if self.compare_shape_for_dynamics and len(push_vec_res.dynamics_model_names) > 0:
                rospy.loginfo('Best matching models: ' + str(push_vec_res.dynamics_model_names))
                if controller_name.startswith(MPC_CONTROLLER_PREFIX):
                    controller_name = MPC_CONTROLLER_PREFIX + push_vec_res.dynamics_model_names[0]
                elif controller_name.startswith(OPEN_LOOP_SQP_PREFIX):
                    controller_name = OPEN_LOOP_SQP_PREFIX + push_vec_res.dynamics_model_names[0]
                else:
                    rospy.logwarn('Not using a model based controller, can\'t use model.')
                rospy.loginfo('Now using controller: ' + controller_name)

            if input_arm is None:
                which_arm = self.choose_arm(push_vec_res.push, controller_name)
            else:
                which_arm = input_arm

            if _USE_LEARN_IO:
                shape_descriptor = push_vec_res.shape_descriptor[:]
                self.learn_io.write_pre_push_line(push_vec_res.centroid, push_vec_res.theta,
                                                  goal_pose, push_vec_res.push.start_point,
                                                  behavior_primitive, controller_name,
                                                  proxy_name, which_arm, object_id, precondition_method,
                                                  shape_descriptor = shape_descriptor)

            res, push_res = self.perform_push(which_arm, behavior_primitive,
                                              push_vec_res, goal_pose,
                                              controller_name, proxy_name)
            push_time = time.time() - start_time
            if self.check_model_performance:
                rospy.loginfo('Most predictive model: ' + push_res.best_model + ' with score ' +
                              str(push_res.best_model_score))
                rospy.loginfo('Used model scored: ' + str(push_res.used_model_score))
            if push_res.failed_pre_position:
                rospy.loginfo('Writing trial bad line because of failed hand placement')
                if _USE_LEARN_IO:
                    self.learn_io.write_bad_trial_line()
            else:
                self.analyze_push(behavior_primitive, controller_name, proxy_name, which_arm, push_time,
                                  push_vec_res, goal_pose, object_id, precondition_method)

            if res == 'quit':
                return res
            elif res == 'aborted':
                if self.servo_head_during_pushing:
                    self.raise_and_look(point_head_only=True)
                restart_count += 1
                if restart_count <= self.max_restart_limit:
                    rospy.loginfo('Continuing after push was aborted')
                    continue
                else:
                    done_with_push = True
                    rospy.loginfo('Stopping push attempt because of too many restarts\n')
            elif push_res.failed_pre_position:
                rospy.loginfo('Stopping push because of failed pre positioning\n')
                done_with_push = True
            else:
                rospy.loginfo('Stopping push attempt because reached goal\n')
                done_with_push = True
        return res

    def explore_push_start_locs(self, num_pushes_per_sample, num_sample_locs, behavior_primitive, controller_name,
                                proxy_name, object_id, which_arm, start_loc_param_path='',
                                precondition_method=CENTROID_PUSH_PRECONDITION):
        rospy.loginfo('Exploring push start locs for triple: (' + behavior_primitive + ', ' +
                      controller_name + ', ' + proxy_name + ')')
        timeout = 2
        rospy.loginfo("Enter something to pause before pushing: ")
        rlist, _, _ = select([sys.stdin], [], [], timeout)
        if rlist:
            s = sys.stdin.readline()
            code_in = raw_input('Move object and press <Enter> to continue: ')
            if quit_code(code_in):
                return 'quit'
        else:
            rospy.loginfo("No input. Moving on...")

        if self.servo_head_during_pushing:
            self.raise_and_look(point_head_only=True)

        start_loc_trials = 0
        self.start_loc_goal_y_delta = 0
        N_PUSH = num_sample_locs
        # HACK: Currently quit after half the locations, assuming pushing on symmetric objects
        # N_PUSH = num_sample_locs/2+1
        i = 0
        while i < N_PUSH:
            rospy.loginfo('Performing push at new pose: ' + str(i))
            i += 1
            # Doesn't matter what the goal_pose is, the start pose server picks it for us
            goal_pose = self.generate_random_table_pose()
            j = 0
            position_worked = True
            while j  < num_pushes_per_sample:
                rospy.loginfo('Performing push iteration: '+str(start_loc_trials))
                rospy.loginfo('Performing sample push ' + str(j) + ' for pose: ' + str(i-1))
                rospy.loginfo('Sending param_path: ' + start_loc_param_path)
                # NOTE: Get initial object pose here to make sure goal pose is far enough away
                init_pose = self.get_feedback_push_initial_obj_pose()
                if self.start_loc_use_fixed_goal: # and position_worked:
                    reset = False
                    while not reset:
                        code_in = raw_input('Move object to initial test pose and press <Enter> to continue: ')
                        if quit_code(code_in):
                            return 'quit'
                        reset = True
                        init_pose = self.get_feedback_push_initial_obj_pose()
                elif not _OFFLINE:
                    while self.out_of_workspace(init_pose):
                        rospy.loginfo('Object out of workspace at pose: (' + str(init_pose.x) + ', ' +
                                      str(init_pose.y) + ')')
                        code_in = raw_input('Move object inside workspace and press <Enter> to continue: ')
                        if quit_code(code_in):
                            return 'quit'
                        init_pose = self.get_feedback_push_initial_obj_pose()

                trial_id = str(object_id) +'_'+ str(self.base_trial_id) + '_' + str(self.push_count)
                self.push_count += 1
                start_time = time.time()
                push_vec_res = self.get_feedback_push_start_pose(goal_pose, controller_name, proxy_name,
                                                                 behavior_primitive, learn_start_loc=True,
                                                                 new_object=(not start_loc_trials),
                                                                 num_clusters=self.num_start_loc_clusters,
                                                                 trial_id=trial_id,
                                                                 num_sample_locs=num_sample_locs,
                                                                 num_pushes_per_sample=num_pushes_per_sample,
                                                                 start_loc_param_path=start_loc_param_path,
                                                                 position_worked=position_worked)
                if push_vec_res is None:
                    rospy.logwarn('Push vector Response is none, exiting')
                    return None
                elif push_vec_res == 'quit':
                    return push_vec_res

                # Pick arm if we are using learned stuff
                if len(start_loc_param_path) > 0:
                    which_arm = self.choose_arm(push_vec_res.push, controller_name)

                goal_pose = push_vec_res.goal_pose
                shape_descriptor = push_vec_res.shape_descriptor[:]

                if _USE_LEARN_IO:
                    # TODO: Check that the predicted_score is sent correctly
                    rospy.loginfo('Predicted score: ' + str(push_vec_res.predicted_score))
                    self.learn_io.write_pre_push_line(push_vec_res.centroid, push_vec_res.theta,
                                                      goal_pose, push_vec_res.push.start_point, behavior_primitive,
                                                      controller_name, proxy_name, which_arm, trial_id,
                                                      precondition_method, push_vec_res.predicted_score,
                                                      shape_descriptor)

                res, push_res = self.perform_push(which_arm, behavior_primitive, push_vec_res, goal_pose,
                                                  controller_name, proxy_name)
                push_time = time.time() - start_time

                if _USE_LEARN_IO:
                    timeout = 2
                    rospy.loginfo("Enter something to not save the previous push trial: ")
                    rlist, _, _ = select([sys.stdin], [], [], timeout)
                    if rlist:
                        self.learn_io.write_bad_trial_line(user_ordered=True)
                        s = sys.stdin.readline()
                        rospy.loginfo('Not saving previous trial.')
                        if s.lower().startswith('q'):
                            return 'quit'
                        position_worked = True
                    elif push_res.failed_pre_position:
                        self.learn_io.write_bad_trial_line()
                        rospy.loginfo('Not saving previous trial because of failed hand placement')
                        position_worked = False
                        # TOOD: Get the next choice, to avoid infinitely trying this one
                        code_in = raw_input('Move object to new pose and press <Enter> to continue: ')
                        if quit_code(code_in):
                            return 'quit'
                    else:
                        rospy.loginfo("No input. Saving trial data")
                        self.analyze_push(behavior_primitive, controller_name, proxy_name, which_arm, push_time,
                                          push_vec_res, goal_pose, trial_id, precondition_method)
                        start_loc_trials += 1
                        j += 1
                        position_worked = True
                else:
                    j += 1
                    position_worked = True

                if res == 'quit':
                    return res
                elif res == 'aborted' or res == 'done':
                    if self.servo_head_during_pushing:
                        self.raise_and_look(point_head_only=True)
        rospy.loginfo('Done performing push loc exploration!')
        return res

    def get_feedback_push_initial_obj_pose(self):
        get_push = True
        while get_push:
            goal_pose = Pose2D()
            controller_name = CENTROID_CONTROLLER
            proxy_name = ELLIPSE_PROXY
            behavior_primitive = OVERHEAD_PUSH
            push_vec_res = self.request_feedback_push_start_pose(goal_pose, controller_name,
                                                                 proxy_name, behavior_primitive,
                                                                 get_pose_only=True)

            if push_vec_res is None:
                return None
            if push_vec_res.no_objects:
                code_in = raw_input('No objects found. Place object and press <Enter>: ')
                if quit_code(code_in):
                    return 'quit'
            else:
                return push_vec_res.centroid


    def get_feedback_push_start_pose(self, goal_pose, controller_name, proxy_name,
                                     behavior_primitive,
                                     learn_start_loc=False, new_object=False, num_clusters=1,
                                     trial_id='',num_sample_locs=1, num_pushes_per_sample=1,
                                     start_loc_param_path='', position_worked=True):
        get_push = True
        while get_push:
            push_vec_res = self.request_feedback_push_start_pose(goal_pose, controller_name,
                                                                 proxy_name, behavior_primitive,
                                                                 learn_start_loc=learn_start_loc,
                                                                 new_object=new_object,
                                                                 num_clusters=num_clusters,
                                                                 trial_id=trial_id,
                                                                 num_sample_locs=num_sample_locs,
                                                                 num_pushes_per_sample=num_pushes_per_sample,
                                                                 start_loc_param_path=start_loc_param_path,
                                                                 position_worked=position_worked)

            if push_vec_res is None:
                return None
            if push_vec_res.no_objects:
                code_in = raw_input('No objects found. Place object and press <Enter>: ')
                if quit_code(code_in):
                    return 'quit'
            else:
                return push_vec_res

    def choose_arm(self, push_vec, controller_name):
        if controller_name == ROTATE_TO_HEADING:
            if (push_vec.start_point.y < 0):
                which_arm = 'r'
            else:
                which_arm = 'l'
            return which_arm
        elif controller_name == DIRECT_GOAL_CONTROLLER:
            if push_vec.push_angle > 0:
                which_arm = 'r'
            else:
                which_arm = 'l'
        elif (fabs(push_vec.start_point.y) > self.use_same_side_y_thresh or
              push_vec.start_point.x > self.use_same_side_x_thresh):
            if (push_vec.start_point.y < 0):
                which_arm = 'r'
            else:
                which_arm = 'l'
        elif push_vec.push_angle > 0:
            which_arm = 'r'
        else:
            which_arm = 'l'

        return which_arm

    def choose_singulation_primitive(self, pose_res):
        # Choose push behavior
        if fabs(pose_res.push_angle) > self.use_pull_angle_thresh:
            behavior_primitive = OVERHEAD_PUSH
        elif pose_res.start_point.x < self.use_overhead_x_thresh:
            behavior_primitive = OVERHEAD_PUSH
        elif fabs(pose_res.push_angle) > self.use_sweep_angle_thresh:
            behavior_primitive = GRIPPER_SWEEP
        else:
            behavior_primitive = GRIPPER_PUSH
        return behavior_primitive

    def choose_singulation_arm(self, pose_res):
        if (fabs(pose_res.start_point.y) > self.use_same_side_y_thresh or
            pose_res.start_point.x > self.use_same_side_x_thresh):
            if (pose_res.start_point.y < 0):
                which_arm = 'r'
                rospy.loginfo('Setting arm to right because of limits')
            else:
                which_arm = 'l'
                rospy.loginfo('Setting arm to left because of limits')
        elif pose_res.push_angle > 0:
            which_arm = 'r'
            rospy.loginfo('Setting arm to right because of angle')
        else:
            which_arm = 'l'
            rospy.loginfo('Setting arm to left because of angle')
        return which_arm

    def perform_push(self, which_arm, behavior_primitive, push_vector_res, goal_pose,
                     controller_name, proxy_name, high_init = True):
        push_angle = push_vector_res.push.push_angle
        # NOTE: Use commanded push distance not visually decided minimal distance
        if push_vector_res is None:
            rospy.logwarn("push_vector_res is None. Exiting pushing");
            return ('quit', None)
        if push_vector_res.no_push:
            rospy.loginfo("No push. Exiting pushing.");
            return ('quit', None)
        # Decide push based on the orientation returned
        rospy.loginfo('Push start_point: (' +
                      str(push_vector_res.push.start_point.x) + ', ' +
                      str(push_vector_res.push.start_point.y) + ', ' +
                      str(push_vector_res.push.start_point.z) + ')')
        rospy.loginfo('Push angle: ' + str(push_angle))

        if behavior_primitive == OVERHEAD_PUSH or behavior_primitive == OPEN_OVERHEAD_PUSH:
            result = self.overhead_feedback_push_object(which_arm,
                                                        push_vector_res, goal_pose,
                                                        controller_name, proxy_name, behavior_primitive)
        elif behavior_primitive == GRIPPER_SWEEP:
            result = self.feedback_sweep_object(which_arm, push_vector_res,
                                                goal_pose, controller_name, proxy_name, behavior_primitive)
        elif (behavior_primitive == GRIPPER_PUSH or behavior_primitive == PINCHER_PUSH or
              behavior_primitive == GRIPPER_PULL):
            result = self.gripper_feedback_push_object(which_arm,
                                                       push_vector_res, goal_pose,
                                                       controller_name, proxy_name, behavior_primitive)
        else:
            rospy.logwarn('Unknown behavior_primitive: ' + str(behavior_primitive))
            result = None

        # NOTE: If the call aborted, recall with the same settings
        if result.action_aborted:
            rospy.logwarn('Push was aborted. Calling push behavior again.')
            return ('aborted', result)

        rospy.loginfo('Done performing push behavior.')
        # if _OFFLINE:
        #     code_in = raw_input("Press <Enter> to try another push: ")
        #     if quit_code(code_in):
        #         return ('quit', result)
        return ('done', result)

    def analyze_push(self, behavior_primitive, controller_name, proxy_name,
                     which_arm, push_time, push_vector_res, goal_pose, object_id,
                     precondition_method=CENTROID_PUSH_PRECONDITION):
        push_angle = push_vector_res.push.push_angle
        analysis_res = self.request_learning_analysis(goal_pose)
        rospy.loginfo('Done getting analysis response.')
        rospy.loginfo('Primitive: ' + str(behavior_primitive))
        rospy.loginfo('Controller: ' + str(controller_name))
        rospy.loginfo('Proxy: ' + str(proxy_name))
        rospy.loginfo('Arm: ' + str(which_arm))
        rospy.loginfo('Push time: ' + str(push_time) + 's')
        rospy.loginfo('Init (X,Y,Theta): (' + str(push_vector_res.centroid.x) +
                      ', ' + str(push_vector_res.centroid.y) + ', ' +
                      str(push_vector_res.theta) +')')
        rospy.loginfo('Final (X,Y,Theta): (' + str(analysis_res.centroid.x) + ', ' +
                       str(analysis_res.centroid.y) + ', ' + str(analysis_res.theta)+ ')')
        rospy.loginfo('Desired (X,Y,Theta): (' + str(goal_pose.x) + ', ' +
                       str(goal_pose.y) + ', ' + str(goal_pose.theta) + ')')
        rospy.loginfo('Error : (' + str(fabs(goal_pose.x-analysis_res.centroid.x)) +
                      ', ' + str(fabs(goal_pose.y-analysis_res.centroid.y)) + ', ' +
                      str(fabs(goal_pose.theta-analysis_res.theta)) + ') : ' +
                      str(hypot(goal_pose.x-analysis_res.centroid.x,goal_pose.y-analysis_res.centroid.y)) + '\n')
        if _USE_LEARN_IO:
            self.learn_io.write_line(
                push_vector_res.centroid, push_vector_res.theta,
                analysis_res.centroid, analysis_res.theta,
                goal_pose, push_vector_res.push.start_point, behavior_primitive, controller_name, proxy_name,
                which_arm, push_time, object_id, precondition_method)

    def request_singulation_push(self, use_guided=True):
        push_vector_req = SingulationPushRequest()
        push_vector_req.use_guided = use_guided
        push_vector_req.initialize = False
        push_vector_req.no_push_calc = False
        rospy.loginfo("Calling singulation push vector service")
        try:
            push_vector_res = self.singulation_push_vector_proxy(push_vector_req)
            return push_vector_res
        except rospy.ServiceException, e:
            rospy.logwarn("Service did not process request: %s"%str(e))
            return None

    def request_feedback_push_start_pose(self, goal_pose, controller_name, proxy_name,
                                         behavior_primitive,
                                         get_pose_only=False, learn_start_loc=False,
                                         new_object=False, num_clusters=1, trial_id='',
                                         num_sample_locs=1, num_pushes_per_sample=1,start_loc_param_path='',
                                         position_worked=True):
        push_vector_req = LearnPushRequest()
        push_vector_req.initialize = False
        push_vector_req.analyze_previous = False
        push_vector_req.goal_pose = goal_pose
        push_vector_req.controller_name = controller_name
        push_vector_req.proxy_name = proxy_name
        push_vector_req.behavior_primitive = behavior_primitive
        push_vector_req.get_pose_only = get_pose_only
        push_vector_req.learn_start_loc = learn_start_loc
        push_vector_req.new_object = new_object
        push_vector_req.trial_id = trial_id
        push_vector_req.num_start_loc_clusters = num_clusters
        push_vector_req.num_start_loc_sample_locs = num_sample_locs
        push_vector_req.num_start_loc_pushes_per_sample = num_pushes_per_sample
        push_vector_req.start_loc_param_path=start_loc_param_path
        push_vector_req.previous_position_worked = position_worked
        push_vector_req.dynamics_learning = self.learning_dynamics
        push_vector_req.compare_shape_for_dynamics = self.compare_shape_for_dynamics
        try:
            rospy.loginfo("Calling feedback push start service")
            push_vector_res = self.learning_push_vector_proxy(push_vector_req)
            return push_vector_res
        except rospy.ServiceException, e:
            rospy.logwarn("Service did not process request: %s"%str(e))
            return None

    def request_learning_analysis(self, goal_pose):
        push_vector_req = LearnPushRequest()
        push_vector_req.initialize = False
        push_vector_req.analyze_previous = True
        push_vector_req.goal_pose = goal_pose
        rospy.loginfo("Calling learning push vector service")
        try:
            push_vector_res = self.learning_push_vector_proxy(push_vector_req)
            return push_vector_res
        except rospy.ServiceException, e:
            rospy.logwarn("Service did not process request: %s"%str(e))
            return None

    def initialize_singulation_push_vector(self):
        push_vector_req = SingulationPushRequest()
        push_vector_req.initialize = True
        push_vector_req.use_guided = True
        push_vector_req.no_push_calc = False
        rospy.loginfo('Initializing singulation push vector service.')
        self.singulation_push_vector_proxy(push_vector_req)

    def initialize_learning_push(self):
        push_vector_req = LearnPushRequest()
        push_vector_req.initialize = True
        push_vector_req.analyze_previous = False
        rospy.loginfo('Initializing learning push vector service.')
        try:
            self.learning_push_vector_proxy(push_vector_req)
        except rospy.ServiceException, e:
            rospy.logwarn("Service did not process request: %s"%str(e))
            return False
        return True

    def raise_and_look(self, request_table=True, init_arms=False, point_head_only=False):
        if request_table:
            table_req = LocateTableRequest()
            table_req.recalculate = True
        raise_req = RaiseAndLookRequest()
        raise_req.point_head_only = True
        raise_req.camera_frame = 'head_mount_kinect_rgb_link'
        # First make sure the head is looking the correct way before estimating
        # the table centroid
        # Also make sure the arms are out of the way
        raise_req.init_arms = True
        rospy.loginfo("Moving head")
        raise_res = self.raise_and_look_proxy(raise_req)
        if point_head_only:
            return
        if request_table:
            raise_req.have_table_centroid = True
            try:
                rospy.loginfo("Getting table pose")
                table_res = self.table_proxy(table_req);
            except rospy.ServiceException, e:
                rospy.logwarn("Service did not process request: %s"%str(e))
                return
            if not table_res.found_table:
                return
            raise_req.table_centroid = table_res.table_centroid
        else:
            raise_req.have_table_centroid = False

        rospy.loginfo("Raising spine");
        raise_req.point_head_only = False
        raise_req.init_arms = init_arms
        raise_res = self.raise_and_look_proxy(raise_req)

    def overhead_feedback_push_object(self, which_arm, learn_push_res, goal_pose, controller_name,
                                      proxy_name, behavior_primitive, high_init=True, open_gripper=False):
        # Convert pose response to correct push request format
        push_req = FeedbackPushRequest()
        push_vector = learn_push_res.push
        push_req.obj_start_pose.x = learn_push_res.centroid.x
        push_req.obj_start_pose.y = learn_push_res.centroid.y
        push_req.obj_start_pose.theta = learn_push_res.theta
        push_req.start_point.header = push_vector.header
        push_req.start_point.point = push_vector.start_point
        push_req.open_gripper = open_gripper
        push_req.goal_pose = goal_pose
        push_req.check_model_performance = self.check_model_performance

        if behavior_primitive == OPEN_OVERHEAD_PUSH:
            push_req.open_gripper = True
        if _USE_LEARN_IO:
            push_req.learn_out_file_name = self.learn_out_file_name

        # Use the sent wrist yaw
        wrist_yaw = push_vector.push_angle
        push_req.wrist_yaw = wrist_yaw

        # Offset pose to not hit the object immediately
        push_req.start_point.point.x += -self.gripper_offset_dist*cos(wrist_yaw)
        push_req.start_point.point.y += -self.gripper_offset_dist*sin(wrist_yaw)
        push_req.start_point.point.z = self.overhead_start_z
        push_req.left_arm = (which_arm == 'l')
        push_req.right_arm = not push_req.left_arm
        push_req.high_arm_init = high_init
        push_req.controller_name = controller_name
        push_req.proxy_name = proxy_name
        push_req.behavior_primitive = behavior_primitive

        rospy.loginfo('Overhead push augmented start_point: (' +
                      str(push_req.start_point.point.x) + ', ' +
                      str(push_req.start_point.point.y) + ', ' +
                      str(push_req.start_point.point.z) + ')')

        if not _OFFLINE:
            rospy.loginfo("Calling overhead feedback pre push service")
            pre_push_res = self.overhead_feedback_pre_push_proxy(push_req)
            rospy.loginfo("Calling overhead feedback push service")
        else:
            pre_push_res = FeedbackPushResponse()
            pre_push_res.failed_pre_position = False

        if _TEST_START_POSE:
            raw_input('waiting for input to recall arm: ')
            push_res = FeedbackPushResponse()
        elif _OFFLINE and not _OFFLINE_WITH_FEEDBACK_CONTROL:
            push_res = FeedbackPushResponse()
        elif pre_push_res.failed_pre_position:
            rospy.logwarn('Failed to properly position in pre-push, aborting push')
            push_res = pre_push_res
        else:
            push_res = self.overhead_feedback_push_proxy(push_req)

        if not _OFFLINE:
            rospy.loginfo("Calling overhead feedback post push service")
            post_push_res = self.overhead_feedback_post_push_proxy(push_req)
        return push_res

    def gripper_feedback_push_object(self, which_arm, learn_push_res, goal_pose, controller_name,
                                     proxy_name, behavior_primitive, high_init=True, open_gripper=False):
        # Convert pose response to correct push request format
        push_req = FeedbackPushRequest()
        push_vector = learn_push_res.push
        push_req.obj_start_pose.x = learn_push_res.centroid.x
        push_req.obj_start_pose.y = learn_push_res.centroid.y
        push_req.obj_start_pose.theta = learn_push_res.theta
        push_req.start_point.header = push_vector.header
        push_req.start_point.point = push_vector.start_point
        push_req.open_gripper = open_gripper
        push_req.goal_pose = goal_pose
        push_req.check_model_performance = self.check_model_performance
        if _USE_LEARN_IO:
            push_req.learn_out_file_name = self.learn_out_file_name

        # Use the sent wrist yaw
        wrist_yaw = push_vector.push_angle
        push_req.wrist_yaw = wrist_yaw
        # Offset pose to not hit the object immediately
        if behavior_primitive == GRIPPER_PULL:
            offset_dist = self.gripper_pull_offset_dist
            start_z = self.gripper_pull_start_z
        elif behavior_primitive == PINCHER_PUSH:
            offset_dist = self.pincher_offset_dist
            start_z = self.pincher_start_z
            push_req.open_gripper = True
        else:
            offset_dist = self.gripper_offset_dist
            start_z = self.gripper_start_z

        rospy.loginfo('Gripper push pre-augmented start_point: (' +
                      str(push_req.start_point.point.x) + ', ' +
                      str(push_req.start_point.point.y) + ', ' +
                      str(push_req.start_point.point.z) + ')')

        push_req.start_point.point.x += -offset_dist*cos(wrist_yaw)
        push_req.start_point.point.y += -offset_dist*sin(wrist_yaw)
        push_req.start_point.point.z = start_z
        push_req.left_arm = (which_arm == 'l')
        push_req.right_arm = not push_req.left_arm
        push_req.high_arm_init = high_init
        push_req.controller_name = controller_name
        push_req.proxy_name = proxy_name
        push_req.behavior_primitive = behavior_primitive

        rospy.loginfo('Gripper push augmented start_point: (' +
                      str(push_req.start_point.point.x) + ', ' +
                      str(push_req.start_point.point.y) + ', ' +
                      str(push_req.start_point.point.z) + ')')

        if not _OFFLINE:
            rospy.loginfo("Calling gripper feedback pre push service")
            pre_push_res = self.gripper_feedback_pre_push_proxy(push_req)
        else:
            pre_push_res = FeedbackPushResponse()
            pre_push_res.failed_pre_position = False

        if _TEST_START_POSE:
            raw_input('waiting for input to recall arm: ')
            push_res = FeedbackPushResponse()
        elif _OFFLINE and not _OFFLINE_WITH_FEEDBACK_CONTROL:
            push_res = FeedbackPushResponse()
        elif pre_push_res.failed_pre_position:
            rospy.logwarn('Failed to properly position in pre-push, aborting push')
            push_res = pre_push_res
        else:
            rospy.loginfo("Calling gripper feedback push service")
            push_res = self.gripper_feedback_push_proxy(push_req)

        if not _OFFLINE:
            rospy.loginfo("Calling gripper feedback post push service")
            post_push_res = self.gripper_feedback_post_push_proxy(push_req)
        return push_res

    def feedback_sweep_object(self, which_arm, learn_push_res, goal_pose, controller_name,
                              proxy_name, behavior_primitive, high_init=True, open_gripper=False):
        # Convert pose response to correct push request format
        push_req = FeedbackPushRequest()
        push_vector = learn_push_res.push
        push_req.obj_start_pose.x = learn_push_res.centroid.x
        push_req.obj_start_pose.y = learn_push_res.centroid.y
        push_req.obj_start_pose.theta = learn_push_res.theta
        push_req.start_point.header = push_vector.header
        push_req.start_point.point = push_vector.start_point
        push_req.open_gripper = open_gripper
        push_req.goal_pose = goal_pose
        push_req.check_model_performance = self.check_model_performance
        if _USE_LEARN_IO:
            push_req.learn_out_file_name = self.learn_out_file_name

        # if push_req.left_arm:
        if push_vector.push_angle > 0:
            y_offset_dir = -1
            wrist_yaw = push_vector.push_angle - pi/2.0
        else:
            y_offset_dir = +1
            wrist_yaw = push_vector.push_angle + pi/2.0
        if abs(push_vector.push_angle) > pi/2:
            x_offset_dir = +1
        else:
            x_offset_dir = -1

        # Set offset in x y, based on distance
        push_req.wrist_yaw = wrist_yaw
        face_x_offset = self.sweep_face_offset_dist*x_offset_dir*abs(sin(wrist_yaw))
        face_y_offset = y_offset_dir*self.sweep_face_offset_dist*cos(wrist_yaw)
        wrist_x_offset = self.sweep_wrist_offset_dist*cos(wrist_yaw)
        wrist_y_offset = self.sweep_wrist_offset_dist*sin(wrist_yaw)
        rospy.loginfo('wrist_yaw: ' + str(wrist_yaw))
        rospy.loginfo('Face offset: (' + str(face_x_offset) + ', ' + str(face_y_offset) +')')
        rospy.loginfo('Wrist offset: (' + str(wrist_x_offset) + ', ' + str(wrist_y_offset) +')')

        push_req.start_point.point.x += face_x_offset + wrist_x_offset
        push_req.start_point.point.y += face_y_offset + wrist_y_offset
        push_req.start_point.point.z = self.sweep_start_z
        push_req.left_arm = (which_arm == 'l')
        push_req.right_arm = not push_req.left_arm
        push_req.high_arm_init = high_init
        push_req.controller_name = controller_name
        push_req.proxy_name = proxy_name
        push_req.behavior_primitive = behavior_primitive

        rospy.loginfo('Gripper sweep augmented start_point: (' +
                      str(push_req.start_point.point.x) + ', ' +
                      str(push_req.start_point.point.y) + ', ' +
                      str(push_req.start_point.point.z) + ')')
        if not _OFFLINE:
            rospy.loginfo("Calling feedback pre sweep service")
            pre_push_res = self.gripper_feedback_pre_sweep_proxy(push_req)
            rospy.loginfo("Calling feedback sweep service")
        else:
            pre_push_res = FeedbackPushResponse()
            pre_push_res.failed_pre_position = False

        if _TEST_START_POSE:
            raw_input('waiting for input to recall arm: ')
            push_res = FeedbackPushResponse()
        elif _OFFLINE and not _OFFLINE_WITH_FEEDBACK_CONTROL:
            push_res = FeedbackPushResponse()
        elif pre_push_res.failed_pre_position:
            rospy.logwarn('Failed to properly position in pre-push, aborting push')
            push_res = pre_push_res
        else:
            push_res = self.gripper_feedback_sweep_proxy(push_req)

        if not _OFFLINE:
            rospy.loginfo("Calling feedback post sweep service")
            post_push_res = self.gripper_feedback_post_sweep_proxy(push_req)

        return push_res

    def gripper_push_object(self, push_dist, which_arm, pose_res, high_init):
        # Convert pose response to correct push request format
        push_req = GripperPushRequest()
        push_req.start_point.header = pose_res.header
        push_req.start_point.point = pose_res.start_point
        push_req.arm_init = True
        push_req.arm_reset = True
        push_req.high_arm_init = True

        # Use the sent wrist yaw
        wrist_yaw = pose_res.push_angle
        push_req.wrist_yaw = wrist_yaw
        push_req.desired_push_dist = push_dist + abs(self.gripper_x_offset)

        # Offset pose to not hit the object immediately
        push_req.start_point.point.x += self.gripper_x_offset*cos(wrist_yaw)
        push_req.start_point.point.y += self.gripper_x_offset*sin(wrist_yaw)
        push_req.start_point.point.z = self.gripper_start_z
        push_req.left_arm = (which_arm == 'l')
        push_req.right_arm = not push_req.left_arm

        rospy.loginfo("Calling gripper pre push service")
        pre_push_res = self.gripper_pre_push_proxy(push_req)
        rospy.loginfo("Calling gripper push service")
        push_res = self.gripper_push_proxy(push_req)
        rospy.loginfo("Calling gripper post push service")
        post_push_res = self.gripper_post_push_proxy(push_req)

    def sweep_object(self, push_dist, which_arm, pose_res, high_init):
        # Convert pose response to correct push request format
        sweep_req = GripperPushRequest()
        sweep_req.left_arm = (which_arm == 'l')
        sweep_req.right_arm = not sweep_req.left_arm
        sweep_req.high_arm_init = True

        # Correctly set the wrist yaw
        if pose_res.push_angle > 0.0:
            y_offset_dir = -1
            wrist_yaw = pose_res.push_angle - pi/2
        else:
            wrist_yaw = pose_res.push_angle + pi/2
            y_offset_dir = +1
        if abs(pose_res.push_angle) > pi/2:
            x_offset_dir = +1
        else:
            x_offset_dir = -1

        # Set offset in x y, based on distance
        sweep_req.wrist_yaw = wrist_yaw
        face_x_offset = self.sweep_face_offset_dist*x_offset_dir*abs(sin(wrist_yaw))
        face_y_offset = y_offset_dir*self.sweep_face_offset_dist*cos(wrist_yaw)
        wrist_x_offset = self.sweep_wrist_offset_dist*cos(wrist_yaw)
        wrist_y_offset = self.sweep_wrist_offset_dist*sin(wrist_yaw)

        sweep_req.start_point.header = pose_res.header
        sweep_req.start_point.point = pose_res.start_point
        sweep_req.start_point.point.x += face_x_offset + wrist_x_offset
        sweep_req.start_point.point.y += face_y_offset + wrist_y_offset
        sweep_req.start_point.point.z = self.sweep_start_z
        sweep_req.arm_init = True
        sweep_req.arm_reset = True

        sweep_req.desired_push_dist = -y_offset_dir*(self.sweep_face_offset_dist + push_dist)


        rospy.loginfo("Calling gripper pre sweep service")
        pre_sweep_res = self.gripper_pre_sweep_proxy(sweep_req)
        rospy.loginfo("Calling gripper sweep service")
        sweep_res = self.gripper_sweep_proxy(sweep_req)
        rospy.loginfo("Calling gripper post sweep service")
        post_sweep_res = self.gripper_post_sweep_proxy(sweep_req)

    def overhead_push_object(self, push_dist, which_arm, pose_res, high_init):
        # Convert pose response to correct push request format
        push_req = GripperPushRequest()
        push_req.start_point.header = pose_res.header
        push_req.start_point.point = pose_res.start_point
        push_req.arm_init = True
        push_req.arm_reset = True
        push_req.high_arm_init = high_init

        # Correctly set the wrist yaw
        wrist_yaw = pose_res.push_angle
        push_req.wrist_yaw = wrist_yaw
        push_req.desired_push_dist = push_dist

        # Offset pose to not hit the object immediately
        push_req.start_point.point.x += self.overhead_offset_dist*cos(wrist_yaw)
        push_req.start_point.point.y += self.overhead_offset_dist*sin(wrist_yaw)
        push_req.start_point.point.z = self.overhead_start_z
        push_req.left_arm = (which_arm == 'l')
        push_req.right_arm = not push_req.left_arm

        rospy.loginfo("Calling pre overhead push service")
        pre_push_res = self.overhead_pre_push_proxy(push_req)
        rospy.loginfo("Calling overhead push service")
        push_res = self.overhead_push_proxy(push_req)
        rospy.loginfo("Calling post overhead push service")
        post_push_res = self.overhead_post_push_proxy(push_req)

    def out_of_workspace(self, init_pose):
        return (init_pose.x < self.min_workspace_x or init_pose.x > self.max_workspace_x or
                init_pose.y < self.min_workspace_y or init_pose.y > self.max_workspace_y)

    def generate_random_table_pose(self, init_pose=None):
        if _USE_FIXED_GOAL or self.start_loc_use_fixed_goal:
            goal_pose = Pose2D()
            goal_pose.x = 0.8
            goal_pose.y = 0.1
            goal_pose.theta = 0
            if self.start_loc_use_fixed_goal:
                goal_pose.y += self.start_loc_goal_y_delta
            return goal_pose
        min_x = self.min_goal_x
        max_x = self.max_goal_x
        min_y = self.min_goal_y
        max_y = self.max_goal_y
        max_theta = pi
        min_theta = -pi

        pose_not_found = True
        while pose_not_found:
            rand_pose = Pose2D()
            rand_pose.x = random.uniform(min_x, max_x)
            rand_pose.y = random.uniform(min_y, max_y)
            rand_pose.theta = random.uniform(min_theta, max_theta)
            pose_not_found = False
            # if (self.previous_rand_pose is not None and
            #     hypot(self.previous_rand_pose.x-rand_pose.x,
            #           self.previous_rand_pose.y-rand_pose.y) < self.min_new_pose_dist):
            #     pose_not_found = True
            if (init_pose is not None and
                hypot(init_pose.x-rand_pose.x, init_pose.y-rand_pose.y) < self.min_new_pose_dist):
                pose_not_found = True

        rospy.loginfo('Rand table pose is: (' + str(rand_pose.x) + ', ' + str(rand_pose.y) +
                      ', ' + str(rand_pose.theta) + ')')
        self.previous_rand_pose = rand_pose
        return rand_pose

    def shutdown_hook(self):
        rospy.loginfo('Cleaning up tabletop_executive_node on shutdown')
        if self.use_learning:
            self.finish_learning()

def get_object_id(previous_id=None):
    need_object_id = True
    while need_object_id:
        code_in = raw_input('Place object on table, enter id, and press <Enter>: ')
        if len(code_in) > 0:
            return code_in
        elif previous_id is not None:
            return previous_id
        else:
            rospy.logwarn("No object id given.")

if __name__ == '__main__':
    random.seed()
    learn_start_loc = True
    # Used for training data collection:
    # num_start_loc_sample_locs = 32
    # Used for testing data collection:
    num_start_loc_sample_locs = 1
    num_start_loc_pushes_per_sample = 10
    use_singulation = False
    use_learning = True
    learning_dynamics = True
    compare_shape_for_dynamics = False
    check_model_performance = False
    num_trials_per_object = 5
    use_guided = True
    max_pushes = 500

    node = TabletopExecutive(use_singulation, use_learning)
    node.learning_dynamics = learning_dynamics
    node.compare_shape_for_dynamics = compare_shape_for_dynamics
    node.check_model_performance = check_model_performance
    if use_singulation:
        node.run_singulation(max_pushes, use_guided)
    elif learn_start_loc:
        # Set the path to the learned parameter file here to use the learned SVM parameters
        hold_out_objects = ['soap_box', 'camcorder', 'toothpaste', 'food_box', 'large_brush', 'small_brush']
        running = True
        for hold_out_object in hold_out_objects:
            if not running:
                break
            # hold_out_object = 'soap_box'
            rospy.loginfo('Testing with hold out object ' + hold_out_object)
            # rospy.loginfo('Collecting training data for object ' + hold_out_object)
            start_loc_param_path = roslib.packages.get_pkg_dir('tabletop_pushing')+'/cfg/ichr_straight/push_svm_no_'+\
                hold_out_object+'.model'
            # start_loc_param_path = roslib.packages.get_pkg_dir('tabletop_pushing')+'/cfg/ichr_rotate/push_svm_no_'+\
            #     hold_out_object+'.model'
            # start_loc_param_path = 'rand'
            # start_loc_param_path = ''
            code_in = get_object_id()
            if quit_code(code_in):
                running = False
            else:
                node.init_learning_io()
                clean_exploration = node.run_start_loc_learning(code_in, num_start_loc_pushes_per_sample,
                                                                num_start_loc_sample_locs, start_loc_param_path)
                node.finish_learning()
    elif use_learning:
        running = True
        node.init_learning_io()
        previous_id = None
        while running:
            code_in = get_object_id(previous_id)
            if quit_code(code_in):
                running = False
            else:
                for i in xrange(num_trials_per_object):
                    node.start_loc_use_fixed_goal = False
                    previous_id = code_in
                    rospy.loginfo('Running push exploration round ' + str(i) + ' for object ' + previous_id)
                    clean_exploration = node.run_push_exploration(object_id=code_in)
                    if not clean_exploration:
                        rospy.loginfo('Not clean end to pushing stuff')
                        running = False
                        break
        node.finish_learning()
    else:
        print 'Nothing to do. Bye bye!'
