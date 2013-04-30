#! /usr/bin/python

import numpy as np

import roslib
roslib.load_manifest("ur_cart_move")

import rospy
from hrl_geom.pose_converter import PoseConv
from hrl_geom.transformations import rotation_from_matrix as mat_to_ang_axis_point
from hrl_geom.transformations import rotation_matrix as ang_axis_point_to_mat
from hrl_geom.transformations import euler_matrix
from min_jerk_sampler import sample_min_jerk_knots

# returns x_a - x_b such that 
def pose_diff(x_a, x_b):
    pos_delta = x_a[:3,3] - x_b[:3,3]
    rot_delta = x_b[:3,:3].T * x_a[:3,:3]
    rot_delta_homo = np.eye(4)
    rot_delta_homo[:3,:3] = rot_delta
    ang_delta, axis_delta, point_delta = mat_to_ang_axis_point(rot_delta_homo)
    return pos_delta, ang_delta, axis_delta, point_delta

##
# Returns x_cur, linearly interpolated from x_init to x_final 
# according to position and axis-angle interpolation.
# s = [0,1]
def pose_interp(x_init, x_final, s):
    x_cur = x_init.copy()
    pos_delta, ang_delta, axis_delta, point_delta = pose_diff(x_final, x_init)
    x_cur[:3,3] += s * pos_delta
    x_cur[:3,:3] *= ang_axis_point_to_mat(s * ang_delta, axis_delta, point_delta)[:3,:3]
    return x_cur

# returns an offset pose using delta values
# For rotations: if in base frame: R = b_O_fe = b_O_f * (b_O_e).T,
#                if in ee frame:   R = e_O_fe = (b_O_e).T * b_O_e * e_O_f
#                WTF: b_O_fb = b_O_f
def pose_offset(b_T_e, pos_delta=3*[0.], rot_delta=3*[0.], 
                pos_in_base_frame=True, rot_in_base_frame=False):
    b_p_e, b_O_e = PoseConv.to_pos_rot(b_T_e)
    p, R = PoseConv.to_pos_rot(pos_delta, rot_delta)
    if pos_in_base_frame:
        b_p_f = b_p_e + p
    else:
        b_p_f = b_p_e + b_O_e * p
    if rot_in_base_frame:
        b_O_f = R * b_O_e
    else:
        b_O_f = b_O_e * R
    return PoseConv.to_homo_mat(b_p_f, b_O_f)

class TrajPlanner(object):
    def __init__(self, kin):
        self.kin = kin

    def interpolate_ik(self, q_init, x_goal, s_knots, 
                       qd_max=[1.]*6, q_min=6*[-2.*np.pi], q_max=6*[2.*np.pi]):
        x_init = self.kin.forward(q_init)
        q_pts = []
        q_prev = q_init
        for s in s_knots:
            x_cur = pose_interp(x_init, x_goal, s)
            q_pt = self.kin.inverse(x_cur, q_prev, q_min=q_min, q_max=q_max)
            #q_pt = self.kin.inverse_rand_search(x_cur, q_prev,
            #                                    pos_tol=0.001, rot_tol=np.deg2rad(1.))
            if q_pt is None:
                print "IK failed, no solution found"
                print 'q_pts', q_pts
                print 'x_cur', x_cur
                return None
            if np.any(np.abs(q_pt - q_prev) > qd_max):
                print "IK failed, joint difference too high"
                print 'q_pt', q_pt
                print 'q_pts', q_pts
                print 'x_cur', x_cur
                return None
            q_pts.append(q_pt)
            q_prev = q_pt
        q_pts = np.array(q_pts)
        return q_pts

    def min_jerk_interp_ik(self, q_init, x_goal, dur, 
                           vel_i=0., vel_f=0., max_dx=0.03, min_dt=0.2, rot_bias=0.15,
                           qd_max=[1.]*6, q_min=6*[-2.*np.pi], q_max=6*[2.*np.pi]):
        s_i, s_f, sdd_i, sdd_f = 0., 1., 0., 0.
        x_init = self.kin.forward(q_init)
        pos_delta, ang_delta, axis_delta, point_delta = pose_diff(x_goal, x_init)
        if np.linalg.norm(pos_delta) > 0.001:
            if abs(ang_delta) < 0.01:
                # linear move
                sd_i = vel_i / np.linalg.norm(pos_delta)
                sd_f = vel_f / np.linalg.norm(pos_delta)
            else:
                # combo move
                sd_i_lin = vel_i / np.linalg.norm(pos_delta)
                sd_i_rot = vel_i / ang_delta
                sd_i = (sd_i_lin + rot_bias * sd_i_rot) / (1. + rot_bias)
                sd_f_lin = vel_f / np.linalg.norm(pos_delta)
                sd_f_rot = vel_f / ang_delta
                sd_f = (sd_f_lin + rot_bias * sd_f_rot) / (1. + rot_bias)
        else:
            if abs(ang_delta) > 0.01:
                # rotation move
                sd_i = vel_i / ang_delta
                sd_f = vel_f / ang_delta
            else:
                # small move
                sd_i = 0.
                sd_f = 0.

        t_knots, s_knots = sample_min_jerk_knots(s_i, sd_i, sdd_i, s_f, sd_f, sdd_f, 
                                                 dur, max_dx, min_dt)
        #print s_knots, 'snots'*10
        q_waypts = self.interpolate_ik(q_init, x_goal, s_knots, 
                                       qd_max=qd_max, q_min=q_min, q_max=q_max)
        if q_waypts is None:
            return None, None
        return np.array(t_knots), np.array(q_waypts)

    def min_jerk_interp_q(self, q_init, q_goal, dur,
                          vel_i=0., vel_f=0., max_dq=0.01, min_dt=0.2):
        t_knots, s_knots = sample_min_jerk_knots(0., vel_i, 0., 1., vel_f, 0., 
                                                 dur, max_dq, min_dt)
        q_waypts = self.interpolate_q(q_init, q_goal, s_knots)
        return np.array(t_knots), np.array(q_waypts)

    # muliplies weights times joint differences to find maximum scaled joint difference
    # finds the duration from the average velocity and joint difference
    def min_jerk_interp_q_vel(self, q_init, q_goal, vel,
                              q_weights=[0.8, 1.0, 0.8, 0.3, 0.2, 0.1], 
                              vel_i=0., vel_f=0., max_dq=0.01, min_dt=0.2):
        dur = np.max(np.abs((q_goal - q_init)*q_weights)) / vel
        return self.min_jerk_interp_q(q_init, q_goal, dur,
                                      vel_i=vel_i, vel_f=vel_f, max_dq=max_dq, min_dt=min_dt)

    def interpolate_q(self, q_init, q_final, s_knots):
        q_init, q_final = np.array(q_init), np.array(q_final)
        return np.array([q_init + (q_final - q_init)*s for s in s_knots])

