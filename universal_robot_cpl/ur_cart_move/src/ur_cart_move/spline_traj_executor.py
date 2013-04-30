#! /usr/bin/python

import numpy as np

import roslib
roslib.load_manifest("ur_cart_move")
roslib.load_manifest("trajectory_msgs")
import rospy

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from cubic_spline_interp import CubicSpline
from traj_planner import TrajPlanner, pose_offset

class SplineTraj(object):
    def __init__(self, splines):
        self.splines = splines
        self.duration = splines[0].tk[-1]

    def sample(self, t):
        qsamp = np.array([self.splines[i].sample(t) for i in range(6)])
        return qsamp[:,0], qsamp[:,1], qsamp[:,2]

    def max_values(self):
        max_vals = np.array([self.splines[i].max_values() for i in range(6)])
        return max_vals[:,0], max_vals[:,1], max_vals[:,2], max_vals[:,3] 

    def to_trajectory_msg(self):
        msg = JointTrajectory()
        msg.header.stamp = rospy.Time.now()
        for i, t in enumerate(self.splines[0].tk):
            point = JointTrajectoryPoint()
            for spline in self.splines:
                point.positions.append(spline.q[i])
                point.velocities.append(spline.qd[i])
                point.accelerations.append(spline.qdd[i])
                point.time_from_start = rospy.Duration(t)
            msg.points.append(point)
        return msg

    @staticmethod
    def generate(t_knots, q_waypts, qd_i=[0.]*6, qd_f=[0.]*6, qdd_i=[0.]*6, qdd_f=[0.]*6):
        t_knots, q_waypts = np.array(t_knots), np.array(q_waypts)
        assert(t_knots[0] == 0 and np.all((t_knots[1:]-t_knots[:-1]) > 0))
        MIN_LEN = 4
        if len(t_knots) == 2:
            t_knots = np.array([t_knots[0], t_knots[1]/3., t_knots[1]*2./3., t_knots[1]])
            diff = q_waypts[-1,:] - q_waypts[0,:]
            q_waypts = np.vstack((q_waypts[0,:], q_waypts[0,:]+diff/3., q_waypts[0,:]+2.*diff/3., q_waypts[-1,:]))
        if False and len(t_knots) < MIN_LEN:
            t_knots = np.append(t_knots, [t_knots[-1]]*(MIN_LEN-len(t_knots)))
            q_waypts = np.vstack((q_waypts, np.repeat(q_waypts[-1,:],[MIN_LEN-len(t_knots),1])))
        splines = []
        for i in range(6):
            splines.append(CubicSpline.generate(t_knots.tolist(), q_waypts[:,i].tolist(), 
                                                qd_i[i], qd_f[i], qdd_i[i], qdd_f[i]))
        return SplineTraj(splines)

    @staticmethod
    def from_trajectory_msg(msg):
        t, q, qd, qdd = [], [], [], []
        for point in msg.points:
            t.append(point.time_from_start.to_sec())
            q.append(point.positions)
            qd.append(point.velocities)
            qdd.append(point.accelerations)
        q, qd, qdd = np.array(q), np.array(qd), np.array(qdd)
        splines = []
        for i in range(6):
            splines.append(CubicSpline(np.array(t), q[:,i], qd[:,i], qdd[:,i]))
        return SplineTraj(splines)

class TrajExecutor(object):
    def __init__(self, arm, traj):
        self.arm = arm
        self.traj = traj

    def run(self):
        r = rospy.Rate(self.arm.CONTROL_RATE)
        start_time = rospy.get_time()
        while not rospy.is_shutdown():
            t = rospy.get_time() - start_time
            if t > self.traj.duration:
                t = self.traj.duration
            q, qd, qdd = self.traj.sample(t)
            self.arm.cmd_pos_vel_acc(q, qd, qdd)
            if not self.arm.is_running_mode():
                print 'ROBOT STOPPED'
                return False
            if t == self.traj.duration:
                return True
            r.sleep()
        return False

def main():
    from ur_cart_move.ur_cart_move import load_ur_robot
    import sys

    rospy.init_node("traj_executor")

    params = [float(sys.argv[i]) for i in range(1,len(sys.argv))]
    pos_delta = np.array(params[:3])
    rot_delta = np.array(params[3:6])
    dur = params[6]

    arm, kin, arm_behav = load_ur_robot(topic_prefix="/sim1")
    traj_plan = TrajPlanner(kin)

    q_init = arm.get_q()
    x_init = kin.forward(q_init)

    x_goal_1 = pose_offset(x_init, pos_delta=pos_delta*0.2, rot_delta=rot_delta*0.2)
    t_knots_1, q_waypts_1 = traj_plan.min_jerk_interp_ik(
            q_init=q_init, x_goal=x_goal_1, dur=dur*0.2,
            vel_i=0., vel_f=0.1)

    x_goal_2 = pose_offset(x_goal_1, pos_delta=pos_delta*0.6, rot_delta=rot_delta*0.6)
    t_knots_2, q_waypts_2 = traj_plan.min_jerk_interp_ik(
            q_init=q_waypts_1[-1,:], x_goal=x_goal_2, dur=dur*0.6,
            vel_i=0.1, vel_f=0.1)

    x_goal_3 = pose_offset(x_goal_2, pos_delta=pos_delta*0.2, rot_delta=rot_delta*0.2)
    t_knots_3, q_waypts_3 = traj_plan.min_jerk_interp_ik(
            q_init=q_waypts_2[-1,:], x_goal=x_goal_3, dur=dur*0.2,
            vel_i=0.1, vel_f=0.0)

    t_knots_t = np.hstack((t_knots_1[:-1],
                           t_knots_2[:-1]+t_knots_1[-1],
                           t_knots_3+t_knots_1[-1]+t_knots_2[-1]))
    q_waypts_t = np.vstack((q_waypts_1[:-1,:],
                            q_waypts_2[:-1,:],
                            q_waypts_3))
    traj_exec = TrajExecutor(arm, SplineTraj.generate(t_knots_t, q_waypts_t))
    success = traj_exec.run()

if __name__ == "__main__":
    main()

