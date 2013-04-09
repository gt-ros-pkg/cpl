#! /usr/bin/python

import numpy as np

import roslib
roslib.load_manifest("ur_cart_move")
import rospy

from actionlib import SimpleActionClient

from ur_cart_move.msg import SplineTrajAction, SplineTrajGoal
from spline_traj_executor import SplineTraj
from traj_planner import TrajPlanner, pose_offset

def main():
    from ur_cart_move.ur_cart_move import load_ur_robot
    import sys

    rospy.init_node("test_spline_traj_as")
    client = SimpleActionClient('spline_traj_as', SplineTrajAction)

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
    traj = SplineTraj.generate(t_knots_t, q_waypts_t)

    goal = SplineTrajGoal(traj=traj.to_trajectory_msg())
    client.wait_for_server()
    client.send_goal(goal)
    client.wait_for_result()
    print client.get_result()

if __name__ == '__main__':
    main()
