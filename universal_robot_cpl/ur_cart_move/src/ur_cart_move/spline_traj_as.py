#! /usr/bin/python

import numpy as np

import roslib
roslib.load_manifest("ur_cart_move")
roslib.load_manifest("actionlib")
import rospy

from actionlib import SimpleActionServer

from ur_cart_move.ur_cart_move import ArmInterface
from spline_traj_executor import TrajExecutor, SplineTraj
from ur_cart_move.msg import SplineTrajAction, SplineTrajFeedback, SplineTrajResult

class SplineTrajActionServer(object):
    feedback = SplineTrajFeedback()
    result = SplineTrajResult()

    def __init__(self, name, arm):
        self.name = name
        self.arm = arm
        self.max_vals = rospy.get_param('~max_joint_values')
        print self.max_vals
        self.asrv = SimpleActionServer(name, SplineTrajAction, 
                                       execute_cb=self.execute_cb, auto_start=False)
        self.asrv.start()

    def execute_cb(self, goal):
        traj = SplineTraj.from_trajectory_msg(goal.traj)
        max_traj_vals = {}
        (max_traj_vals['q'], max_traj_vals['qd'], 
         max_traj_vals['qdd'], max_traj_vals['qddd']) = traj.max_values()
        q_init, _, _ = traj.sample(0.)
        q_goal, _, _ = traj.sample(traj.duration)
        rospy.loginfo("\n%s: Received trajectory\nfrom [%s]\nto   [%s]\nwith duration %f" %
                (self.name,
                 ", ".join(["%1.4f" % q for q in q_init]), 
                 ", ".join(["%1.4f" % q for q in q_goal]), 
                 traj.duration))
        for val_name in max_traj_vals:
            if np.any(max_traj_vals[val_name] > self.max_vals[val_name]):
                rospy.loginfo('\n%s: Max %s exceeded: \nThresh [%s]\nTraj  [%s]' %
                    (self.name, val_name,
                     ", ".join(["%1.4f" % v for v in self.max_vals[val_name]]), 
                     ", ".join(["%1.4f" % v for v in max_traj_vals[val_name]])))
                self.result.success, self.result.is_robot_running = False, True
                self.asrv.set_aborted(self.result)


        self.arm.unlock_security_stop()
        r = rospy.Rate(2.*self.arm.CONTROL_RATE)
        start_time = rospy.get_time()
        while True:

            if rospy.is_shutdown() or self.asrv.is_preempt_requested():
                rospy.loginfo("\n%s: Preempted." % self.name)
                self.result.success, self.result.is_robot_running = False, True
                self.asrv.set_preempted(self.result)
                return

            if not self.arm.is_running_mode():
                rospy.loginfo("\n%s: Robot is not running." % self.name)
                self.result.success, self.result.is_robot_running = False, False
                self.asrv.set_aborted(self.result)
                return

            t = rospy.get_time() - start_time
            if t > traj.duration:
                t = traj.duration

            q, qd, qdd = traj.sample(t)
            self.arm.cmd_pos_vel_acc(q, qd, qdd)

            self.feedback.q, self.feedback.qd, self.feedback.qdd = q, qd, qdd
            self.feedback.t, self.feedback.duration = t, traj.duration
            self.asrv.publish_feedback(self.feedback)

            if t == traj.duration:
                self.result.success, self.result.is_robot_running = True, True
                rospy.loginfo("\n%s: Trajectory complete." % self.name)
                self.asrv.set_succeeded(self.result)
                return

            r.sleep()

if __name__ == '__main__':
    rospy.init_node('spline_traj_as')
    arm_prefix = rospy.get_param("~arm_prefix", "")
    arm = ArmInterface(timeout=3., topic_prefix=arm_prefix)
    SplineTrajActionServer(rospy.get_name(), arm)
    rospy.spin()
