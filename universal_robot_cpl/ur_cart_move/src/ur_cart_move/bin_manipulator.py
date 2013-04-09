#! /usr/bin/python

import numpy as np

import roslib
roslib.load_manifest("ur_cart_move")
import rospy

from hrl_geom.pose_converter import PoseConv
from traj_planner import TrajPlanner, pose_offset, pose_interp
from ur_cart_move.ur_cart_move import ArmInterface, RAVEKinematics

from ur_cart_move.srv import MoveBin, MoveBinResponse

class BinManipulator(object):
    def __init__(self):
        cmd_prefix = rospy.get_param("~cmd_prefix", "/sim1")
        sim_prefix = rospy.get_param("~sim_prefix", "/sim2")
        self.arm_cmd = ArmInterface(timeout=0., topic_prefix=cmd_prefix)
        self.arm_sim = ArmInterface(timeout=0., topic_prefix=cmd_prefix)
        self.kin = RAVEKinematics()
        if (not self.arm_cmd.wait_for_states(timeout=1.) or
            not self.arm_sim.wait_for_states(timeout=1.)):
            print 'Arms not connected!'
        self.move_bin_srv = rospy.Service('/move_bin', MoveBin, self.move_bin_handler)

    def move_bin_handler(self, req):
        q_init = self.arm_cmd.get_q()
        x_init = self.kin.forward(q_init)
        resp = MoveBinResponse()
        grasp_tag = PoseConv.to_homo_mat(req.grasp_tag)
        place_tag = PoseConv.to_homo_mat(req.place_tag)
        resp.success.data = True
        return resp

def main():
    rospy.init_node("bin_manipulator")
    bm = BinManipulator()
    rospy.spin()

if __name__ == "__main__":
    main()
