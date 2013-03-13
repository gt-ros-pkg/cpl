#! /usr/bin/python

import numpy as np

import roslib
roslib.load_manifest("ur_cart_move")
import rospy

from geometry_msgs.msg import PoseStamped
from hrl_geom.pose_converter import PoseConv
from traj_planner import TrajPlanner, pose_offset
from ur_cart_move.ur_cart_move import ArmInterface, RAVEKinematics
from hrl_geom.transformations import rotation_matrix as ang_axis_point_to_mat

from ur_cart_move.srv import MoveBin, MoveBinResponse

class BinManipulator(object):
    def __init__(self):
        cmd_prefix = rospy.get_param("~cmd_prefix", "/sim1")
        sim_prefix = rospy.get_param("~sim_prefix", "/sim2")
        self.table_offset = rospy.get_param("~table_offset", -0.2)
        self.bin_height = rospy.get_param("~bin_height", 0.1)
        self.ar_offset = rospy.get_param("~ar_offset", 0.115)
        self.grasp_height = rospy.get_param("~grasp_height", 0.10)
        self.grasp_rot = rospy.get_param("~grasp_rot", 0.0)
        self.grasp_lift = rospy.get_param("~grasp_lift", 0.18)
        self.table_cutoff = rospy.get_param("~table_cutoff", 0.05)
        self.waypt_offset = rospy.get_param("~waypt_offset", 0.25)
        self.arm_cmd = ArmInterface(timeout=0., topic_prefix=cmd_prefix)
        self.arm_sim = ArmInterface(timeout=0., topic_prefix=cmd_prefix)
        self.kin = RAVEKinematics()
        if (not self.arm_cmd.wait_for_states(timeout=1.) or
            not self.arm_sim.wait_for_states(timeout=1.)):
            print 'Arms not connected!'
        self.move_bin_srv = rospy.Service('/move_bin', MoveBin, self.move_bin_handler)
        self.pose_pub = rospy.Publisher('/test', PoseStamped)

    def move_bin_handler(self, req):
        resp = MoveBinResponse()
        grasp_tag = PoseConv.to_homo_mat(req.grasp_tag)
        place_tag = PoseConv.to_homo_mat(req.place_tag)
        resp.success.data = self.move_bin(grasp_tag, place_tag)
        return resp

    def find_grasp_pose(self, ar_pose, is_table):
        ang = np.arctan2(ar_pose[1,0], ar_pose[0,0])
        ar_pose_fixed = PoseConv.to_homo_mat(
                [float(ar_pose[0,3]), float(ar_pose[1,3]), float(is_table*self.table_offset+self.bin_height)], 
                [0., 0., ang])
        grasp_offset = PoseConv.to_homo_mat(
                [0., self.ar_offset, self.grasp_height],
                [np.pi, 0, self.grasp_rot])
        return ar_pose_fixed*grasp_offset

    def move_bin(self, ar_grasp, ar_place):
        q_init = self.arm_cmd.get_q()
        x_init = self.kin.forward(q_init)

        grasp_is_table = ar_grasp[2,3] < self.table_cutoff
        grasp_pose = self.find_grasp_pose(ar_grasp, grasp_is_table)
        lift_pose = grasp_pose.copy()
        lift_pose[2,3] += self.grasp_lift
        
        place_is_table = ar_place[2,3] < self.table_cutoff
        place_pose = self.find_grasp_pose(ar_place, place_is_table)
        preplace_pose = place_pose.copy()
        preplace_pose[2,3] += self.grasp_lift
        waypt_line = [0., 1., self.waypt_offset]
        grasp_pt = [grasp_pose[0,3], grasp_pose[1,3], 1.]
        place_pt = [place_pose[0,3], place_pose[1,3], 1.]

        waypts = []
        if place_is_table + grasp_is_table == 1:
            waypt_pt = np.cross(waypt_line, np.cross(grasp_pt, place_pt))
            waypt_pt /= waypt_pt[2]
            _, euler = PoseConv.to_pos_euler(np.linalg.inv(lift_pose) * preplace_pose)
            waypt = PoseConv.to_homo_mat([waypt_pt[0], waypt_pt[1], 
                                          self.grasp_height+self.bin_height],
                                         [np.pi, 0., euler[2]/2])
            waypts.append(waypt)

        all_poses = [lift_pose, grasp_pose, lift_pose] + waypts + [preplace_pose, place_pose, preplace_pose]
        r = rospy.Rate(1)
        for pose in all_poses:
            self.pose_pub.publish(PoseConv.to_pose_stamped_msg("/base_link", pose))
            r.sleep()
        print lift_pose, grasp_pose, lift_pose, waypts, preplace_pose, place_pose, preplace_pose

        # bin_pose
        # ar_pose[0,0], 
        # x =
        # z = is_table*self.table_offset
        # bin_grasp/bin_place are [x, y, r, is_table]
        # move to pregrasp, above the bin (traversal)
        # move down towards the bin
        # grasp the bin
        # lift up
        # move to preplace (traversal)
        # move down
        # release bin
        return True

def main():
    rospy.init_node("bin_manipulator")
    bm = BinManipulator()
    rospy.spin()

if __name__ == "__main__":
    main()
