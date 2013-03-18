#! /usr/bin/python

import numpy as np

import roslib
roslib.load_manifest("ur_cart_move")
roslib.load_manifest("ar_track_alvar")
roslib.load_manifest('tf')
roslib.load_manifest('robotiq_c_model_control')
import rospy

import tf
from geometry_msgs.msg import PoseStamped
from actionlib import SimpleActionClient

from ar_track_alvar.msg import AlvarMarkers
from hrl_geom.pose_converter import PoseConv
from traj_planner import TrajPlanner, pose_offset, pose_interp
from spline_traj_executor import SplineTraj
from ur_cart_move.msg import SplineTrajAction, SplineTrajGoal
from ur_cart_move.ur_cart_move import ArmInterface, RAVEKinematics
from robotiq_c_model_control.robotiq_c_ctrl import RobotiqCGripper

class BinManager(object):
    def __init__(self):
        self.table_offset = rospy.get_param("~table_offset", -0.2)
        self.bin_height = rospy.get_param("~bin_height", 0.1)
        self.ar_offset = rospy.get_param("~ar_offset", 0.115)
        self.grasp_height = rospy.get_param("~grasp_height", 0.10)
        self.grasp_rot = rospy.get_param("~grasp_rot", 0.0)
        self.grasp_lift = rospy.get_param("~grasp_lift", 0.18)
        self.table_cutoff = rospy.get_param("~table_cutoff", 0.05)
        self.waypt_offset = rospy.get_param("~waypt_offset", 0.25)
        self.waypt_robot_min_dist = rospy.get_param("~waypt_robot_min_dist", -0.20)

        self.pregrasp_vel = rospy.get_param("~pregrasp_vel", 0.10)
        self.grasp_dur = rospy.get_param("~grasp_dur", 3.50)
        self.grasp_vel = rospy.get_param("~grasp_vel", 0.03)

        cmd_prefix = rospy.get_param("~cmd_prefix", "")
        sim_prefix = rospy.get_param("~sim_prefix", "/sim2")
        tf_list = tf.TransformListener()
        self.arm_cmd = ArmInterface(timeout=0., topic_prefix=cmd_prefix)
        self.arm_sim = ArmInterface(timeout=0., topic_prefix=cmd_prefix)
        self.gripper = RobotiqCGripper()
        self.kin = RAVEKinematics()
        self.traj_plan = TrajPlanner(self.kin)

        self.ar_poses = {}
        self.ar_sub = rospy.Subscriber("/ar_pose_marker", AlvarMarkers, 
                                       self.ar_cb)
        self.pose_pub = rospy.Publisher('/test', PoseStamped)
        #self.move_bin = rospy.ServiceProxy('/move_bin', MoveBin)
        self.traj_as = SimpleActionClient('spline_traj_as', SplineTrajAction)

        print 'a'
        rospy.sleep(0.5)
        tf_list.waitForTransform('/base_link', '/lifecam1_optical_frame', 
                                 rospy.Time(), rospy.Duration(3.0))
        print 'b'
        while not rospy.is_shutdown():
            try:
                now = rospy.Time.now()
                tf_list.waitForTransform('/base_link', '/lifecam1_optical_frame', 
                                         now, rospy.Duration(3.0))
                pose = tf_list.lookupTransform('/base_link', '/lifecam1_optical_frame', 
                                               now)
                print 'c'
                break
            except:
                continue
        self.camera_pose = PoseConv.to_homo_mat(pose)
        print 'd'

        self.traj_as.wait_for_server()
        print 'e'
        self.gripper.wait_for_connection()
        print 'f'
        if (not self.arm_cmd.wait_for_states(timeout=1.) or
            not self.arm_sim.wait_for_states(timeout=1.)):
            print 'Arms not connected!'
        print 'g'
        #while not rospy.is_shutdown():
        #    try:
        #        rospy.wait_for_service('/move_bin', timeout=0.01)
        #        break
        #    except (rospy.ROSException):
        #        continue

    def ar_cb(self, msg):
        cur_time = rospy.get_time()
        for marker in msg.markers:
            marker.pose.header = marker.header
            self.ar_poses[marker.id] = [cur_time, marker.pose]

    def find_grasp_pose(self, ar_pose, is_table):
        ang = np.arctan2(ar_pose[1,0], ar_pose[0,0])
        ar_pose_fixed = PoseConv.to_homo_mat(
                [float(ar_pose[0,3]), float(ar_pose[1,3]), 
                 float(is_table*self.table_offset+self.bin_height)], 
                [0., 0., ang])
        grasp_offset = PoseConv.to_homo_mat(
                [0., self.ar_offset, self.grasp_height],
                [0., np.pi/2, self.grasp_rot])
        return ar_pose_fixed*grasp_offset

    def create_bin_waypts(self, ar_grasp, ar_place):

        grasp_is_table = ar_grasp[2,3] < self.table_cutoff
        grasp_pose = self.find_grasp_pose(ar_grasp, grasp_is_table)
        pregrasp_pose = grasp_pose.copy()
        pregrasp_pose[2,3] += self.grasp_lift

        place_is_table = ar_place[2,3] < self.table_cutoff
        place_pose = self.find_grasp_pose(ar_place, place_is_table)
        preplace_pose = place_pose.copy()
        preplace_pose[2,3] += self.grasp_lift

        mid_pts = []
        if np.logical_xor(place_is_table, grasp_is_table):
            waypt_line = [0., 1., self.waypt_offset]
            grasp_pt = [grasp_pose[0,3], grasp_pose[1,3], 1.]
            place_pt = [place_pose[0,3], place_pose[1,3], 1.]
            waypt_pt = np.cross(waypt_line, np.cross(grasp_pt, place_pt))
            waypt_pt /= waypt_pt[2]
            waypt_pt[0] = min(waypt_pt[0], self.waypt_robot_min_dist)
            waypt = pose_interp(pregrasp_pose, preplace_pose, 0.5)
            waypt[:3,3] = np.mat([waypt_pt[0], waypt_pt[1], 
                                  self.grasp_height+self.bin_height+self.grasp_lift]).T
            mid_pts.append(waypt)

        waypts = (  [pregrasp_pose, grasp_pose, pregrasp_pose] 
                  + mid_pts 
                  + [preplace_pose, place_pose, preplace_pose])
        if False:
            r = rospy.Rate(1)
            for pose in waypts:
                self.pose_pub.publish(PoseConv.to_pose_stamped_msg("/base_link", pose))
                r.sleep()
            print pregrasp_pose, grasp_pose, pregrasp_pose, mid_pts, preplace_pose, place_pose, preplace_pose

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
        return pregrasp_pose, grasp_pose, mid_pts, preplace_pose, place_pose

    def plan_move_traj(self, pregrasp_pose, mid_pts, preplace_pose, place_pose):

        q_init = self.arm_cmd.get_q()
        x_init = self.kin.forward(q_init)

        q_knots = [q_init]
        t_knots = [0.]

        # move upward to pregrasp pose
        t_knots_new, q_knots_new = self.traj_plan.min_jerk_interp_ik(
                q_init=q_knots[-1], x_goal=pregrasp_pose, dur=self.grasp_dur,
                vel_i=0., vel_f=self.grasp_vel)
        q_knots.extend(q_knots_new[1:])
        t_knots.extend(t_knots[-1] + t_knots_new[1:])

        # move to preplace
        last_pose = pregrasp_pose
        for mid_pose in mid_pts + [preplace_pose]: 
            q_mid_pose = self.kin.inverse_rand_search(mid_pose, q_knots[-1],
                                                      pos_tol=0.001, rot_tol=np.deg2rad(1.))
            q_knots.append(q_mid_pose)
            dist = np.linalg.norm(mid_pose[:3,3]-last_pose[:3,3])
            t_knots.append(t_knots[-1] + dist / self.pregrasp_vel)
            last_pose = mid_pose

        # move downward to place pose
        t_knots_new, q_knots_new = self.traj_plan.min_jerk_interp_ik(
                q_init=q_knots[-1], x_goal=place_pose, dur=self.grasp_dur,
                vel_i=self.grasp_vel, vel_f=0.)
        q_knots.extend(q_knots_new[1:])
        t_knots.extend(t_knots[-1] + t_knots_new[1:])

        move_traj = SplineTraj.generate(t_knots, q_knots)
        return move_traj

    def plan_retreat_traj(self, preplace_pose, place_pose):
        q_init = self.arm_cmd.get_q()
        x_init = self.kin.forward(q_init)

        q_knots = [q_init]
        t_knots = [0.]

        # move upward to preplace pose
        t_knots_new, q_knots_new = self.traj_plan.min_jerk_interp_ik(
                q_init=q_knots[-1], x_goal=preplace_pose, dur=self.grasp_dur,
                vel_i=0., vel_f=self.grasp_vel)
        q_knots.extend(q_knots_new[1:])
        t_knots.extend(t_knots[-1] + t_knots_new[1:])

        q_home = [-2.75203516454, -1.29936272152, 1.97292018645, 
                  -2.28456617769, -1.5054511996, -1.1]
        q_knots.append(q_home)
        x_home = self.kin.forward(q_home)
        dist = np.linalg.norm(x_home[:3,3]-preplace_pose[:3,3])
        t_knots.append(t_knots[-1] + dist / self.pregrasp_vel)

        retreat_traj = SplineTraj.generate(t_knots, q_knots)
        return retreat_traj

    def plan_home_traj(self):
        q_init = self.arm_cmd.get_q()
        x_init = self.kin.forward(q_init)

        q_knots = [q_init]
        t_knots = [0.]

        q_home = [-2.75203516454, -1.29936272152, 1.97292018645, 
                  -2.28456617769, -1.5054511996, -1.1]
        q_knots.append(q_home)
        x_home = self.kin.forward(q_home)
        dist = np.linalg.norm(x_home[:3,3]-x_init[:3,3])
        t_knots.append(t_knots[-1] + dist / self.pregrasp_vel)

        home_traj = SplineTraj.generate(t_knots, q_knots)
        return home_traj

    def plan_grasp_traj(self, pregrasp_pose, grasp_pose):

        q_init = self.arm_cmd.get_q()
        x_init = self.kin.forward(q_init)

        q_knots = [q_init]
        t_knots = [0.]

        # move to pregrasp pose
        q_pregrasp = self.kin.inverse_rand_search(pregrasp_pose, q_knots[-1],
                                                  pos_tol=0.001, rot_tol=np.deg2rad(1.))
        q_knots.append(q_pregrasp)
        dist = np.linalg.norm(pregrasp_pose[:3,3]-x_init[:3,3])
        t_knots.append(t_knots[-1] + dist / self.pregrasp_vel)

        # move downward to grasp pose
        t_knots_new, q_knots_new = self.traj_plan.min_jerk_interp_ik(
                q_init=q_knots[-1], x_goal=grasp_pose, dur=self.grasp_dur,
                vel_i=self.grasp_vel, vel_f=0.)
        q_knots.extend(q_knots_new[1:])
        t_knots.extend(t_knots[-1] + t_knots_new[1:])

        grasp_traj = SplineTraj.generate(t_knots, q_knots)
        return grasp_traj

    def execute_traj(self, traj):
        goal = SplineTrajGoal(traj=traj.to_trajectory_msg())
        self.traj_as.wait_for_server()
        self.traj_as.send_goal(goal)
        self.traj_as.wait_for_result()
        print self.traj_as.get_result()

    def move_bin(self, grasp_tag, place_tag):
        (pregrasp_pose, grasp_pose, mid_pts, 
         preplace_pose, place_pose) = self.create_bin_waypts(grasp_tag, place_tag)
        grasp_traj = self.plan_grasp_traj(pregrasp_pose, grasp_pose)
        self.execute_traj(grasp_traj)
        self.gripper.close(block=True)
        move_traj = self.plan_move_traj(pregrasp_pose, mid_pts, preplace_pose, place_pose)
        self.execute_traj(move_traj)
        self.gripper.goto(0.042, 0., 0., block=True)
        rospy.sleep(1.0)
        retreat_traj = self.plan_retreat_traj(preplace_pose, place_pose)
        self.execute_traj(retreat_traj)

    def do_thing(self):
        raw_input("Move to home")
        home_traj = self.plan_home_traj()
        self.execute_traj(home_traj)
        if self.gripper.is_reset():
            self.gripper.reset()
            self.gripper.activate()
        while not rospy.is_shutdown():
            if self.gripper.get_pos() != 0.042:
                self.gripper.goto(0.042, 0., 0., block=False)
            raw_input("Ready")
            grasp_tag = self.camera_pose * PoseConv.to_homo_mat(self.ar_poses[5][1])
            place_tag = self.camera_pose * PoseConv.to_homo_mat(self.ar_poses[6][1])
            self.move_bin(grasp_tag, place_tag)

def main():
    rospy.init_node("bin_manager")
    bm = BinManager()
    bm.do_thing()
    rospy.spin()

if __name__ == "__main__":
    main()
