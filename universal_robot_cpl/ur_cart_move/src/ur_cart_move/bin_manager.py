#! /usr/bin/python

import numpy as np
from scipy.spatial import KDTree
from collections import deque
import yaml
from threading import RLock

import roslib
roslib.load_manifest("ur_cart_move")
roslib.load_manifest("ar_track_alvar")
roslib.load_manifest('tf')
roslib.load_manifest('robotiq_c_model_control')
roslib.load_manifest('hrl_geom')
roslib.load_manifest('pykdl_utils')
import rospy

import tf
from geometry_msgs.msg import PoseStamped, PoseArray
from actionlib import SimpleActionClient

from ar_track_alvar.msg import AlvarMarkers
from hrl_geom.pose_converter import PoseConv
from traj_planner import TrajPlanner, pose_offset, pose_interp
from spline_traj_executor import SplineTraj
from msg import SplineTrajAction, SplineTrajGoal
from ur_cart_move import ArmInterface, RAVEKinematics
from robotiq_c_model_control.robotiq_c_ctrl import RobotiqCGripper
from pykdl_utils.kdl_kinematics import create_kdl_kin

BIN_HEIGHT_DEFAULT = 0.1
TABLE_OFFSET_DEFAULT = -0.2
TABLE_CUTOFF_DEFAULT = 0.05
VEL_MULT = 4.0

def create_slot_tree(bin_slots):
    pos_data = np.zeros((len(bin_slots),3))
    for i, bid in enumerate(sorted(bin_slots.keys())):
        pos_data[i,:] = bin_slots[bid][0]
    return KDTree(pos_data)

class ARTagManager(object):
    def __init__(self, bin_slots, human_slots=range(3)):
        self.bin_height = rospy.get_param("~bin_height", BIN_HEIGHT_DEFAULT)
        self.table_offset = rospy.get_param("~table_offset", TABLE_OFFSET_DEFAULT)
        self.table_cutoff = rospy.get_param("~table_cutoff", TABLE_CUTOFF_DEFAULT)
        self.filter_window = rospy.get_param("~filter_window", 5.)
        # distance from marker to slot which can be considered unified
        self.ar_unification_thresh = rospy.get_param("~ar_unification_thresh", 0.08)

        self.bin_slots = bin_slots
        self.human_slots = human_slots
        self.slot_tree = create_slot_tree(bin_slots)
        lifecam_kin = create_kdl_kin('base_link', 'lifecam1_optical_frame')
        self.camera_pose = lifecam_kin.forward([])

        self.ar_poses = {}
        self.ar_sub = rospy.Subscriber("/ar_pose_marker", AlvarMarkers, 
                                       self.ar_cb)
        self.lock = RLock()

    def ar_cb(self, msg):
        with self.lock:
            cur_time = rospy.get_time()
            for marker in msg.markers:
                marker.pose.header = marker.header
                if marker.id not in self.ar_poses:
                    self.ar_poses[marker.id] = deque()
                self.ar_poses[marker.id].append([cur_time, marker.pose])
            #for mid in self.ar_poses:
            #    while (len(self.ar_poses[mid]) > 0 and 
            #           cur_time - self.ar_poses[mid][0][0] > self.filter_window):
            #        self.ar_poses[mid].popleft()

    # forces bin location from a pose in the base_link frame
    def set_bin_location(self, mid, pose):
        with self.lock:
            cur_time = rospy.get_time()
            self.ar_poses[mid].clear()
            self.ar_poses[mid].append([cur_time, 
                                       self.camera_pose**-1 * PoseConv.to_homo_mat(pose)])

    def get_available_bins(self):
        with self.lock:
            bins = []
            for mid in self.ar_poses:
                if len(self.ar_poses[mid]) > 0:
                    bins.append(mid)
            return bins

    def clean_ar_pose(self, ar_pose):
        ar_pose_mat = self.camera_pose * PoseConv.to_homo_mat(ar_pose)
        ang = np.arctan2(ar_pose_mat[1,0], ar_pose_mat[0,0])
        return (ar_pose_mat[:3,3].T.A[0], 
                [0., 0., ang])

    def get_bin_pose(self, bin_id):
        with self.lock:
            if bin_id not in self.ar_poses:
                print "Bin ID %d not found!" % bin_id
                return None, None
            pos_list, rot_list = [], []
            for cur_time, pose in self.ar_poses[bin_id]:
                pos, rot = self.clean_ar_pose(pose)
                pos_list.append(pos)
                rot_list.append(rot)
            med_pos, med_rot = np.median(pos_list,0), np.median(rot_list,0)
            is_table = bool(med_pos[2] < self.table_cutoff)
            med_pos[2] = is_table*self.table_offset+self.bin_height
            return (med_pos.tolist(), med_rot.tolist()), is_table

    def get_all_bin_poses(self):
        with self.lock:
            bin_data = {}
            for bin_id in self.get_available_bins():
                bin_pose, is_table = self.get_bin_pose(bin_id)
                bin_data[bin_id] = [bin_pose[0], bin_pose[1], is_table]
            return bin_data

    def get_bin_slot(self, slot_id):
        return self.bin_slots[slot_id]

    def get_slot_ids(self):
        return sorted(self.bin_slots.keys())

    def get_bin_slot_states(self):
        bin_poses = self.get_all_bin_poses()
        bin_ids = sorted(bin_poses.keys())
        bin_pos_data = np.array([bin_poses[bin_id][0] for bin_id in bin_ids])
        dists, inds = self.slot_tree.query(bin_pos_data, k=1, 
                                           distance_upper_bound=self.ar_unification_thresh)

        slot_states = [-1] * len(self.slot_tree.data)
        missing_bins = []
        for i, ind in enumerate(inds):
            bin_id = bin_ids[i]
            if ind == len(slot_states):
                missing_bins.append(bin_id)
                continue
            slot_states[ind] = bin_id
        return slot_states, missing_bins

    # finds the closest empty slot to the pos position
    def get_filled_slots(self, near_human):
        slot_states, _ = self.get_bin_slot_states()
        bins = []
        for ind, slot_state in enumerate(slot_states):
            if slot_state != -1:
                slot_near_human = ind in self.human_slots
                if np.logical_xor(not slot_near_human, near_human):
                    bins.append(slot_state)
        return sorted(bins)

    # finds the closest empty slot to the pos position
    def get_empty_slot(self, near_human, pos=[-0.46, -0.93, -0.1]):
        slot_states, _ = self.get_bin_slot_states()
        dists, inds = self.slot_tree.query(pos, k=len(self.slot_tree.data)) 
        for ind in inds:
            if slot_states[ind] == -1:
                slot_near_human = ind in self.human_slots
                if np.logical_xor(not slot_near_human, near_human):
                    return self.get_slot_ids()[ind]
        return -1

class BinManager(object):
    def __init__(self, arm_prefix, bin_slots):
        self.bin_height = rospy.get_param("~bin_height", BIN_HEIGHT_DEFAULT)
        self.table_offset = rospy.get_param("~table_offset", TABLE_OFFSET_DEFAULT)
        self.table_cutoff = rospy.get_param("~table_cutoff", TABLE_CUTOFF_DEFAULT)

        self.place_offset = rospy.get_param("~place_offset", 0.02)
        self.ar_offset = rospy.get_param("~ar_offset", 0.110)
        self.grasp_height = rospy.get_param("~grasp_height", 0.10)
        self.grasp_rot = rospy.get_param("~grasp_rot", 0.0)
        self.grasp_lift = rospy.get_param("~grasp_lift", 0.18)
        self.waypt_offset = rospy.get_param("~waypt_offset", 0.25)
        self.waypt_robot_min_dist = rospy.get_param("~waypt_robot_min_dist", -0.40)

        self.pregrasp_vel = rospy.get_param("~pregrasp_vel", VEL_MULT*0.10)
        self.grasp_dur = rospy.get_param("~grasp_dur", 3.50/VEL_MULT)
        self.grasp_vel = rospy.get_param("~grasp_vel", 0.03*VEL_MULT)

        self.qd_max = [0.2]*6
        self.q_min = [-4.78, -2.4, 0.3, -3.8, -3.3, -2.*np.pi]
        self.q_max = [-1.2, -0.4, 2.7, -1.6, 0.3, 2.*np.pi]

        sim_prefix = rospy.get_param("~sim_arm_prefix", "/sim2")
        tf_list = tf.TransformListener()
        self.arm_cmd = ArmInterface(timeout=0., topic_prefix=arm_prefix)
        self.arm_sim = ArmInterface(timeout=0., topic_prefix=sim_prefix)
        if arm_prefix not in ["", "/"]:
            self.gripper = None
        else:
            self.gripper = RobotiqCGripper()
        self.kin = RAVEKinematics()
        self.traj_plan = TrajPlanner(self.kin)

        self.ar_man = ARTagManager(bin_slots)
        self.pose_pub = rospy.Publisher('/test', PoseStamped)
        #self.move_bin = rospy.ServiceProxy('/move_bin', MoveBin)
        self.traj_as = SimpleActionClient('spline_traj_as', SplineTrajAction)

        #print 'Waiting on TF'
        #rospy.sleep(0.5)
        #tf_list.waitForTransform('/base_link', '/lifecam1_optical_frame', 
        #                         rospy.Time(), rospy.Duration(3.0))
        #while not rospy.is_shutdown():
        #    try:
        #        now = rospy.Time.now()
        #        tf_list.waitForTransform('/base_link', '/lifecam1_optical_frame', 
        #                                 now, rospy.Duration(3.0))
        #        pose = tf_list.lookupTransform('/base_link', '/lifecam1_optical_frame', 
        #                                       now)
        #        print 'c'
        #        break
        #    except:
        #        continue
        #self.camera_pose = PoseConv.to_homo_mat(pose)
        #print 'Got TF transform for webcam frame'

        print 'Waiting for trajectory AS...', 
        self.traj_as.wait_for_server()
        print 'found.'
        if self.gripper is not None:
            print 'Waiting for gripper...', 
            self.gripper.wait_for_connection()
            print 'found.'
        print 'Waiting for arms...', 
        if (not self.arm_cmd.wait_for_states(timeout=1.) or
            not self.arm_sim.wait_for_states(timeout=1.)):
            print 'Arms not connected!'
        print 'found.'
        self.update_payload(empty=True)

        #while not rospy.is_shutdown():
        #    try:
        #        rospy.wait_for_service('/move_bin', timeout=0.01)
        #        break
        #    except (rospy.ROSException):
        #        continue
    def update_payload(self, empty):
        if empty:
            self.arm_cmd.set_payload(pose=([0., -0.0086, 0.0353], [0.]*3), payload=0.89)
        else:
            self.arm_cmd.set_payload(pose=([-0.03, -0.0086, 0.0453], [0.]*3), payload=1.40)

    def plan_move_traj(self, pregrasp_pose, mid_pts, preplace_pose, place_pose):

        q_init = self.arm_cmd.get_q()
        x_init = self.kin.forward(q_init)

        q_knots = [q_init]
        t_knots = [0.]

        # move upward to pregrasp pose
        start_time = rospy.get_time()
        t_knots_new, q_knots_new = self.traj_plan.min_jerk_interp_ik(
                q_init=q_knots[-1], x_goal=pregrasp_pose, dur=self.grasp_dur,
                vel_i=0., vel_f=self.grasp_vel,
                qd_max=self.qd_max, q_min=self.q_min, q_max=self.q_max)
        if q_knots_new is None:
            print 'move upward to pregrasp pose failed'
            print pregrasp_pose
            print q_knots
            return None
        print 'Planning time:', rospy.get_time() - start_time, len(t_knots_new)
        q_knots.extend(q_knots_new[1:])
        t_knots.extend(t_knots[-1] + t_knots_new[1:])

        # move to preplace
        last_pose = pregrasp_pose
        for mid_pose in mid_pts + [preplace_pose]: 
            #q_mid_pose = self.kin.inverse_rand_search(mid_pose, q_knots[-1],
            #                                          pos_tol=0.001, rot_tol=np.deg2rad(1.))
            q_mid_pose = self.kin.inverse(mid_pose, q_knots[-1],
                                          q_min=self.q_min, q_max=self.q_max)
            if q_mid_pose is None:
                print 'failed move to replace'
                print mid_pose
                print q_knots
                return None
            q_knots.append(q_mid_pose)
            dist = np.linalg.norm(mid_pose[:3,3]-last_pose[:3,3])
            t_knots.append(t_knots[-1] + dist / self.pregrasp_vel)
            last_pose = mid_pose

        # move downward to place pose
        start_time = rospy.get_time()
        t_knots_new, q_knots_new = self.traj_plan.min_jerk_interp_ik(
                q_init=q_knots[-1], x_goal=place_pose, dur=self.grasp_dur,
                vel_i=self.grasp_vel, vel_f=0.,
                qd_max=self.qd_max, q_min=self.q_min, q_max=self.q_max)
        if q_knots_new is None:
            print 'move downward to place pose'
            print place_pose
            print q_knots
            return None
        print 'Planning time:', rospy.get_time() - start_time, len(t_knots_new)
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
        start_time = rospy.get_time()
        t_knots_new, q_knots_new = self.traj_plan.min_jerk_interp_ik(
                q_init=q_knots[-1], x_goal=preplace_pose, dur=self.grasp_dur,
                vel_i=0., vel_f=self.grasp_vel,
                qd_max=self.qd_max, q_min=self.q_min, q_max=self.q_max)
        if q_knots_new is None:
            print 'move upward to preplace pose failed'
            print preplace_pose
            print q_knots
            return None
        print 'Planning time:', rospy.get_time() - start_time, len(t_knots_new)
        q_knots.extend(q_knots_new[1:])
        t_knots.extend(t_knots[-1] + t_knots_new[1:])

        #q_home = [-2.75203516454, -1.29936272152, 1.97292018645, 
        #          -2.28456617769, -1.5054511996, -1.1]
        #q_knots.append(q_home)
        #x_home = self.kin.forward(q_home)
        #dist = np.linalg.norm(x_home[:3,3]-preplace_pose[:3,3])
        #t_knots.append(t_knots[-1] + dist / self.pregrasp_vel)

        retreat_traj = SplineTraj.generate(t_knots, q_knots)
        return retreat_traj

    def plan_home_traj(self):
        q_home = [-2.75203516454, -1.29936272152, 1.97292018645, 
                  -2.28456617769, -1.5054511996, -1.1]
        q_init = self.arm_cmd.get_q()
        start_time = rospy.get_time()
        t_knots, q_knots = self.traj_plan.min_jerk_interp_q_vel(q_init, q_home, self.pregrasp_vel)
        print 'Planning time:', rospy.get_time() - start_time
        home_traj = SplineTraj.generate(t_knots, q_knots)
        #print t_knots, q_knots
        return home_traj

    def plan_grasp_traj(self, pregrasp_pose, grasp_pose):

        q_init = self.arm_cmd.get_q()
        x_init = self.kin.forward(q_init)

        q_knots = [q_init]
        t_knots = [0.]

        # move to pregrasp pose
        #q_pregrasp = self.kin.inverse_rand_search(pregrasp_pose, q_knots[-1],
        #                                          pos_tol=0.001, rot_tol=np.deg2rad(1.))
        q_pregrasp = self.kin.inverse(pregrasp_pose, q_knots[-1],
                                      q_min=self.q_min, q_max=self.q_max)
        if q_pregrasp is None:
            print 'move to pregrasp pose failed'
            self.pose_pub.publish(PoseConv.to_pose_stamped_msg("/base_link", pregrasp_pose))
            print pregrasp_pose
            print q_knots
            return None
        q_knots.append(q_pregrasp)
        dist = np.linalg.norm(pregrasp_pose[:3,3]-x_init[:3,3])
        dur = max(dist / self.pregrasp_vel, 0.5*VEL_MULT)
        t_knots.append(t_knots[-1] + dur)

        # move downward to grasp pose
        start_time = rospy.get_time()
        t_knots_new, q_knots_new = self.traj_plan.min_jerk_interp_ik(
                q_init=q_knots[-1], x_goal=grasp_pose, dur=self.grasp_dur,
                vel_i=self.grasp_vel, vel_f=0.,
                qd_max=self.qd_max, q_min=self.q_min, q_max=self.q_max)
        if q_knots_new is None:
            print 'move downward to grasp pose failed'
            print grasp_pose
            print q_knots
            return None
        print 'Planning time:', rospy.get_time() - start_time, len(t_knots_new)
        q_knots.extend(q_knots_new[1:])
        t_knots.extend(t_knots[-1] + t_knots_new[1:])

        grasp_traj = SplineTraj.generate(t_knots, q_knots)
        return grasp_traj

    def execute_traj(self, traj):
        goal = SplineTrajGoal(traj=traj.to_trajectory_msg())
        self.arm_cmd.unlock_security_stop()
        self.traj_as.wait_for_server()
        self.traj_as.send_goal(goal)
        self.traj_as.wait_for_result()
        result = self.traj_as.get_result()
        rospy.loginfo("Trajectory result:" + str(result))
        return result.success, result.is_robot_running

    def move_bin(self, ar_grasp_id, ar_place_id):
        # bin_pose
        # move to pregrasp, above the bin (traversal)
        # move down towards the bin
        # grasp the bin
        # lift up
        # move to preplace (traversal)
        # move down
        # release bin

        ########################## create waypoints #############################
        ar_grasp_tag_pose, grasp_is_table = self.ar_man.get_bin_pose(ar_grasp_id)
        if ar_grasp_tag_pose is None:
            rospy.loginfo('Failed getting grasp pose')
            return False
        grasp_offset = PoseConv.to_homo_mat(
                [-0.013, self.ar_offset, self.grasp_height],
                [0., np.pi/2, self.grasp_rot])
        grasp_pose = PoseConv.to_homo_mat(ar_grasp_tag_pose) * grasp_offset
        pregrasp_pose = grasp_pose.copy()
        pregrasp_pose[2,3] += self.grasp_lift

        if ar_place_id not in self.ar_man.get_slot_ids():
            rospy.loginfo('Failed getting place pose')
            return False
        ar_place_tag_pos, ar_place_tag_rot, place_is_table = self.ar_man.get_bin_slot(ar_place_id)
        ar_place_tag_pose = (ar_place_tag_pos, ar_place_tag_rot)
        place_offset = PoseConv.to_homo_mat(
                [-0.013, self.ar_offset, self.grasp_height + self.place_offset],
                [0., np.pi/2, self.grasp_rot])
        place_pose = PoseConv.to_homo_mat(ar_place_tag_pose) * place_offset
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

        if False:
            waypts = (  [pregrasp_pose, grasp_pose, pregrasp_pose] 
                      + mid_pts 
                      + [preplace_pose, place_pose, preplace_pose])
            r = rospy.Rate(1)
            for pose in waypts:
                self.pose_pub.publish(PoseConv.to_pose_stamped_msg("/base_link", pose))
                r.sleep()
            print (pregrasp_pose, grasp_pose, pregrasp_pose, mid_pts, 
                   preplace_pose, place_pose, preplace_pose)
        #########################################################################

        grasp_traj = self.plan_grasp_traj(pregrasp_pose, grasp_pose)
        if grasp_traj is None:
            return False
        success, is_robot_running = self.execute_traj(grasp_traj)
        if not success:
            rospy.loginfo('Failed on grasping bin')
            return False
        if self.gripper is not None:
            self.gripper.close(block=True)
            self.update_payload(empty=False)
        move_traj = self.plan_move_traj(pregrasp_pose, mid_pts, preplace_pose, place_pose)
        if move_traj is None:
            return False
        success, is_robot_running = self.execute_traj(move_traj)
        if not success:
            rospy.loginfo('Failed on moving bin')
            return False
        if self.gripper is not None:
            self.gripper.goto(0.042, 0., 0., block=False)
            rospy.sleep(0.5)
            self.update_payload(empty=True)
        retreat_traj = self.plan_retreat_traj(preplace_pose, place_pose)
        if retreat_traj is None:
            return False
        success, is_robot_running = self.execute_traj(retreat_traj)
        if not success:
            rospy.loginfo('Failed on retreating from bin')
            return False
        # update bin location
        self.ar_man.set_bin_location(ar_grasp_id, ar_place_tag_pose)
        return True

    def system_reset(self):
        home_traj = self.plan_home_traj()
        self.execute_traj(home_traj)
        if self.gripper is not None:
            if self.gripper.is_reset():
                self.gripper.reset()
                self.gripper.activate()
        self.update_payload(empty=True)
        if self.gripper is not None:
            if self.gripper.get_pos() != 0.042:
                self.gripper.goto(0.042, 0., 0., block=False)

    def do_random_move_test(self):
        raw_input("Move to home")
        reset = True
        while not rospy.is_shutdown():
            if reset:
                self.system_reset()
            #raw_input("Ready")
            ar_tags = self.ar_man.get_available_bins()
            ar_locs = self.ar_man.get_slot_ids()
            grasp_tag_num = ar_tags[np.random.randint(0,len(ar_tags))]
            place_tag_num = ar_locs[np.random.randint(0,len(ar_locs))]
            if not self.move_bin(grasp_tag_num, place_tag_num):
                reset = True
                print 'Failed moving bin from %d to %d' % (grasp_tag_num, place_tag_num)

    def do_move_demo(self, ar_bin):
        reset = True
        while not rospy.is_shutdown():
            if reset:
                self.system_reset()
            #raw_input("Ready")
            ar_locs = self.ar_man.get_slot_ids()
            place_tag_num = ar_locs[np.random.randint(0,len(ar_locs))]
            if not self.move_bin(ar_bin, place_tag_num):
                reset = True
                print 'Failed moving bin from %d to %d' % (ar_bin, place_tag_num)

def main():
    np.set_printoptions(precision=4)
    rospy.init_node("bin_manager")

    from optparse import OptionParser
    p = OptionParser()
    p.add_option('-f', '--file', dest="filename", default="bin_locs.yaml",
                 help="YAML file of bin locations.")
    p.add_option('-s', '--save', dest="is_save",
                 action="store_true", default=False,
                 help="Save ar tag locations to file.")
    p.add_option('-t', '--test', dest="is_test",
                 action="store_true", default=False,
                 help="Test robot in simulation moving bins around.")
    p.add_option('-d', '--demo', dest="is_demo",
                 action="store_true", default=False,
                 help="Test robot in reality moving a bin around.")
    (opts, args) = p.parse_args()

    if opts.is_save:
        ar_man = ARTagManager()
        rospy.sleep(4.)
        bin_data = ar_man.get_all_bin_poses()
        f = file(opts.filename, 'w')
        for bid in bin_data:
            print bid, bin_data[bid]
        yaml.dump({'data':bin_data}, f)
        f.close()

        poses_pub = rospy.Publisher('/ar_pose_array', PoseArray)
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            poses = PoseArray()
            poses.header.stamp = rospy.Time.now()
            poses.header.frame_id = '/base_link'
            for bid in bin_data:
                poses.poses.append(PoseConv.to_pose_msg(bin_data[bid][:2]))
            poses_pub.publish(poses)
            r.sleep()
    elif opts.is_test:
        f = file(opts.filename, 'r')
        bin_slots = yaml.load(f)['data']
        f.close()
        arm_prefix = "/sim1"
        bm = BinManager(arm_prefix, bin_slots)
        bm.do_random_move_test()
    elif opts.is_demo:
        f = file(opts.filename, 'r')
        bin_slots = yaml.load(f)['data']
        f.close()
        arm_prefix = ""
        bm = BinManager(arm_prefix, bin_slots)
        bm.do_move_demo(5)
    else:
        print 'h'

    
    #bm = BinManager()
    #bm.do_random_move_test()
    #rospy.spin()

if __name__ == "__main__":
    if False:
        import cProfile
        cProfile.run('main()', 'bm_prof')
    else:
        main()
