#!/usr/bin/python

import numpy as np

import roslib
roslib.load_manifest('rospy')
roslib.load_manifest('interactive_markers')
roslib.load_manifest('hrl_geom')
roslib.load_manifest('ur_cart_move')

import rospy
from interactive_markers.interactive_marker_server import InteractiveMarkerServer 
from interactive_markers.interactive_marker_server import InteractiveMarker
from interactive_markers.interactive_marker_server import InteractiveMarkerControl
from interactive_markers.interactive_marker_server import Marker

from interactive_markers.menu_handler import MenuHandler

from hrl_geom.pose_converter import PoseConv
from ur_cart_move.ur_cart_move import load_ur_robot, ArmInterface

rospy.init_node('grasp_tester')
arm, kin, arm_behav = load_ur_robot()
grasp_prefix = rospy.get_param("~grasp_prefix", "")
arm_grasp = ArmInterface(timeout=1., topic_prefix=grasp_prefix)
place_prefix = rospy.get_param("~place_prefix", "")
arm_place = ArmInterface(timeout=1., topic_prefix=place_prefix)
place_pose = PoseConv.to_pose_msg([-0.4, -0.4, -0.1], [0., np.pi/2., 0.])

def proc_grasp_fb(fb):
    pose = PoseConv.to_homo_mat(fb.pose)
    sol = kin.inverse(pose, arm.get_q())
    if sol is not None:
        if True:
            print 80*'-'
            print pose
            print sol
        if False:
            print np.linalg.cond(kin.jacobian(sol))
        arm_grasp.cmd_pos_vel_acc(sol, [0.]*6, [0.]*6)
        
def proc_place_fb(fb):
    global place_pose
    pose = PoseConv.to_homo_mat(fb.pose)
    place_pose = pose
    sol = kin.inverse(pose, arm.get_q())
    if sol is not None:
        arm_place.cmd_pos_vel_acc(sol, [0.]*6, [0.]*6)

def menu_press(fb):
    global place_pose
    base_B_ee = PoseConv.to_homo_mat(fb.pose)
    ee_B_offset = np.mat(np.eye(4))
    ee_B_offset[0,3] = -0.1
    base_B_offset = base_B_ee * ee_B_offset 
    q_offset = kin.inverse(base_B_offset, arm.get_q())
    arm_behav.move_to_q(q_offset, velocity=0.5)
    ik_traj = arm_behav.interpolated_ik(arm.get_q(), base_B_ee, 1.0, 10)
    arm_behav.exec_parab_blend(ik_traj, 1.0) 
    rospy.sleep(1.)
    ik_traj = arm_behav.interpolated_ik(arm.get_q(), base_B_offset, 1.0, 10)
    arm_behav.exec_parab_blend(ik_traj, 1.0) 

    ee_B_offset2 = np.mat(np.eye(4))
    ee_B_offset2[0,3] = -0.3
    base_B_offset2 = place_pose * ee_B_offset2 
    arm_behav.move_to_x(base_B_offset2, velocity=0.5)
    ik_traj = arm_behav.interpolated_ik(arm.get_q(), place_pose, 1.0, 10)
    arm_behav.exec_parab_blend(ik_traj, 1.0) 
    rospy.sleep(1.)
    ik_traj = arm_behav.interpolated_ik(arm.get_q(), base_B_offset2, 2.0, 10)
    arm_behav.exec_parab_blend(ik_traj, 2.0) 

def make_pose_imkr(name, frame_id):
    pose_imkr = InteractiveMarker()
    pose_imkr.header.frame_id = frame_id
    pose_imkr.name = name
    pose_imkr.scale = 0.2
    mkr = Marker()
    mkr.type = Marker.CYLINDER
    mkr.scale.x = 0.09
    mkr.scale.y = 0.09
    mkr.scale.z = 0.03
    mkr.color.r = 0.5
    mkr.color.g = 0.5
    mkr.color.b = 0.5
    mkr.color.a = 1.0
    mkr.pose.orientation.y = 1
    mkr.pose.orientation.w = 1
    imkr_ctrl = InteractiveMarkerControl()
    imkr_ctrl.name = 'wrist'
    imkr_ctrl.always_visible = True
    imkr_ctrl.markers.append(mkr)
    pose_imkr.controls.append(imkr_ctrl)

    names = {'x' : [1, 0, 0, 1], 'y' : [0, 1, 0, 1], 'z' : [0, 0, 1, 1]}
    ctrl_types = {'move_%s' : InteractiveMarkerControl.MOVE_AXIS, 
                  'rotate_%s' : InteractiveMarkerControl.ROTATE_AXIS}
    for name in names:
        for ctrl_type in ctrl_types:
            ctrl = InteractiveMarkerControl()
            ctrl.name = ctrl_type % name
            ctrl.interaction_mode = ctrl_types[ctrl_type]
            q = names[name]
            ctrl.orientation.x = q[0]
            ctrl.orientation.y = q[1]
            ctrl.orientation.z = q[2]
            ctrl.orientation.w = q[3]
            pose_imkr.controls.append(ctrl)

    return pose_imkr

def main():
    global place_pose
    imkr_srv = InteractiveMarkerServer('grasp_tester')

    grasp_tf_prefix = "" #rospy.get_param("~grasp_tf_prefix", "")
    grasp_frame = grasp_tf_prefix + '/base_link'

    imkr_srv.insert(make_pose_imkr('grasp_imkr', grasp_frame), proc_grasp_fb)
    imkr_srv.applyChanges()
    imkr_srv.setPose('grasp_imkr', PoseConv.to_pose_msg([-0.6, 0.3, 0.1], [0., np.pi/2., 0.]))

    menu_hand = MenuHandler()
    menu_hand.insert('Grasp this', callback=menu_press)
    menu_hand.apply(imkr_srv, 'grasp_imkr')

    place_tf_prefix = "" #rospy.get_param("~place_tf_prefix", "")
    place_frame = place_tf_prefix + '/base_link'

    imkr_srv.insert(make_pose_imkr('place_imkr', place_frame), proc_place_fb)
    imkr_srv.applyChanges()
    imkr_srv.setPose('place_imkr', place_pose)
    imkr_srv.applyChanges()

    rospy.spin()


if __name__ == '__main__':
    main()
