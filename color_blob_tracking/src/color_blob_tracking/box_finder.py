#!/usr/bin/python

import numpy as np

import roslib
roslib.load_manifest('rospy')
roslib.load_manifest('interactive_markers')
roslib.load_manifest('hrl_geom')
#roslib.load_manifest('ur_cart_move')

import rospy
from geometry_msgs.msg import Pose
from interactive_markers.interactive_marker_server import InteractiveMarkerServer 
from interactive_markers.interactive_marker_server import InteractiveMarker
from interactive_markers.interactive_marker_server import InteractiveMarkerControl
from interactive_markers.interactive_marker_server import Marker

from interactive_markers.menu_handler import MenuHandler

from hrl_geom.pose_converter import PoseConv

rospy.init_node('box_finder')
imkr_srv = InteractiveMarkerServer('box_finder')

#box_quat = [0., 0., 0., 1.]
box_size = np.array([1., 1., 1.])
last_pose1 = [[0.,0.,0.],[0.,0.,0.,1.]]
last_pose2 = [[1.,1.,1.],[0.,0.,0.,1.]]

def pose1_fb(fb):
    global box_size, last_pose1
    cur_pose1 = PoseConv.to_pos_quat(fb.pose)
    base_B_bcorner = PoseConv.to_homo_mat(fb.pose)
    if not np.allclose(last_pose1[0], cur_pose1[0]):
        # position change
        base_B_old_tcorner = PoseConv.to_homo_mat(imkr_srv.get('pose2').pose)
        box_size = (base_B_bcorner[:3,:3].T * (base_B_old_tcorner[:3,3] - base_B_bcorner[:3,3])).A.T[0]

    bcorner_B_tcorner = PoseConv.to_homo_mat(box_size.tolist(), [0., 0., 0.])
    bcorner_B_pivot = PoseConv.to_homo_mat((box_size/2.).tolist(), [0., 0., 0.])
    base_B_tcorner = base_B_bcorner * bcorner_B_tcorner
    base_B_pivot = base_B_bcorner * bcorner_B_pivot
    imkr_srv.setPose('pose2', PoseConv.to_pose_msg(base_B_tcorner))

    box_size = (base_B_bcorner[:3,:3].T * (base_B_tcorner[:3,3] - base_B_bcorner[:3,3])).A.T[0]
    cube_imkr = imkr_srv.get('box')
    cube_imkr.controls[0].markers[0].scale.x = box_size[0]
    cube_imkr.controls[0].markers[0].scale.y = box_size[1]
    cube_imkr.controls[0].markers[0].scale.z = box_size[2]
    cube_imkr.pose = PoseConv.to_pose_msg(base_B_pivot)
    imkr_srv.erase('box')
    imkr_srv.insert(cube_imkr)

    imkr_srv.applyChanges()

    last_pose1 = PoseConv.to_pos_quat(fb.pose)
    print_state()

def pose2_fb(fb):
    global box_size, last_pose2
    cur_pose2 = PoseConv.to_pos_quat(fb.pose)
    base_B_tcorner = PoseConv.to_homo_mat(fb.pose)
    if not np.allclose(last_pose2[0], cur_pose2[0]):
        # position change
        base_B_old_bcorner = PoseConv.to_homo_mat(imkr_srv.get('pose1').pose)
        box_size = (base_B_tcorner[:3,:3].T * (base_B_tcorner[:3,3] - base_B_old_bcorner[:3,3])).A.T[0]

    bcorner_B_tcorner = PoseConv.to_homo_mat(box_size.tolist(), [0., 0., 0.])
    bcorner_B_pivot = PoseConv.to_homo_mat((box_size/2.).tolist(), [0., 0., 0.])
    base_B_bcorner = base_B_tcorner * bcorner_B_tcorner ** -1
    base_B_pivot = base_B_bcorner * bcorner_B_pivot
    imkr_srv.setPose('pose1', PoseConv.to_pose_msg(base_B_bcorner))

    box_size = (base_B_bcorner[:3,:3].T * (base_B_tcorner[:3,3] - base_B_bcorner[:3,3])).A.T[0]
    cube_imkr = imkr_srv.get('box')
    cube_imkr.controls[0].markers[0].scale.x = box_size[0]
    cube_imkr.controls[0].markers[0].scale.y = box_size[1]
    cube_imkr.controls[0].markers[0].scale.z = box_size[2]
    cube_imkr.pose = PoseConv.to_pose_msg(base_B_pivot)
    imkr_srv.erase('box')
    imkr_srv.insert(cube_imkr)

    imkr_srv.applyChanges()

    last_pose2 = PoseConv.to_pos_quat(fb.pose)
    print_state()

def print_state():
    print "Bottom corner:"
    print PoseConv.to_pos_quat(imkr_srv.get('pose1').pose)
    print "Top corner:"
    print PoseConv.to_pos_quat(imkr_srv.get('pose2').pose)
    print "Box size:"
    print box_size

def menu_press(fb):
    print fb
    #pose = PoseConv.to_homo_mat(fb.pose)
    #sol = kin.inverse(pose, arm.get_q())
    #arm_behav.move_to_q(sol)

def make_pose_ctrl(name, x):
    pose_imkr = InteractiveMarker()
    pose_imkr.header.frame_id = '/base_link'
    pose_imkr.name = name
    pose_imkr.scale = 0.2
    pose_imkr.pose.position.x = x[0]
    pose_imkr.pose.position.y = x[1]
    pose_imkr.pose.position.z = x[2]
    names = {'x' : [1, 0, 0, 1], 'y' : [0, 1, 0, 1], 'z' : [0, 0, 1, 1]}
    ctrl_types = {'move_%s' : InteractiveMarkerControl.MOVE_AXIS, 
                  'rotate_%s' : InteractiveMarkerControl.ROTATE_AXIS}
    ctrls = []
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

    cube_imkr = InteractiveMarker()
    cube_imkr.header.frame_id = '/base_link'
    cube_imkr.name = 'box'
    cube_imkr.scale = 1.0
    cube_imkr.pose.position.x = 0.5
    cube_imkr.pose.position.y = 0.5
    cube_imkr.pose.position.z = 0.5
    cube_mkr = Marker()
    cube_mkr.type = Marker.CUBE
    cube_mkr.scale.x = 1.0
    cube_mkr.scale.y = 1.0
    cube_mkr.scale.z = 1.0
    cube_mkr.color.r = 0.2
    cube_mkr.color.g = 0.8
    cube_mkr.color.b = 0.2
    cube_mkr.color.a = 0.3
    #cube_mkr.pose.orientation.y = 1
    cube_mkr.pose.orientation.w = 1
    cube_imkr_ctrl = InteractiveMarkerControl()
    cube_imkr_ctrl.name = 'box'
    cube_imkr_ctrl.always_visible = True
    cube_imkr_ctrl.markers.append(cube_mkr)
    cube_imkr.controls.append(cube_imkr_ctrl)
    imkr_srv.insert(cube_imkr)
    imkr_srv.applyChanges()

    pose1_imkr = make_pose_ctrl('pose1', [0.0, 0.0, 0.0])
    imkr_srv.insert(pose1_imkr, pose1_fb)
    imkr_srv.applyChanges()
    pose2_imkr = make_pose_ctrl('pose2', [1., 1., 1.])
    imkr_srv.insert(pose2_imkr, pose2_fb)
    imkr_srv.applyChanges()

    imkr_srv.applyChanges()
    #imkr_srv.setPose('pose_finder', PoseConv.to_pose_msg(kin.forward(arm.get_q())))

    menu_hand = MenuHandler()
    menu_hand.insert('Go to pose', callback=menu_press)
    menu_hand.apply(imkr_srv, 'box')

    imkr_srv.applyChanges()

    rospy.spin()


if __name__ == '__main__':
    main()
