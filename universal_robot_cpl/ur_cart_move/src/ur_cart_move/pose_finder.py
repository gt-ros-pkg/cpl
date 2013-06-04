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
from ur_cart_move.ur_cart_move import load_ur_robot

class PoseFinder(object):
    def __init__(self):
        self.arm, self.kin, self.arm_behav = load_ur_robot()
        test_prefix = rospy.get_param("~test_prefix", "")
        self.arm_test, _, _ = load_ur_robot(topic_prefix=test_prefix)

        self.imkr_srv = InteractiveMarkerServer('pose_finder')

        pose_imkr = InteractiveMarker()
        tf_prefix = rospy.get_param("~tf_prefix", "")
        pose_imkr.header.frame_id = tf_prefix + '/base_link'
        pose_imkr.name = 'pose_finder'
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

        self.imkr_srv.insert(pose_imkr, self.proc_fb)
        self.imkr_srv.applyChanges()
        self.imkr_srv.setPose('pose_finder', PoseConv.to_pose_msg(self.kin.forward(self.arm.get_q())))

        #goto_hand = MenuHandler()
        #goto_hand.insert('Go to pose', callback=self.goto_press)
        #goto_hand.apply(self.imkr_srv, 'pose_finder')

        sideways_hand = MenuHandler()
        sideways_hand.insert('Sideways point', callback=self.sideways_press)
        sideways_hand.insert('Vertical point', callback=self.vertical_press)
        sideways_hand.insert('Go to pose', callback=self.goto_press)
        sideways_hand.apply(self.imkr_srv, 'pose_finder')

        self.imkr_srv.applyChanges()
        self.last_q = self.arm.get_q()

    def proc_fb(self, fb):
        pose = PoseConv.to_homo_mat(fb.pose)
        if False:
            sol = self.kin.inverse(pose, self.last_q)
            self.last_q = sol
        else:
            sol = self.kin.inverse(pose, self.last_q)
        if sol is not None:
            if True:
                print 80*'-'
                print pose
                print sol
                print 'Number solutions: %d' % len(self.kin.inverse_all(pose))
            if False:
                print np.linalg.cond(self.kin.jacobian(sol))
            self.arm_test.cmd_pos_vel_acc(sol, [0.]*6, [0.]*6)

    def goto_press(self, fb):
        pose = PoseConv.to_homo_mat(fb.pose)
        sol = self.kin.inverse(pose, self.arm.get_q())
        self.arm.unlock_security_stop()
        move_velocity = rospy.get_param("~move_velocity", 0.15)
        self.arm_behav.move_to_q(sol, velocity=move_velocity)

    def sideways_press(self, fb):
        pos, euler = PoseConv.to_pos_euler(fb.pose)
        euler = [0., 0., 0.]
        self.imkr_srv.setPose('pose_finder', PoseConv.to_pose_msg(pos, euler))
        self.imkr_srv.applyChanges()
        self.proc_fb(fb)

    def vertical_press(self, fb):
        pos, euler = PoseConv.to_pos_euler(fb.pose)
        euler = [0., np.pi/2., 0.]
        self.imkr_srv.setPose('pose_finder', PoseConv.to_pose_msg(pos, euler))
        self.imkr_srv.applyChanges()
        self.proc_fb(fb)


def main():
    rospy.init_node('pose_finder')
    pf = PoseFinder()
    rospy.spin()


if __name__ == '__main__':
    main()
