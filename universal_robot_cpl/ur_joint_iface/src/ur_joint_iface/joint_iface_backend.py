#! /usr/bin/python

import numpy as np
import sys
from PyQt4 import QtCore, QtGui, uic
import functools

import roslib
roslib.load_manifest("rospy")
roslib.load_manifest("ur_joint_iface")
roslib.load_manifest("hrl_geom")
import rospy

from joint_iface_gui import Ui_Frame as QTJointIFaceGUIFrame
from ur_cart_move.ur_cart_move import load_ur_robot
from hrl_geom.pose_converter import PoseConv

BUTTONS = {
    "shoulder_pan_left" : [0, -1.0],
    "shoulder_pan_right" : [0, 1.0],
    "shoulder_lift_up" : [1, -1.0],
    "shoulder_lift_down" : [1, 1.0],
    "elbow_up" : [2, -1.0],
    "elbow_down" : [2, 1.0],
    "wrist_1_up" : [3, -1.0],
    "wrist_1_down" : [3, 1.0],
    "wrist_2_left" : [4, -1.0],
    "wrist_2_right" : [4, 1.0],
    "wrist_3_up" : [5, -1.0],
    "wrist_3_down" : [5, 1.0]}

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]

MONITOR_RATE = 1000./125.
class JointIFaceGUIFrame(QtGui.QFrame):
    def __init__(self, clipboard):
        super(JointIFaceGUIFrame, self).__init__()
        self.button_down = None
        self.joint_ctrl = SingleJointController()
        self.clipboard = clipboard
        self.init_ui()

    def init_ui(self):
        self.ui = QTJointIFaceGUIFrame()
        self.ui.setupUi(self)
        self.monitor_timer = QtCore.QTimer(self)
        QtCore.QObject.connect(self.monitor_timer, QtCore.SIGNAL("timeout()"), self.monitor_cb)
        self.monitor_timer.start(MONITOR_RATE)

    def monitor_cb(self):
        last_button_down = self.button_down
        self.button_down = None
        q = self.joint_ctrl.arm.get_q().tolist()
        for i, joint in enumerate(JOINTS):
            exec("self.ui.%s_num.setPlainText(QtCore.QString.fromAscii('%f'))" % (joint, q[i]))
            if q[i] >= 0:
                exec("self.ui.%s_prog_pos.setValue(%d)" % (joint, int(q[i]/(2.*np.pi)*100.)))
                exec("self.ui.%s_prog_neg.setValue(%d)" % (joint, 0))
            if q[i] < 0:
                exec("self.ui.%s_prog_pos.setValue(%d)" % (joint, 0))
                exec("self.ui.%s_prog_neg.setValue(%d)" % (joint, int(-q[i]/(2.*np.pi)*100.)))
        if self.ui.copy_joint_config.isDown():
            q_str = ", ".join([str(q[i]) for i in range(6)])
            self.clipboard.setText(QtCore.QString.fromAscii(q_str))
        if self.ui.copy_ee_pose.isDown():
            ee_pose = PoseConv.to_pos_quat(self.joint_ctrl.kin.forward(q))
            ee_pose = ", ".join([str(x) for x in (ee_pose[0] + ee_pose[1])])
            self.clipboard.setText(QtCore.QString.fromAscii(ee_pose))
        for button in BUTTONS:
            exec("is_down = self.ui.%s.isDown()" % button)
            if is_down:
                self.button_down = button
                break
        if self.button_down != last_button_down:
            if last_button_down is not None:
                # do button up on last
                self.joint_ctrl.stop_moving()

            if self.button_down is not None:
                # do button down on current
                joint_ind = BUTTONS[self.button_down][0]
                moving_dir = BUTTONS[self.button_down][1]
                exec("vel = self.ui.%s_vel.value()" % JOINTS[joint_ind])
                self.joint_ctrl.start_moving(joint_ind,
                                             moving_dir, vel)
        self.joint_ctrl.update()

class SingleJointController(object):
    def __init__(self):
        self.moving_joint = None
        self.moving_dir = None
        self.delta_x = 0.01
        self.vel_f = 0.2
        self.t_f = 0.5 
        self.acc = self.vel_f / self.t_f
        self.pos_f = 0.5*self.acc*(self.t_f**2)
        self.t = 0.0
        prefix = rospy.get_param("~prefix", "")
        self.arm, self.kin, self.arm_behav = load_ur_robot(topic_prefix=prefix)
        self.q_init = None

    def update(self):
        if self.moving_joint is not None:
            t, t_f, vel_f, acc, pos_f = self.t, self.t_f, self.vel_f, self.acc, self.pos_f
            q_cmd = self.q_init.copy()
            qd_cmd = np.zeros(6)
            qdd_cmd = np.zeros(6)
            if t < t_f:
                q_cmd[self.moving_joint] += self.moving_dir*(0.5*acc*(t**2))
                qd_cmd[self.moving_joint] += self.moving_dir*(acc*t)
                qdd_cmd[self.moving_joint] += self.moving_dir*(acc)
                self.arm.cmd_pos_vel_acc(q_cmd,qd_cmd,qdd_cmd)
            else:
                q_cmd[self.moving_joint] += self.moving_dir*(vel_f*(t-t_f) + pos_f)
                qd_cmd[self.moving_joint] = self.moving_dir*vel_f
                self.arm.cmd_pos_vel_acc(q_cmd, qd_cmd, qdd_cmd)
            self.t += 1.0/self.arm.CONTROL_RATE

    def start_moving(self, joint_ind, direction, vel):
        self.arm.unlock_security_stop()
        self.moving_joint = joint_ind
        self.moving_dir = direction
        self.t = 0.0
        self.vel_f = vel
        self.acc = self.vel_f / self.t_f
        self.pos_f = 0.5*self.acc*(self.t_f**2)
        self.q_init = np.array(self.arm.get_q())

    def stop_moving(self):
        self.moving_joint = None

def main():
    rospy.init_node("arm_cart_control_interface")
    app = QtGui.QApplication(sys.argv)
    cb = app.clipboard()
    frame = JointIFaceGUIFrame(cb)
    frame.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
