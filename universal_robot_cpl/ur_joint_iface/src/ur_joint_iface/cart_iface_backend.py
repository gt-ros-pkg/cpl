#! /usr/bin/python

import numpy as np
import sys
from PyQt4 import QtCore, QtGui, uic
import functools

import roslib
roslib.load_manifest("rospy")
roslib.load_manifest("ur_demos")
import rospy

from joint_iface_gui import Ui_Frame as QTJointIFaceGUIFrame
from ur_demos.ur_robot import create_ur_robot

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

MONITOR_RATE = 1000./125.
class JointIFaceGUIFrame(QtGui.QFrame):
    def __init__(self):
        super(JointIFaceGUIFrame, self).__init__()
        self.button_down = None
        self.joint_ctrl = SingleJointController()
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
                self.joint_ctrl.start_moving(BUTTONS[self.button_down][0],
                                             BUTTONS[self.button_down][1])
        self.joint_ctrl.update()

class SingleJointController(object):
    def __init__(self):
        self.moving_joint = None
        self.moving_dir = None
        self.ur_robot = create_ur_robot("base_link", "wrist_3_link")
        self.delta_x = 0.01
        self.t_f = 0.2
        self.v_f = 0.1
        self.t = 0.0

    def update(self):
        if self.moving_joint is not None:
            print "Moving:", self.moving_joint
            q = self.ur_robot.get_joint_angles()
            J = self.ur_robot.jacobian()
            dx = 0.03 * np.mat([0., 0., 1., 0., 0., 0.]).T
            qd_cmd = (J.T * dx).T.A[0]
            print qd_cmd
            #qd_cmd = np.zeros(6)
            self.ur_robot.cmd_vel(qd_cmd)
            self.t += 1.0/self.ur_robot.CONTROL_RATE

    def start_moving(self, joint_ind, direction):
        print "Down:", joint_ind, direction
        self.moving_joint = joint_ind
        self.moving_dir = direction
        p = min_jerk_accelerate(x_i=0., 
                                x_f=self.delta_x, 
                                t_f=self.t_f, v_f=self.v_f)
        self.pd = (p * np.linspace(len(p)-1,0,len(p)))[0:-1]
        self.pdd = (p * np.linspace(len(p)-1,0,len(p)) * np.linspace(len(p)-2,-1,len(p)))[0:-2]
        self.p = p
        self.t = 0.0

    def stop_moving(self):
        print "Up"
        self.moving_joint = None

def min_jerk_accelerate(x_i, x_f, t_f, v_f):
    A = np.matrix([[t_f**5,    t_f**4,    t_f**3   ], 
                   [5*t_f**4,  4*t_f**3,  3*t_f**2 ],
                   [20*t_f**3, 12*t_f**2, 6*t_f    ]])
    b = np.matrix([x_f, v_f, 0]).T
    x = np.linalg.solve(A,b)
    return np.concatenate([x.T.A[0], np.array([0., 0., x_i])])

def main():
    rospy.init_node("arm_cart_control_interface")
    app = QtGui.QApplication(sys.argv)
    frame = JointIFaceGUIFrame()
    frame.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
