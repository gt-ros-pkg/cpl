#! /usr/bin/python

from threading import Lock
import numpy as np
from openravepy import *
from scipy.interpolate import PiecewisePolynomial
import sys
import wx
import xml.dom.minidom
from sensor_msgs.msg import JointState
from math import pi
from threading import Thread

import roslib
roslib.load_manifest("ur_cart_move")

import rospy
import roslaunch.substitution_args
from sensor_msgs.msg import JointState

from hrl_geom.pose_converter import PoseConv
from hrl_geom.transformations import rotation_from_matrix as mat_to_ang_axis_point
from hrl_geom.transformations import rotation_matrix as ang_axis_point_to_mat
from hrl_geom.transformations import euler_matrix
from ur_controller_manager.msg import URJointCommand, URModeStates, URJointStates, URModeCommand

def get_param(name, value=None):
    private = "~%s" % name
    if rospy.has_param(private):
        return rospy.get_param(private)
    elif rospy.has_param(name):
        return rospy.get_param(name)
    else:
        return value

RANGE = 10000

class ArmSimulator(object):
    CONTROL_RATE = 125
    JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                   'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    def __init__(self, q_init=[0.]*6):
        self.q_des = q_init
        self.qd_des = [0.]*6
        self.qdd_des = [0.]*6
        self.free_joints = {}
        for joint_name in ArmSimulator.JOINT_NAMES:
            self.free_joints[joint_name] = {'min':-2.*np.pi, 'max':2.*np.pi, 'zero':0., 'value':0. }
        self.joint_states = self.make_joint_state()

        use_gui = get_param("use_gui", True)

        if use_gui:
            app = wx.App()
            self.gui = JointStatePublisherGui("UR Joint State Publisher", self)
            self.gui.Show()
            Thread(target=app.MainLoop).start()
        else:
            self.gui = None

        self.js_pub = rospy.Publisher('/joint_states', JointState)
        self.ur_js_pub = rospy.Publisher('/ur_joint_states', URJointStates)
        self.ur_ms_pub = rospy.Publisher('/ur_mode_states', URModeStates)
        rospy.Subscriber("/ur_joint_command", URJointCommand, self._ur_joint_command_cb)
        rospy.Subscriber("/ur_mode_command", URModeCommand, self._ur_mode_command_cb)

    def make_joint_state(self, q=[0.]*6, qd=[0.]*6, effort=[0.]*6):
        js = JointState()
        js.header.stamp = rospy.Time.now()
        for i, joint_name in enumerate(ArmSimulator.JOINT_NAMES):
            js.name.append(joint_name)
            js.position.append(q[i])
            js.velocity.append(qd[i])
            js.effort.append(effort[i])
        return js

    def make_ur_mode_state(self):
        ms = URModeStates()
        ms.header.stamp = rospy.Time.now()
        ms.robot_mode_id = 0
        ms.is_power_on_robot = True
        return ms

    def _ur_joint_command_cb(self, cmd):
        self.q_des = list(cmd.q_des)
        self.qd_des = list(cmd.qd_des)
        self.qdd_des = list(cmd.qdd_des)
        for i, name in enumerate(ArmSimulator.JOINT_NAMES):
            self.free_joints[name]['value'] = self.q_des[i]
        if self.gui is not None:
            wx.PostEvent(self.gui, ResultEvent(None))
            #self.gui.update_sliders()

    def _ur_mode_command_cb(self, msg):
        pass

    def control_loop(self):
        r = rospy.Rate(ArmSimulator.CONTROL_RATE)
        while not rospy.is_shutdown():
            for i, name in enumerate(ArmSimulator.JOINT_NAMES):
                self.q_des[i] = self.free_joints[name]['value']
            self.js_pub.publish(self.make_joint_state(self.q_des, self.qd_des))
            ur_js = URJointStates()
            ur_js.q_act, ur_js.qd_act = self.q_des, self.qd_des
            self.ur_js_pub.publish(ur_js)
            self.ur_ms_pub.publish(self.make_ur_mode_state())
            r.sleep()

EVT_RESULT_ID = wx.NewId()
def EVT_RESULT(win, func):
    win.Connect(-1, -1, EVT_RESULT_ID, func)
class ResultEvent(wx.PyEvent):
    def __init__(self, data):
        wx.PyEvent.__init__(self)
        self.SetEventType(EVT_RESULT_ID)
        self.data = data

# Much of this code borrowed from David Lu's joint_state_publisher package
class JointStatePublisherGui(wx.Frame):
    def __init__(self, title, arm_sim):
        wx.Frame.__init__(self, None, -1, title, (-1, -1));
        self.lock = Lock()
        self.arm_sim = arm_sim
        self.joint_map = {}
        panel = wx.Panel(self, wx.ID_ANY);
        box = wx.BoxSizer(wx.VERTICAL)
        font = wx.Font(9, wx.SWISS, wx.NORMAL, wx.BOLD)
        
        ### Sliders ###
        for name in self.arm_sim.JOINT_NAMES:
            joint = self.arm_sim.free_joints[name]

            if joint['min'] == joint['max']:
                continue

            row = wx.GridSizer(1,2)
            label = wx.StaticText(panel, -1, name)
            label.SetFont(font)
            row.Add(label, 1, wx.ALIGN_CENTER_VERTICAL)

            display = wx.TextCtrl (panel, value=str(0), 
                        style=wx.TE_READONLY | wx.ALIGN_RIGHT)

            row.Add(display, flag= wx.ALIGN_RIGHT| wx.ALIGN_CENTER_VERTICAL)
            box.Add(row, 1, wx.EXPAND)
            slider = wx.Slider(panel, -1, RANGE/2, 0, RANGE, 
                        style= wx.SL_AUTOTICKS | wx.SL_HORIZONTAL)
            slider.SetFont(font)
            box.Add(slider, 1, wx.EXPAND)

            self.joint_map[name] = {'slidervalue':0, 'display':display, 
                                    'slider':slider, 'joint':joint}

        ### Buttons ###
        self.ctrbutton = wx.Button(panel, 1, 'Center')
        self.Bind(wx.EVT_SLIDER, self.sliderUpdate)
        
        wx.EVT_BUTTON(self, 1, self.center_event)

        box.Add(self.ctrbutton, 0, wx.EXPAND)
        
        panel.SetSizer(box)
        self.center()
        box.Fit(self)
        self.update_values()
        EVT_RESULT(self, self._update_sliders)


    def update_values(self):
        for (name,joint_info) in self.joint_map.items():
            purevalue = joint_info['slidervalue']
            joint = joint_info['joint']
            value = self.sliderToValue(purevalue, joint)
            joint['value'] = value
        self.update_sliders()

    def _update_sliders(self, msg):
        self.update_sliders()

    def update_sliders(self):
        with self.lock:
            for (name,joint_info) in self.joint_map.items():
                joint = joint_info['joint']
                joint_info['slidervalue'] = self.valueToSlider(joint['value'], joint)
                joint_info['slider'].SetValue(joint_info['slidervalue'])
                joint_info['display'].SetValue("%.2f"%joint['value'])

    def center_event(self, event):
        self.center()

    def center(self):
        rospy.loginfo("Centering")
        for (name,joint_info) in self.joint_map.items():
            joint = joint_info['joint']
            joint_info['slidervalue'] = self.valueToSlider(joint['zero'], joint)
        self.update_values()

    def sliderUpdate(self, event):
        with self.lock:
            for (name,joint_info) in self.joint_map.items():
                joint_info['slidervalue'] = joint_info['slider'].GetValue()
        self.update_values()

    def valueToSlider(self, value, joint):
        return (value - joint['min']) * float(RANGE) / (joint['max'] - joint['min'])
        
    def sliderToValue(self, slider, joint):
        pctvalue = slider / float(RANGE)
        return joint['min'] + (joint['max']-joint['min']) * pctvalue

def main():
    rospy.init_node("ur_arm_sim")
    arm_sim = ArmSimulator()
    arm_sim.control_loop()

if __name__ == "__main__":
    main()
