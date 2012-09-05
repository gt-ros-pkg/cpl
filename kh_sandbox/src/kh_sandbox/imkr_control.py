#! /usr/bin/python

import numpy as np

import roslib
roslib.load_manifest("rospy")
roslib.load_manifest("interactive_markers")
roslib.load_manifest("visualization_msgs")
roslib.load_manifest("hrl_pr2_arms")

import rospy
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import InteractiveMarker, Marker, InteractiveMarkerControl
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

from hrl_pr2_arms.ep_arm_base import create_ep_arm, EPArmBase
from hrl_geom.pose_converter import PoseConv
import hrl_geom.transformations as trans

def six_axis_mkr(name, frame="/base_link", scale=0.25):
    imkr = InteractiveMarker()
    imkr.header.frame_id = frame
    imkr.name = name
    imkr.scale = scale
    imkr.controls.extend(six_axis_ctrls())
    return imkr

def six_axis_ctrls():
    ctrls = []
    for ctrl_type in ['move', 'rotate']:
        for axis in ['x', 'y', 'z']:
            ctrl = InteractiveMarkerControl()
            exec("ctrl.interaction_mode = InteractiveMarkerControl.%s_AXIS" % ctrl_type.upper())
            ctrl.name = "%s_%s" % (ctrl_type, axis)
            ctrl.orientation.w = 1
            ctrl.always_visible = True
            exec("ctrl.orientation.%s = 1" % axis)
            ctrls.append(ctrl)
    return ctrls

class IMarkerController(object):
    def __init__(self):
        self.arm = create_ep_arm('r', EPArmBase, base_link="base_link", end_link="r_ee_link")
        self.js_pub = rospy.Publisher("/joint_states", JointState)
        rospy.sleep(1)
        self.pub_js([0.]*8)
        self.q_cur = np.array([0.]*8)
        self.pos_last = None
        self.quat_last = None
        self.pos_goal = [0., 0., 0.]
        self.quat_goal = [0., 0., 0., 1.]

        self.imkr_srv = InteractiveMarkerServer("imarkers")
        self.imkr_srv.insert(six_axis_mkr("ee_pose", "/base_link", 0.25), self.ee_pose_cb)
        self.imkr_srv.applyChanges()
        #pose_pub = rospy.Publisher("/cur_ee_pose", PoseStamped)
        #while not rospy.is_shutdown():
        #    pose_pub.publish(PoseConv.to_pose_stamped_msg("base_link", 
        #                                                  arm.get_end_effector_pose()))

    def update_arm(self):
        pos_cur, quat_cur = PoseConv.to_pos_quat(self.arm.get_end_effector_pose())
        if pos_cur is None:
            return
        if self.quat_last is None:
            self.pos_last = np.array(pos_cur)
            self.quat_last = np.array(quat_cur)
        quat_diff = trans.quaternion_multiply(trans.quaternion_inverse(quat_cur), 
                                              self.quat_goal)
        euler_diff = np.array(trans.euler_from_quaternion(quat_diff))
        #euler_diff *= 0.
        pos_diff = np.array(self.pos_goal) - pos_cur
        #pos_diff *= 0.
        x_diff = np.mat([pos_diff[0], pos_diff[1], pos_diff[2], 
                         euler_diff[0], euler_diff[1], euler_diff[2]]).T
        xd = np.array(pos_cur) - self.pos_last
        rd_diff = trans.quaternion_multiply(trans.quaternion_inverse(self.quat_last), 
                                              quat_cur)
        euler_rd = np.array(trans.euler_from_quaternion(quat_diff))
        xd_diff = [xd[0], xd[1], xd[2], euler_rd[0], euler_rd[1], euler_rd[2]]
        pP, pD = 0.02, 0.09
        rP, rD = 0.0008, 0.09
        tau = self.arm.jacobian().T * (np.mat(np.diag([pP]*3+[rP]*3)) * x_diff + 
                                       np.mat(np.diag([rP]*3+[rD]*3)) * np.mat(xd_diff).T)
        weights = [0.08, 0.08, 3, 3, 3, 1, 1, 1]
        #weights = [1]*8
        max_diff = 0.001
        control = np.clip(np.array(weights) * tau.T.A[0], -max_diff, max_diff)
        self.q_cur += control #np.where(np.fabs(control) < 0.000007, 0., control)
        #print "%4f %4f %4f" % (np.linalg.norm(x_diff * P), np.linalg.norm(xd * D),
        #                       np.linalg.norm(x_diff))
        self.pos_last = pos_cur
        self.quat_last = quat_cur
        self.pub_js([0.]*8)
        #self.pub_js(self.q_cur)

    def pub_js(self, q):
        if type(q) is np.array:
            q = q.tolist()
        js = JointState()
        js.header.frame_id = "/base_link"
        js.header.stamp = rospy.Time.now()
        js.name = [name.encode('utf-8') for name in self.arm.get_joint_names()]
        js.position = q
        self.js_pub.publish(js)

    def ee_pose_cb(self, feedback):
        self.pos_goal, self.quat_goal = PoseConv.to_pos_quat(feedback.pose)

def main():
    rospy.init_node("imarkers")

    imkr_ctrler = IMarkerController()

    r = rospy.Rate(1000)
    while not rospy.is_shutdown():
        imkr_ctrler.update_arm()
        r.sleep()

    rospy.spin()

if __name__ == "__main__":
    main()
