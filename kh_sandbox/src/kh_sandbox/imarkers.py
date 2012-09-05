#! /usr/bin/python

import roslib
roslib.load_manifest("rospy")
roslib.load_manifest("interactive_markers")
roslib.load_manifest("visualization_msgs")

import rospy
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import InteractiveMarker, Marker, InteractiveMarkerControl

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

def ee_pose_cb(data):
    print data

def main():
    rospy.init_node("imarkers")

    imkr_srv = InteractiveMarkerServer("imarkers")

    imkr_srv.insert(six_axis_mkr("ee_pose", "/base_link", 0.25), ee_pose_cb)
    imkr_srv.applyChanges()

    rospy.spin()

if __name__ == "__main__":
    main()
