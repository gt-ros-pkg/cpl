#! /usr/bin/python

import roslib
roslib.load_manifest("rospy")
roslib.load_manifest("interactive_markers")
roslib.load_manifest("visualization_msgs")

import rospy
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import InteractiveMarker, Marker, InteractiveMarkerControl

def main():
    rospy.init_node("imarkers")

    imkr_srv = InteractiveMarkerServer("imarkers")

    imkr = InteractiveMarker()
    imkr.header.frame_id = "/base_link"
    imkr.name = "ee_pose"
    imkr.description = "ee_pose"

    sph_mkr = Marker()
    sph_mkr.type = Marker.SPHERE
    sph_mkr.scale.x = 0.1
    sph_mkr.scale.y = 0.1
    sph_mkr.scale.z = 0.1
    sph_mkr.color.r = 1.0
    sph_mkr.color.g = 0.0
    sph_mkr.color.b = 1.0
    sph_mkr.color.a = 1.0

    box_ctrl = InteractiveMarkerControl()
    box_ctrl.always_visible = True
    box_ctrl.markers.append(sph_mkr)
    imkr.controls.append(box_ctrl)

    rot_ctrl = InteractiveMarkerControl()
    rot_ctrl.name = "move_x"
    rot_ctrl.always_visible = True
    rot_ctrl.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    imkr.controls.append(rot_ctrl)

    imkr_srv.insert(imkr)
    imkr_srv.applyChanges()

    rospy.spin()

if __name__ == "__main__":
    main()
