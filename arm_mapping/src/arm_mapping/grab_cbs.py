#! /usr/bin/python

import numpy as np

import roslib
roslib.load_manifest("pykdl_utils")
roslib.load_manifest('camera_calibration')

import rospy
import rosbag
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, PointCloud2

from pykdl_utils.joint_kinematics import create_joint_kin
from hrl_geom.pose_converter import PoseConv
from cv_bridge import CvBridge, CvBridgeError
from camera_calibration.calibrator import ChessboardInfo, Calibrator
from point_cloud import read_points, create_cloud, create_cloud_xyz32

class DataListener(object):
    def __init__(self):
        self.cam_sub = rospy.Subscriber("/camera/rgb/image_rect_color", Image, self.sub_img)
        self.pc_sub = rospy.Subscriber("/camera/depth_registered/points", PointCloud2, self.sub_pc)
        self.kin = create_joint_kin("base_link", "ee_link")

    def sub_img(self, img):
        self.cur_img = img

    def sub_pc(self, pc):
        self.cur_pc = pc

def main():
    rospy.init_node("grab_data")

    chessboard = ChessboardInfo()
    chessboard.n_cols = 6
    chessboard.n_rows = 7
    chessboard.dim = 0.0258 
    calib = Calibrator([chessboard])
    bridge = CvBridge()
        
    l = DataListener()
    bag = rosbag.Bag('data.bag', 'w')
    while not rospy.is_shutdown():
        if raw_input() == 'q':
            break
        print l.cur_img.header.frame_id
        print l.cur_pc.header.frame_id
        cv_img = bridge.imgmsg_to_cv(l.cur_img, "rgb8")
        has_corners, corners_2d, chess = calib.get_corners(cv_img)
        if not has_corners:
            print "No corners detected"
            continue
        corners_2d = np.uint32(np.round(corners_2d)).tolist()
        #for i in range(7):
        #    for j in range(6):
        #        print corners_2d[j+i*6],
        #    print
        # 6 is height
        corners_3d = []
        for x,y,z in read_points(l.cur_pc, field_names=['x', 'y', 'z'], uvs=corners_2d):
            corners_3d.append((x,y,z))
        pose = l.kin.forward()
        print pose
        now = rospy.Time.now()
        corners_pc = create_cloud_xyz32(l.cur_pc.header, corners_3d)
        bag.write("/pose", PoseConv.to_pose_stamped_msg(pose), now)
        bag.write("/pc", corners_pc, now)
    bag.close()

if __name__ == "__main__":
    main()
