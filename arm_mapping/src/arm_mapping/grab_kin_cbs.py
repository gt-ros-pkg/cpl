#! /usr/bin/python

import numpy as np

import roslib
roslib.load_manifest("pykdl_utils")
roslib.load_manifest('camera_calibration')

import rospy
import rosbag
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, PointCloud2

from hrl_geom.pose_converter import PoseConv
from cv_bridge import CvBridge, CvBridgeError
from camera_calibration.calibrator import ChessboardInfo, Calibrator
from point_cloud import read_points, create_cloud, create_cloud_xyz32

class DataListener(object):
    def __init__(self, kinect_name):
        self.cam_sub = rospy.Subscriber("/%s/rgb/image_raw" % kinect_name, Image, self.sub_img)
        self.pc_sub = rospy.Subscriber("/%s/depth_registered/points" % kinect_name, PointCloud2, self.sub_pc)

    def sub_img(self, img):
        self.cur_img = img

    def sub_pc(self, pc):
        self.cur_pc = pc

def main():
    rospy.init_node("grab_kin_data")

    chessboard = ChessboardInfo()
    chessboard.n_cols = 6
    chessboard.n_rows = 7
    chessboard.dim = 0.0258 
    calib = Calibrator([chessboard])
    bridge = CvBridge()

    num_listeners = 2
    listeners = [DataListener("kinect%d"%i) for i in range(1,num_listeners+1)]
    bag = rosbag.Bag('data.bag', 'w')
    while not rospy.is_shutdown():
        if raw_input() == 'q':
            break
        corners_pcs = []
        num_corner_detects = 0
        for i in range(num_listeners):
            cv_img = bridge.imgmsg_to_cv(listeners[i].cur_img, "rgb8")
            has_corners, corners_2d, chess = calib.get_corners(cv_img)
            if not has_corners:
                corners_pc = create_cloud_xyz32(listeners[i].cur_pc.header, [])
                corners_pcs.append(corners_pc)
                continue
            corners_2d = np.uint32(np.round(corners_2d)).tolist()
            #for i in range(7):
            #    for j in range(6):
            #        print corners_2d[j+i*6],
            #    print
            # 6 is height
            corners_3d = []
            for x,y,z in read_points(listeners[i].cur_pc, field_names=['x', 'y', 'z'], uvs=corners_2d):
                corners_3d.append((x,y,z))
            now = rospy.Time.now()
            corners_pc = create_cloud_xyz32(listeners[i].cur_pc.header, corners_3d)
            corners_pcs.append(corners_pc)
            num_corner_detects += 1
        if num_corner_detects > 1:
            bag.write("/pc", corners_pc, now)
    bag.close()

if __name__ == "__main__":
    main()
