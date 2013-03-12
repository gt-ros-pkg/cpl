#! /usr/bin/python

import numpy as np
import yaml
import sys
import cv

import roslib
roslib.load_manifest('camera_calibration')
roslib.load_manifest('hrl_geom')
roslib.load_manifest('tf')

import rospy
import rosbag
import tf
from sensor_msgs.msg import Image, PointCloud2, CameraInfo

from hrl_geom.pose_converter import PoseConv
from cv_bridge import CvBridge, CvBridgeError
from camera_calibration.calibrator import ChessboardInfo, Calibrator
from point_cloud import read_points, create_cloud, create_cloud_xyz32

class DataListener(object):
    def __init__(self, is_kinect, bridge, calib):
        self.is_kinect = is_kinect
        self.cur_img = None
        self.cur_pc = None
        self.cam_info = None
        self.cur_corners = None
        self.bridge = bridge
        self.calib = calib
        self.cam_sub = rospy.Subscriber("/camera", Image, self.sub_img)
        if self.is_kinect:
            self.pc_sub = rospy.Subscriber("/pc", PointCloud2, self.sub_pc)
        else:
            self.cam_info_sub = rospy.Subscriber("/camera_info", CameraInfo, self.sub_info)
        self.vis_pub = rospy.Publisher("/cb_img_raw", Image)
        print "Waiting for image/PC"
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.cur_img is not None:
                if self.is_kinect:
                    if self.cur_pc is not None:
                        break
                else:
                    if self.cam_info is not None:
                        break
            r.sleep()
        print "Received image/PC"

    def sub_img(self, img):
        self.cur_img = img
        cv_img = self.bridge.imgmsg_to_cv(img, "rgb8")
        has_corners, cur_corners, chess = self.calib.get_corners(cv_img)
        if not has_corners:
            cur_corners = None
            return
        self.cur_corners = cur_corners
        cv.DrawChessboardCorners(cv_img, 
                                 (self.calib._boards[0].n_cols, self.calib._boards[0].n_rows), 
                                 self.cur_corners, has_corners)
        new_img = self.bridge.cv_to_imgmsg(cv_img, "rgb8")
        new_img.header = img.header
        self.vis_pub.publish(new_img)

    def sub_pc(self, pc):
        self.cur_pc = pc

    def sub_info(self, cam_info):
        self.cam_info = cam_info

    def wait_for_new(self, timeout):
        last_img_id = self.cur_img.header.seq
        if self.is_kinect:
            last_other_id = self.cur_pc.header.seq
        else:
            last_other_id = self.cam_info.header.seq
        r = rospy.Rate(10)
        start_time = rospy.get_time()
        while not rospy.is_shutdown():
            if rospy.get_time() - start_time > timeout:
                print "Timed out"
                return None
            if last_img_id != self.cur_img.header.seq:
                if self.is_kinect:
                    cur_id = self.cur_pc.header.seq
                else:
                    cur_id = self.cam_info.header.seq
                if last_other_id != cur_id:
                    return self.cur_corners
            r.sleep()

def main():
    if len(sys.argv) < 3:
        print 'grab_cbs_auto cb_config.yaml output_bag.bag'
        return
    rospy.init_node("grab_cbs")

    f = file(sys.argv[1], 'r')
    cb_config = yaml.safe_load(f.read())
    print cb_config
    f.close()
    is_kinect = rospy.get_param("/grab_cbs/is_kinect", True)

    # load cb stuff
    chessboard = ChessboardInfo()
    chessboard.n_cols = cb_config['cols'] # 6
    chessboard.n_rows = cb_config['rows'] # 7
    chessboard.dim = cb_config['dim'] # 0.0258 
    calib = Calibrator([chessboard])
    bridge = CvBridge()

    l = DataListener(is_kinect, bridge, calib)
    tf_list = tf.TransformListener()

    cb_knowns = []
    for j in range(chessboard.n_cols):
        for i in range(chessboard.n_rows):
            cb_knowns.append((chessboard.dim*i, chessboard.dim*j, 0.0))
        
    bag = rosbag.Bag(sys.argv[2], 'w')
    i = 0
    while not rospy.is_shutdown():
        if raw_input("Press enter to take CB, type 'q' to quit: ") == "q":
            break

        tries = 0
        while not rospy.is_shutdown() and tries < 3:
            corners = l.wait_for_new(5.)
            if corners is None:
                print "No corners detected"
                tries += 1
                continue
            corners_2d = np.uint32(np.round(corners)).tolist()
            corners_3d = []
            if is_kinect:
                for x,y,z in read_points(l.cur_pc, field_names=['x', 'y', 'z'], uvs=corners_2d):
                    corners_3d.append((x,y,z))
                frame_id = l.cur_pc.header
            else:
                obj_pts = cv.fromarray(np.array(cb_knowns))
                img_pts = cv.fromarray(np.array(corners))
                K = cv.fromarray(np.reshape(l.cam_info.K,(3,3)))
                D = cv.fromarray(np.array([l.cam_info.D]))
                R_vec = cv.fromarray(np.zeros((3,1)))
                t = cv.fromarray(np.zeros((3,1)))
                cv.FindExtrinsicCameraParams2(obj_pts, img_pts, K, D, R_vec, t)
                R_mat = cv.fromarray(np.zeros((3,3)))
                cv.Rodrigues2(R_vec, R_mat)
                T = PoseConv.to_homo_mat(np.mat(np.asarray(t)).T.A.tolist(), 
                                         np.mat(np.asarray(R_mat)).A.tolist())
                cb_knowns_mat = np.vstack((np.mat(cb_knowns).T, np.mat(np.ones((1, len(cb_knowns))))))
                corners_3d = np.array((T * cb_knowns_mat)[:3,:].T)
                frame_id = l.cur_img.header
            print corners_3d
            if np.any(np.isnan(corners_3d)):
                print "Pointcloud malformed"
                tries += 1
                continue
            now = rospy.Time.now()
            corners_pc = create_cloud_xyz32(frame_id, corners_3d)
            try:
                pose = tf_list.lookupTransform('/base_link', '/ee_link', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print "TF screwed up..."
                continue
            bag.write("/pose", PoseConv.to_pose_stamped_msg('/base_link', pose), now)
            bag.write("/pc", corners_pc, now)
            print "Wrote pose/CB to bag file"
            break
        i += 1
    bag.close()

if __name__ == "__main__":
    main()
