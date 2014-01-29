#! /usr/bin/python

import numpy as np
import yaml
import sys
import cv

import roslib
roslib.load_manifest("ur_cart_move")
roslib.load_manifest('camera_calibration')

import rospy
import rosbag
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, PointCloud2
import roslaunch.substitution_args

from ur_cart_move.ur_cart_move import RAVEKinematics, ArmInterface, ArmBehaviors
from hrl_geom.pose_converter import PoseConv
from cv_bridge import CvBridge, CvBridgeError
from camera_calibration.calibrator import ChessboardInfo, Calibrator
from point_cloud import read_points, create_cloud, create_cloud_xyz32

class DataListener(object):
    def __init__(self, kinect_name, bridge, calib):
        self.cur_img = None
        self.cur_pc = None
        self.cur_corners = None
        self.bridge = bridge
        self.calib = calib
        self.cam_sub = rospy.Subscriber("/%s/rgb/image_rect_color"%kinect_name, 
                                        Image, self.sub_img)
        self.pc_sub = rospy.Subscriber("/%s/depth_registered/points"%kinect_name, 
                                       PointCloud2, self.sub_pc)
        self.vis_pub = rospy.Publisher("/cb_img_raw", Image)
        print "Waiting for image/PC"
        r = rospy.Rate(10)
        while not rospy.is_shutdown() and (self.cur_img is None or self.cur_pc is None):
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

    def wait_for_new(self, timeout):
        last_img_id = self.cur_img.header.seq
        last_pc_id = self.cur_pc.header.seq
        r = rospy.Rate(10)
        start_time = rospy.get_time()
        while not rospy.is_shutdown():
            if rospy.get_time() - start_time > timeout:
                print "Timed out"
                return None
            if (last_img_id != self.cur_img.header.seq and
                last_pc_id != self.cur_pc.header.seq):
                return self.cur_corners
            r.sleep()

def main():
    if len(sys.argv) < 4:
        print 'grab_cbs_auto cb_config.yaml output_bag.bag kinect_name [joint_configs.yaml]'
        return
    rospy.init_node("grab_cbs_auto")

    f = file(sys.argv[1], 'r')
    cb_config = yaml.safe_load(f.read())
    print cb_config
    f.close()
    if len(sys.argv) > 4:
        f = file(sys.argv[4], 'r')
        qs = yaml.safe_load(f.read())
        print qs
        f.close()
    else:
        qs = None

    # load cb stuff
    chessboard = ChessboardInfo()
    chessboard.n_cols = cb_config['cols'] # 6
    chessboard.n_rows = cb_config['rows'] # 7
    chessboard.dim = cb_config['dim'] # 0.0258 
    calib = Calibrator([chessboard])
    bridge = CvBridge()

    l = DataListener(sys.argv[3], bridge, calib)

    # load arm stuff
    robot_descr = roslaunch.substitution_args.resolve_args('$(find ur10_description)/ur10_robot.dae')
    arm = ArmInterface(timeout=0.)
    kin = RAVEKinematics(robot_descr)
    if not arm.wait_for_states(timeout=5.):
        print 'arm not connected!'
        return
    arm_behav = ArmBehaviors(arm, kin)
        
    bag = rosbag.Bag(sys.argv[2], 'w')
    i = 0
    while not rospy.is_shutdown():
        if qs is not None:
            if i >= len(qs):
                break
            q = qs[i]
            restarts = 2
            while restarts >= 0:
                arm.unlock_security_stop()
                if arm_behav.move_to_q(q, velocity=0.07):
                    break
                restarts -= 1
                rospy.sleep(0.3)
            rospy.sleep(1.5)
        else:
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
            #for i in range(7):
            #    for j in range(6):
            #        print corners_2d[j+i*6],
            #    print
            # 6 is height
            corners_3d = []
            for x,y,z in read_points(l.cur_pc, field_names=['x', 'y', 'z'], uvs=corners_2d):
                corners_3d.append((x,y,z))
            print corners_3d
            if np.any(np.isnan(corners_3d)):
                print "Pointcloud malformed"
                tries += 1
                continue
            pose = kin.forward(arm.get_q())
            print pose
            now = rospy.Time.now()
            corners_pc = create_cloud_xyz32(l.cur_pc.header, corners_3d)
            bag.write("/pose", PoseConv.to_pose_stamped_msg('/base_link', pose), now)
            bag.write("/pc", corners_pc, now)
            break
        i += 1
    bag.close()

if __name__ == "__main__":
    main()
