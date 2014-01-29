#! /usr/bin/python

import roslib
roslib.load_manifest("pykdl_utils")
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, PointCloud2
from pykdl_utils.joint_kinematics import create_joint_kin
from hrl_geom.pose_converter import PoseConv
import rospy
import rosbag

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
    l = DataListener()
    bag = rosbag.Bag('data.bag', 'w')
    while not rospy.is_shutdown():
        if raw_input() == 'q':
            break
        print l.cur_img.header.frame_id
        print l.cur_pc.header.frame_id
        pose = l.kin.forward()
        print pose
        now = rospy.Time.now()
        bag.write("/image", l.cur_img, now)
        bag.write("/pc", l.cur_pc, now)
        bag.write("/pose", PoseConv.to_pose_stamped_msg(pose), now)
    bag.close()

if __name__ == "__main__":
    main()
