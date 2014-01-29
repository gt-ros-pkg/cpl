#! /usr/bin/python

import roslib
roslib.load_manifest("pykdl_utils")
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, PointCloud2
from pykdl_utils.joint_kinematics import create_joint_kin
from hrl_geom.pose_converter import PoseConv
import rospy
import rosbag
import tf

RATE = 10

class DataListener(object):
    def __init__(self):
        self.cam_sub = rospy.Subscriber("/camera/rgb/image_color", Image, self.sub_img)
        self.pc_sub = rospy.Subscriber("/camera/rgb/points", PointCloud2, self.sub_pc)
        self.tf_list = tf.TransformListener()
        self.cur_img = None
        self.cur_pc = None
        self.img_ind = 0
        self.pc_ind = 0

    def sub_img(self, img):
        self.cur_img = img
        self.img_ind += 1
        self.img_ind += 1

    def sub_pc(self, pc):
        self.cur_pc = pc
        self.pc_ind += 1

def main():
    rospy.init_node("grab_data")
    l = DataListener()
    bag = rosbag.Bag('data_conv.bag', 'w')
    r = rospy.Rate(RATE)
    done_first = False
    last_img_ind = -1
    last_pc_ind = -1
    while not rospy.is_shutdown():
        if l.cur_img is None or l.cur_pc is None:
            r.sleep()
            continue
        else:
            if not done_first:
                print "got topics"
                done_first = True
        if l.img_ind == last_img_ind:
            r.sleep()
            continue
        if l.pc_ind == last_pc_ind:
            r.sleep()
            continue
        pose = PoseConv.to_pose_stamped_msg(
                    l.tf_list.lookupTransform("/world", "/kinect", rospy.Time(0)))
        now = rospy.Time.now()
        bag.write("/image", l.cur_img, now)
        bag.write("/pc", l.cur_pc, now)
        bag.write("/pose", PoseConv.to_pose_stamped_msg(pose), now)
        last_img_ind = l.img_ind
        last_pc_ind = l.pc_ind
        r.sleep()
    bag.close()

if __name__ == "__main__":
    main()
