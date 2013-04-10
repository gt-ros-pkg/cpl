#! /usr/bin/python

import numpy as np
import yaml

import roslib
roslib.load_manifest("ur_cart_move")
import rospy

from bin_manager import BinManager

def main():
    np.set_printoptions(precision=4)
    rospy.init_node("bin_delivery_demo")
    f = file("bin_locs.yaml", 'r')
    ar_empty_locs = yaml.load(f)['data']
    f.close()
    arm_prefix = ""
    bm = BinManager(arm_prefix, ar_empty_locs)

    move_plan = [(2, 2),
                 (3, 3),
                 (14, 1),
                 (2, 6),
                 (11, 2),
                 (3, 7),
                 (12, 3),
                 (14, 5),
                 (13, 1),
                 (11, 8),
                 (12, 9),
                 (13, 10),
                 (14, 11)
                 ]
    reset = True
    plan_step = 0
    while not rospy.is_shutdown():
        print 'Plan step:', plan_step
        if reset:
            bm.system_reset()
            reset = False
        ar_bin, place_tag_num = move_plan[plan_step]
        if not bm.move_bin(ar_bin, place_tag_num):
            reset = True
            print 'Failed moving bin from %d to %d' % (ar_bin, place_tag_num)
        else:
            plan_step += 1

if __name__ == "__main__":
    main()
