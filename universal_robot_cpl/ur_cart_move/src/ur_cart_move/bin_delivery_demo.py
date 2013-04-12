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

    move_plan = [(2, 1),
                 (3, 2),
                 (14, 0),
                 (2, 5),
                 (11, 1),
                 (3, 6),
                 (12, 2),
                 (14, 4),
                 (13, 0),
                 (11, 7),
                 (12, 8),
                 (13, 9),
                 (14, 10)
                 ]
    reset = True
    plan_step = 0
    while not rospy.is_shutdown():
        print 'Plan step:', plan_step
        if reset:
            raw_input("Move to home")
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
