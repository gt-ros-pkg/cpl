#! /usr/bin/python

import numpy as np
import yaml

import roslib
roslib.load_manifest("ur_cart_move")
import rospy

from ur_cart_move.bin_manager import BinManager
from greedy_bin_plan import GreedyBinPlanner

def main():
    np.set_printoptions(precision=4)
    rospy.init_node("bin_delivery_demo")
    f = file("bin_locs.yaml", 'r')
    ar_empty_locs = yaml.load(f)['data']
    f.close()
    arm_prefix = "/sim1"
    #arm_prefix = ""
    bm = BinManager(arm_prefix, ar_empty_locs)
    gbp = GreedyBinPlanner()

    reset = True
    plan_step = 0
    while not rospy.is_shutdown():
        if reset:
            raw_input("Move to home")
            bm.system_reset()
            reset = False

        slot_states, _ = bm.ar_man.get_bin_slot_states()
        grasp_tag_num, near_human = gbp.best_current_bin_move(slot_states, bm.ar_man.human_slots)
        if grasp_tag_num < 0:
            rospy.sleep(0.3)
            continue
        place_tag_num = bm.ar_man.get_empty_slot(near_human)

        print 'Plan step:', plan_step
        print grasp_tag_num, place_tag_num

        if False:
            if not bm.move_bin(grasp_tag_num, place_tag_num):
                reset = True
                print 'Failed moving bin from %d to %d' % (grasp_tag_num, place_tag_num)
            else:
                plan_step += 1

if __name__ == "__main__":
    main()
