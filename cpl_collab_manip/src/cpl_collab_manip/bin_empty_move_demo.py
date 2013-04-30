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
    #arm_prefix = "/sim1"
    arm_prefix = ""
    bm = BinManager(arm_prefix, ar_empty_locs)
    #rospy.sleep(3.)
    #print bm.ar_man.get_filled_slots(True), bm.ar_man.get_filled_slots(False)
    #return

    reset = True
    plan_step = 0
    while not rospy.is_shutdown():
        print 'Plan step:', plan_step
        if reset:
            raw_input("Move to home")
            bm.system_reset()
            reset = False

        place_tag_num = bm.ar_man.get_empty_slot(True)
        if place_tag_num < 0:
            place_tag_num = bm.ar_man.get_empty_slot(False)
            grasp_tags = bm.ar_man.get_filled_slots(True)
        else:
            grasp_tags = bm.ar_man.get_filled_slots(False)
        grasp_tag_num = grasp_tags[np.random.randint(0,len(grasp_tags))]
        print bm.ar_man.get_bin_slot_states()
        print grasp_tag_num, place_tag_num

        if not bm.move_bin(grasp_tag_num, place_tag_num):
            reset = True
            print 'Failed moving bin from %d to %d' % (grasp_tag_num, place_tag_num)
        else:
            plan_step += 1

if __name__ == "__main__":
    main()
