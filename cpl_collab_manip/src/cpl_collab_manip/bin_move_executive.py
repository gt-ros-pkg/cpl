#! /usr/bin/python

import numpy as np
import yaml
from functools import partial

import roslib
roslib.load_manifest("cpl_collab_manip")
import rospy

from bin_manager import BinManager
from greedy_bin_plan import GreedyBinPlanner
from msg import BinRewards

class MoveExecutive(object):
    def __init__(self, arm_prefix, ar_empty_locs, available_bins):
        self.bm = BinManager(arm_prefix, ar_empty_locs, available_bins)
        self.gbp = GreedyBinPlanner()
        self.reward_pub = rospy.Publisher("/bin_rewards", BinRewards)
        rospy.sleep(0.2)
        self.grasp_tag_num = -1
        self.place_tag_num = -1
        self.near_human = False
        self.slot_states = None
        def publish_rewards(me, te):
            slot_states, _ = me.bm.ar_man.get_bin_slot_states()
            if slot_states is not None:
                msg = BinRewards()
                if me.gbp.inf is not None:
                    msg.is_swap, msg.rewards_rm, msg.rewards_dv = me.gbp.get_current_rewards(
                                                        slot_states, me.bm.ar_man.human_slots)
                msg.slot_states = slot_states
                msg.reachable_slots = me.bm.ar_man.human_slots
                me.reward_pub.publish(msg)
        pub_rewards = partial(publish_rewards, self)
        self.pub_timer = rospy.Timer(rospy.Duration(0.1), pub_rewards, oneshot=False)

    def move_loop(self):
        reset = True
        plan_step = 0
        self.slot_states, _ = self.bm.ar_man.get_bin_slot_states()
        while not rospy.is_shutdown():
            if reset:
                raw_input("Move to home")
                self.bm.system_reset()
                reset = False
                raw_input("Start moving")

            self.slot_states, _ = self.bm.ar_man.get_bin_slot_states()
            self.grasp_tag_num, self.near_human = self.gbp.best_current_bin_move(self.slot_states, 
                                                                         self.bm.ar_man.human_slots)

            print '-'*35, 'Planning', '-'*35
            print self.slot_states
            print self.bm.ar_man.human_slots
            if self.slot_states is None:
                print "Inference not running"
            else:
                print "bin_id: %d\nmove_near_human: %d" % (self.grasp_tag_num, self.near_human)
            print '-'*80
            print

            if self.grasp_tag_num < 0:
                rospy.sleep(0.3)
                continue
            self.place_tag_num = self.bm.ar_man.get_empty_slot(self.near_human)

            if not self.bm.move_bin(self.grasp_tag_num, self.place_tag_num):
                reset = True
                print 'Failed moving bin from %d to %d' % (self.grasp_tag_num, self.place_tag_num)
            else:
                plan_step += 1

def main():
    np.set_printoptions(precision=4)
    rospy.init_node("bin_delivery_demo")
    from roslaunch.substitution_args import resolve_args
    f = file(resolve_args("$(find cpl_collab_manip)/config/bin_locs.yaml"), 'r')
    ar_empty_locs = yaml.load(f)['data']
    f.close()
    #arm_prefix = "/sim1"
    arm_prefix = ""
    available_bins = [3, 11, 12, 14, 2, 15, 10, 7, 13]
    me = MoveExecutive(arm_prefix, ar_empty_locs, available_bins)
    me.move_loop()

if __name__ == "__main__":
    main()
