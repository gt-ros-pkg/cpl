#! /usr/bin/python

import numpy as np

import roslib
roslib.load_manifest("rospy")
roslib.load_manifest("project_simulation")
import rospy

from greedy_bin_plan import GreedyBinPlanner
from project_simulation.msg import BinStateEstimate, move_bin

class SimBinMoveExecutive(object):
    def __init__(self):
        self.gbp = GreedyBinPlanner()
        rospy.Subscriber("/current_state", BinStateEstimate, self.state_cb)
        self.move_bin_pub = rospy.Publisher("/move_bin", move_bin)

    def state_cb(self, s):
        if s.robot_is_moving:
            return
        mb = move_bin()
        print '-'*35, 'Planning', '-'*35
        print s.bin_slots
        print s.reachable_slots
        mb.bin_id, mb.move_near_human = self.gbp.plan_action(s.inference, s.header.stamp, s.bin_slots, 
                                                             s.reachable_slots)
        print mb
        print '-'*80
        print
        if not mb.bin_id < 0:
            self.move_bin_pub.publish(mb)

def main():
    np.set_printoptions(precision=4)
    rospy.init_node("sim_bin_move_executive")
    sim_exec = SimBinMoveExecutive()
    rospy.spin()

if __name__ == "__main__":
    main()
