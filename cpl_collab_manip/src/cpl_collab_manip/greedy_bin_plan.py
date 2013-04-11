import numpy as np
import matplotlib.pyplot as plt
import roslib
roslib.load_manifest("rospy")
roslib.load_manifest("std_msgs")
import rospy
from std_msgs.msg import Float32MultiArray

class GreedyBinPlanner(object):
    def __init__(self):
        self.dur_rm = 5.
        self.dur_dv = 3. * self.dur_rm
        rospy.Subscriber("/inference/all_distributions", Float32MultiArray, self.plot)
        self.i = 1
        self.fig = plt.figure(1)
        self.plt, = plt.plot([], [])

    # Reward (negative cost) for removing a bin early
    # @param start_probs  Probability distribution for the start of this bin step
    # @param end_probs  Probability distribution for the end of this bin step
    # @param i_rm  Time index at which the bin is removed
    def remove_reward(self, start_probs, end_probs, i_rm):
        t, i, dur_rm, dur_dv = self.t, i_rm, self.dur_rm, self.dur_dv
        const_integ = np.sum(-(dur_rm + dur_dv)**2 * end_probs[:i])
        lin_reward = -(np.max(dur_rm + dur_dv - (t[i+1:]-t[i]),0))**2
        lin_integ = np.sum(lin_reward * end_probs[i+1:])
        start_integ = const_integ + lin_integ
        return start_integ * np.sum(start_probs[i+1:])

    # Reward (negative cost) for delivering a bin late
    # @param start_probs  Probability distribution for the start of this bin step
    # @param i_dv  Time index at which the bin is delivered
    def late_reward(self, start_probs, i_dv):
        t, i, = self.t, i_dv
        return np.sum(-(np.clip(t[i] - t, 0., np.inf))**2 * start_probs[:])

    # Planning for bin swapping.
    # @param s BinStateEstimate message representing the current state of the system 
    #          and inference estimates
    def plan_action(self, s):
        if s.robot_is_moving:
            # Can't stop the robot while it's moving
            return
        num_bins = len(s.inference.distributions) / s.inference.T_len / 2
        distribs = np.reshape(s.inference.distributions, 
                              (2*num_bins, s.inference.T_len))
        t_cur = (s.header.stamp - s.inference.start_time).to_sec()
        # time bin gets removed if we start moving now
        i_rm = round((t_cur + self.dur_rm) / s.period)
        # time bin gets delivered if we start moving now
        i_dv = round((t_cur + self.dur_rm + self.dur_dv) / s.period)

        # calculate rewards for removing a bin early
        rewards_rm = np.zeros(len(s.reachable_slots))
        for i, slot in enumerate(s.reachable_slots):
            b = s.bin_slots[slot]
            if b < 0:
                rewards_rm[i] = -np.inf
            else:
                start_probs, end_probs = distribs[2*b+0], distribs[2*b+1]
                rewards_rm[i] = self.remove_reward(start_probs, end_probs, i_rm)

        # calculate rewards for delivering a bin late
        rewards_dv = np.inf * np.ones(len(num_bins))
        bin_locs = [-1] * len(num_bins)
        for i, b in enumerate(s.bin_slots):
            if b < 0:
                continue
            bin_locs[b] = i
            start_probs = distribs[2*b+0]
            rewards_dv[i] = self.late_reward(start_probs, i_dv)

        # determine whether and which bins to swap
        least_necessary_bin = np.argmax(rewards_rm)
        most_delayed_bin = np.argmin(rewards_dv)
        least_necessary_reward = rewards_rm[least_necessary_bin]
        most_delayed_reward = rewards_dv[most_delayed_bin]

        if most_delayed_reward < least_necessary_reward:
            # delaying this bin is becoming too costly
            print "Swap bin %d for bin %d" % (least_necessary_bin, most_delayed_bin)

def main():
    rospy.init_node("greedy_bin_plan")
    gbp = GreedyBinPlanner()
    rospy.spin()

if __name__ == "__main__":
    main()
