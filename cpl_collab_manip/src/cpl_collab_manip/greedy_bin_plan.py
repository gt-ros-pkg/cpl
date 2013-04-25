import numpy as np
import matplotlib.pyplot as plt
import roslib
roslib.load_manifest("rospy")
roslib.load_manifest("std_msgs")
roslib.load_manifest("project_simulation")
import rospy
from std_msgs.msg import Float32MultiArray
from project_simulation.msg import BinInference

class GreedyBinPlanner(object):
    def __init__(self):
        self.dur_rm = 5.
        self.dur_dv = 3. * self.dur_rm
        self.delay_thresh = -0.01

        self.inf = None
        rospy.Subscriber("/inference/all_distributions", BinInference, self.save_inference)

    def save_inference(self, msg):
        self.inf = msg

    def best_current_bin_move(self, slot_states, reachable_slots):
        if self.inf is None:
            return None, None
        return self.plan_action(self.inf, rospy.Time.now(), slot_states, reachable_slots)
        #if s.robot_is_moving:
        #    # Can't stop the robot while it's moving
        #    return

    def get_current_rewards(self, slot_states, reachable_slots):
        inference = self.inf
        cur_time = rospy.Time.now()
        distribs = self.conv_distribs(inference)
        t_cur = (cur_time - inference.start_time).to_sec()
        is_swap = self.is_swap(slot_states, reachable_slots)
        if is_swap:
            rewards_rm, rewards_dv = self.swap_rewards(distribs, t_cur, inference.period, 
                                                       slot_states, reachable_slots)
        else:
            rewards_dv = self.fill_rewards(distribs, t_cur, inference.period, 
                                           slot_states, reachable_slots)
            rewards_rm = []
        return bool(is_swap), np.array(rewards_rm).tolist(), np.array(rewards_dv).tolist()

    # Reward (negative cost) for removing a bin early
    # @param start_probs  Probability distribution for the start of this bin step
    # @param end_probs  Probability distribution for the end of this bin step
    # @param i_rm  Time index at which the bin is removed
    def remove_reward(self, start_probs, end_probs, i_rm, t):
        i, dur_rm, dur_dv = i_rm, self.dur_rm, self.dur_dv
        if i >= len(t):
            return 0.
        const_integ = np.sum(-(dur_rm + dur_dv)**2 * start_probs[:i])
        lin_reward = -(np.max(dur_rm + dur_dv - (t[i+1:]-t[i]),0))**2
        lin_integ = np.sum(lin_reward * start_probs[i+1:])
        start_integ = const_integ + lin_integ
        return start_integ * np.sum(end_probs[i+1:])

    # Reward (negative cost) for delivering a bin late
    # @param start_probs  Probability distribution for the start of this bin step
    # @param end_probs  Probability distribution for the end of this bin step
    # @param i_dv  Time index at which the bin is delivered
    def late_reward(self, start_probs, end_probs, i_dv, t):
        i = i_dv
        if i >= len(t):
            return 0.
        late_integ = np.sum(-(np.clip(t[i] - t, 0., np.inf))**2 * start_probs[:])
        return late_integ * np.sum(end_probs[i+1:])

    def is_swap(self, slot_states, reachable_slots):
        reachable_states = np.array([slot_states[i] for i in reachable_slots])
        return np.all(reachable_states >= 0)

    def conv_distribs(self, inference):
        num_bins = len(inference.distributions) / inference.T_len / 2
        return np.reshape(inference.distributions, 
                          (2*num_bins, inference.T_len))

    # @param s BinStateEstimate message representing the current state of the system 
    #          and inference estimates
    # @return (bin_to_move, near_human)
    def plan_action(self, inference, cur_time, slot_states, reachable_slots):
        distribs = self.conv_distribs(inference)
        t_cur = (cur_time - inference.start_time).to_sec()
        if self.is_swap(slot_states, reachable_slots):
            # no open slots, need to swap
            least_necessary_bin, most_delayed_bin = self.plan_swap(distribs, t_cur, inference.period, 
                                                                   slot_states, reachable_slots)
            if least_necessary_bin != -1:
                print "Swap bin %d for bin %d" % (least_necessary_bin, most_delayed_bin)
                # move least_necessary_bin away
                return least_necessary_bin, False
            else:
                print "Too costly to swap."
                # do nothing
                return -1, False
        else:
            most_delayed_bin = self.plan_fill(distribs, t_cur, inference.period,
                                              slot_states, reachable_slots)
            if most_delayed_bin != -1:
                print "Fill bin %d" % (most_delayed_bin)
                # move most_delayed_bin towards
                return most_delayed_bin, True
            else:
                print "No bins needed."
                # do nothing
                return -1, True

    def fill_rewards(self, distribs, t_cur, period, slot_states, reachable_slots):
        num_bins, T_len = distribs.shape[0], distribs.shape[1]
        t = np.linspace(0., T_len * period, T_len)
        i_dv = round((t_cur + self.dur_dv) / period)

        # calculate rewards for delivering a bin late
        rewards_dv = np.inf * np.ones(num_bins)
        for i, b in enumerate(slot_states):
            if b < 0 or i in reachable_slots:
                continue
            start_probs, end_probs = distribs[2*(b-1)+0], distribs[2*(b-1)+1]
            rewards_dv[i] = self.late_reward(start_probs, end_probs, i_dv, t)
        return rewards_dv

    def plan_fill(self, distribs, t_cur, period, slot_states, reachable_slots):
        rewards_dv = self.fill_rewards(distribs, t_cur, period, slot_states, reachable_slots)

        most_delayed_bin = np.argmin(rewards_dv)
        most_delayed_reward = rewards_dv[most_delayed_bin]

        print "Fill rewards delay:"
        print rewards_dv

        if most_delayed_reward < self.delay_thresh:
            return slot_states[most_delayed_bin]
        else:
            return -1

    # Planning for bin swapping.
    def swap_rewards(self, distribs, t_cur, period, slot_states, reachable_slots):
        num_bins, T_len = distribs.shape[0], distribs.shape[1]
        t = np.linspace(0., T_len * period, T_len)
        # time bin gets removed if we start moving now
        i_rm = round((t_cur + self.dur_rm) / period)
        # time bin gets delivered if we start moving now
        i_dv = round((t_cur + self.dur_rm + self.dur_dv) / period)

        # calculate rewards for removing a bin early
        rewards_rm = np.zeros(len(reachable_slots))
        for i, slot in enumerate(reachable_slots):
            b = slot_states[slot]
            if b < 0 or i not in reachable_slots:
                rewards_rm[i] = -np.inf
            else:
                start_probs, end_probs = distribs[2*(b-1)+0], distribs[2*(b-1)+1]
                rewards_rm[i] = self.remove_reward(start_probs, end_probs, i_rm, t)

        # calculate rewards for delivering a bin late
        rewards_dv = np.inf * np.ones(num_bins)
        #bin_locs = [-1] * num_bins
        for i, b in enumerate(slot_states):
            if b < 0 or i in reachable_slots:
                continue
            #bin_locs[b] = i
            start_probs, end_probs = distribs[2*(b-1)+0], distribs[2*(b-1)+1]
            rewards_dv[i] = self.late_reward(start_probs, end_probs, i_dv, t)
        return rewards_rm, rewards_dv

    def plan_swap(self, distribs, t_cur, period, slot_states, reachable_slots):
        rewards_rm, rewards_dv = self.swap_rewards(distribs, t_cur, period, 
                                                   slot_states, reachable_slots)

        # determine whether and which bins to swap
        least_necessary_bin = np.argmax(rewards_rm)
        most_delayed_bin = np.argmin(rewards_dv)
        least_necessary_reward = rewards_rm[least_necessary_bin]
        most_delayed_reward = rewards_dv[most_delayed_bin]
        print "Swap rewards remove:"
        print rewards_rm
        print "Swap rewards delay:"
        print rewards_dv

        if most_delayed_reward < least_necessary_reward:
            # delaying this bin is becoming too costly
            return slot_states[least_necessary_bin], slot_states[most_delayed_bin]
        else:
            return -1, -1

def main():
    rospy.init_node("greedy_bin_plan")
    gbp = GreedyBinPlanner()
    rospy.spin()

if __name__ == "__main__":
    main()
