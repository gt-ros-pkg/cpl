import rospy
import roslib

roslib.load_manifest('utilities_aa0413')

import utilities_aa0413.ar_tag_tracker










rospy.init_node('test2')
utilities_aa0413.ar_tag_tracker.init()
rospy.sleep(1)
print utilities_aa0413.ar_tag_tracker.get_empty_workspace_location()
print utilities_aa0413.ar_tag_tracker.get_empty_nonworkspace_location()
utilities_aa0413.ar_tag_tracker.sim_movebinin(14)
utilities_aa0413.ar_tag_tracker.sim_movebinout(2)
utilities_aa0413.ar_tag_tracker.sim_movebinout(15)
utilities_aa0413.ar_tag_tracker.sim_movebinout(11)
utilities_aa0413.ar_tag_tracker.sim_movebinin(14)
utilities_aa0413.ar_tag_tracker.sim_movebinin(7)
utilities_aa0413.ar_tag_tracker.sim_movebinin(10)

#print utilities_aa0413.ar_tag_tracker.is_bin_in_workspace(14)
#print utilities_aa0413.ar_tag_tracker.is_bin_in_workspace(12)














