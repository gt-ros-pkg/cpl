import rospy
import roslib

roslib.load_manifest('utilities_aa0413')

import utilities_aa0413.task_description








#rospy.init_node('test2')

t =  utilities_aa0413.task_description.taskfile2dict('task2.xml')
actions = utilities_aa0413.task_description.gen_random_task(t)
print actions












