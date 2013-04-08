#!/usr/bin/env python
import rospy
import sys

import roslib
roslib.load_manifest('project_simulation')
import time


from geometry_msgs.msg import *
from std_msgs.msg import *
from project_simulation.msg import *
from visualization_msgs.msg import *

import math
import copy
import tf

#publish Hz
PUB_RATE = 60

'''
Collects following information:
- Probability distribution from the inference engine
- Bin - alvar_markers or bin-id & loc-names
- endfactor position, as well as if its carrying a bin(bin_id in that case)
'''
def collect_info():
    
