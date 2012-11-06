#!/usr/bin/env python
# Software License Agreement (BSD License)
#
#  Copyright (c) 2012, Georgia Institute of Technology
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#  * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#  * Neither the name of the Georgia Institute of Technology nor the names of
#     its contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  'AS IS' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import roslib; roslib.load_manifest('pr2_gripper_segmentation')
import rospy
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import PoseStamped
import tf
from pr2_gripper_segmentation.srv import *
import sys
import numpy as np
import math
from math import sqrt
import cv2
import cv2.cv as cv


class DataProcess:

  def process(self, name):
    file = open(name)
    while 1:
      line = file.readline()
      if not line:
        break

      line_str = line.split(',')

      fn = int(line_str[0])
      im = cv2.imread('/u/swl33/data/'+self.getFileName(fn) +'l.jpg')
      cv2.imshow('1', im)

  def getFileName(self, num):
    if (num < 100):
      if (num < 10):
        return "00" + str(num)
      else:
        return "0" + str(num)

if __name__ == '__main__':
  try:
    node = DataProcess()
    node.process('/u/swl33/data/myData.csv')
    rospy.loginfo('Done...')

  except rospy.ROSInterruptException: pass
