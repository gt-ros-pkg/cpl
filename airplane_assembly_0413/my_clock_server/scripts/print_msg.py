#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys

import rospy
import roslib
roslib.load_manifest('my_clock_server')

import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4']='PySide'
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

#import pylab
import matplotlib.pyplot as pyplot
pyplot.ion()

from PySide import QtCore, QtGui

#from mainwindow import *
from rosgraph_msgs.msg import Clock

USE_TIMER      = True

################################################################
# 
################################################################


################################################################
#  Main
################################################################



def main():

    rospy.init_node("What")
 
    r = rospy.Rate(3)
 
    i = 0    

    while (~rospy.is_shutdown()):

       print i

       r.sleep()
       
       i = i + 1

if __name__ == "__main__":
    main()












