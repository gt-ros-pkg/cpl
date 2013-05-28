#!/usr/bin/python
# -*- coding: utf-8 -*-

import rospy
import roslib

roslib.load_manifest('utilities_aa0413')

import utilities_aa0413.ar_tag_tracker

import sys
from PySide import QtCore, QtGui

from movebin import *

mymainwindowui = None


def getcurrentbinid():
   e = mymainwindowui.bin_list.currentItem ()
   return int(e.text())

def movein_button_clicked():
   binid = getcurrentbinid()
   utilities_aa0413.ar_tag_tracker.sim_movebinin  (binid) 

def moveout_button_clicked():
   binid = getcurrentbinid()
   utilities_aa0413.ar_tag_tracker.sim_movebinout  (binid) 

def main():
   
    # set up ROS
    rospy.init_node('movebinUI')
    utilities_aa0413.ar_tag_tracker.init()

    # set up UI
    global mymainwindowui
    app            = QtGui.QApplication(sys.argv)
    win            = QtGui.QMainWindow()
    mymainwindowui = Ui_movebin()
    mymainwindowui.setupUi(win)

    # bin list
    while utilities_aa0413.ar_tag_tracker.get_latest_msg() is None:
          rospy.sleep(1)

    for m in utilities_aa0413.ar_tag_tracker.get_latest_msg().markers:
       mymainwindowui.bin_list.addItem(str(m.id))
       
    # buttons
    mymainwindowui.movein_button.clicked.connect(movein_button_clicked)
    mymainwindowui.moveout_button.clicked.connect(moveout_button_clicked)

    # run
    win.show()
    app.exec_()


if __name__ == '__main__':
    main()















