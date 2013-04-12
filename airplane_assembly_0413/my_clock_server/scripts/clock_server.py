#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys

import rospy
import roslib
roslib.load_manifest('my_clock_server')
from rosgraph_msgs.msg import Clock

import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4']='PySide'
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

#import pylab
import matplotlib.pyplot as pyplot
pyplot.ion()

from PySide import QtCore, QtGui
from threading import Thread

from my_clock_server_gui import *

import signal


def signal_handler(signal, frame):
        print 'Signal received, quit!'
        rospy.signal_shutdown('we re quitting')
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


################################################################
# 
################################################################

RATE           = 30
mymainwindowui = None
running        = False
speed          = 1

def speed_slider_valuechanged(v):
    if v <= 100:
        v = v / 100.0
    else:
        v = ((200 - v)/100.0) * 1 + (1 - (200 - v)/100.0) * 10

    mymainwindowui.speed_label.setText(str(v))
    
    global speed
    speed = v

  

def play_button_clicked():
    global running
    running = not running
    if running:
       mymainwindowui.play_button.setText('Stop')
    else:
       mymainwindowui.play_button.setText('Resume')


################################################################
#  Main
################################################################

def ros_thread():

    pub = rospy.Publisher('clock', Clock)

    clockmsg = Clock()

    clockmsg.clock.secs  = 1
    clockmsg.clock.nsecs = 1

    r   = rospy.Rate(RATE)
    
    while (not rospy.is_shutdown()):

       if running:
           ddd = rospy.Duration(1.0 / RATE * speed)
           clockmsg.clock = clockmsg.clock + ddd

       pub.publish(clockmsg)

       mymainwindowui.label1.setText('Sec        ' + str(clockmsg.clock.secs))
       # mymainwindowui.label2.setText('NSecs   ' + str(clockmsg.clock.nsecs))

       r.sleep()


def run_gui():

    global mymainwindowui

    app            = QtGui.QApplication(sys.argv)
    win            = QtGui.QMainWindow()
    mymainwindowui = Ui_ClockServerMainWindow()
    mymainwindowui.setupUi(win)

    mymainwindowui.speed_slider.valueChanged[int].connect(speed_slider_valuechanged)
    mymainwindowui.play_button.clicked.connect(play_button_clicked)

    # refresh to catch signals
    timer = QtCore.QTimer()
    timer.start(1000)  # You may change this if you wish.
    timer.timeout.connect(lambda: None)  # Let the interpreter run each 500 ms.

    win.show()
    app.exec_()


def main():

    print 'Running...'
    rospy.init_node("MyClockServer")

    thread = Thread(target = ros_thread)
    thread.start()
    
    run_gui()
    rospy.signal_shutdown('we re quitting')
    

if __name__ == "__main__":
    main()












