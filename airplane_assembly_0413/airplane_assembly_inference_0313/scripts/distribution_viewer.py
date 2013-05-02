#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys

import rospy
import roslib
roslib.load_manifest('learn_pyside')
from rospy_tutorials.msg import Floats

import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4']='PySide'
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

#import pylab
import matplotlib.pyplot as pyplot
pyplot.ion()

from PySide import QtCore, QtGui

from mainwindow import *


USE_TIMER      = True

################################################################
# 
################################################################

mymainwindowui = 0

ax     	       = 0
canvas         = 0

subscribers    = {}
distributions  = {}

def update_plot():
    try:
       ax.cla()
       for k in subscribers.keys():
          ax.plot(distributions[k], label=k)
       ax.legend()
       canvas.draw()

    except Exception as e:
       print 'draw error'
       print e

def data_callback(d, topicname):
    
    global distributions

    distributions[topicname] = d.data

    if (USE_TIMER == False):
        update_plot()



################################################################
# Subscribers 
################################################################

def update_topic_combobox():

    #mymainwindowui.topic_list.clear()

    topics = rospy.get_published_topics()
    
    # add new topics
    for t in topics:

       topicname = t[0]
       msgtype   = t[1]

       if len(mymainwindowui.topic_list.findItems(unicode(topicname), QtCore.Qt.MatchFlag.MatchExactly) ) == 0 and msgtype == 'rospy_tutorials/Floats' :
           print 'Add ', topicname, ' to list'
           mymainwindowui.topic_list.addItem(t[0])
           mymainwindowui.topic_list.item(mymainwindowui.topic_list.count() - 1).setCheckState(QtCore.Qt.Unchecked)

    # remove old topics
    for i in range(mymainwindowui.topic_list.count() - 1, -1, -1):

       to_remove  = True
       topicname = mymainwindowui.topic_list.item(i).text()

       for t in topics:
           if t[0] == topicname:
               to_remove = False

       if to_remove:
           print 'Remove ', topicname, ' from list'
           mymainwindowui.topic_list.takeItem(i)

           if subscribers.has_key(topicname):
              subscribers[topicname].unregister()
              subscribers.pop(topicname)
    # ok
    mymainwindowui.topic_list.sortItems()

def topic_list_item_changed(item):

    global subscribers

    topicname = item.text()

    if subscribers.has_key(topicname):
        subscribers[topicname].unregister()
        subscribers.pop(topicname)

    if (item.checkState() == QtCore.Qt.Unchecked):
        print 'Unscribe ', topicname

    if (item.checkState() == QtCore.Qt.Checked):
        subscribers[topicname] = rospy.Subscriber(topicname, Floats, data_callback, topicname)
        print 'Subscribe ', topicname

def button2_clicked():
    print 'button 2'

    update_topic_combobox()


def button1_clicked():
    print 'button 1'

################################################################
#  Main
################################################################



def main():

    global topic
    if (len(sys.argv) > 1):
       topic = sys.argv[1]

    global ax
    global canvas
    global mymainwindowui

    app            = QtGui.QApplication(sys.argv)
    win            = QtGui.QMainWindow()
    mymainwindowui = Ui_MainWindow()
    mymainwindowui.setupUi(win)

    fig    = Figure(figsize=(600,600), dpi=72, facecolor=(1,1,1), edgecolor=(0,0,0))
    ax     = fig.add_subplot(111)
    canvas = FigureCanvas(fig)

    mymainwindowui.matplot_area.addWidget(canvas)

    mymainwindowui.button1.clicked.connect(button1_clicked)
    mymainwindowui.button2.clicked.connect(button2_clicked)
    
    
    if (USE_TIMER):
       mytimer = QtCore.QTimer()
       mytimer.timeout.connect(update_plot)
       mytimer.start(500);

    rospy.init_node("distribution_plot", anonymous=True)
    global subscriber
  
    update_topic_combobox()

    mymainwindowui.topic_list.itemChanged.connect(topic_list_item_changed)

    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()












