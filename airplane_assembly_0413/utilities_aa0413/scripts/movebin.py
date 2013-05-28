# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'movebin.ui'
#
# Created: Fri Apr 12 15:23:12 2013
#      by: pyside-uic 0.2.13 running on PySide 1.1.0
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_movebin(object):
    def setupUi(self, movebin):
        movebin.setObjectName("movebin")
        movebin.resize(375, 283)
        self.centralwidget = QtGui.QWidget(movebin)
        self.centralwidget.setObjectName("centralwidget")
        self.bin_list = QtGui.QListWidget(self.centralwidget)
        self.bin_list.setGeometry(QtCore.QRect(20, 20, 161, 192))
        self.bin_list.setObjectName("bin_list")
        self.movein_button = QtGui.QPushButton(self.centralwidget)
        self.movein_button.setGeometry(QtCore.QRect(230, 40, 99, 27))
        self.movein_button.setObjectName("movein_button")
        self.moveout_button = QtGui.QPushButton(self.centralwidget)
        self.moveout_button.setGeometry(QtCore.QRect(230, 80, 99, 27))
        self.moveout_button.setObjectName("moveout_button")
        movebin.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(movebin)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 375, 25))
        self.menubar.setObjectName("menubar")
        movebin.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(movebin)
        self.statusbar.setObjectName("statusbar")
        movebin.setStatusBar(self.statusbar)

        self.retranslateUi(movebin)
        QtCore.QMetaObject.connectSlotsByName(movebin)

    def retranslateUi(self, movebin):
        movebin.setWindowTitle(QtGui.QApplication.translate("movebin", "MainWindow", None, QtGui.QApplication.UnicodeUTF8))
        self.movein_button.setText(QtGui.QApplication.translate("movebin", "Move in", None, QtGui.QApplication.UnicodeUTF8))
        self.moveout_button.setText(QtGui.QApplication.translate("movebin", "Move out", None, QtGui.QApplication.UnicodeUTF8))

