# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'arm_cart_control_gui.ui'
#
# Created: Mon Jan 30 02:05:49 2012
#      by: PyQt4 UI code generator 4.7.2
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_Frame(object):
    def setupUi(self, Frame):
        Frame.setObjectName("Frame")
        Frame.resize(702, 476)
        Frame.setFrameShape(QtGui.QFrame.StyledPanel)
        Frame.setFrameShadow(QtGui.QFrame.Raised)
        self.layoutWidget = QtGui.QWidget(Frame)
        self.layoutWidget.setGeometry(QtCore.QRect(360, 20, 321, 311))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout_2 = QtGui.QGridLayout(self.layoutWidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.translate_out = QtGui.QPushButton(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.translate_out.sizePolicy().hasHeightForWidth())
        self.translate_out.setSizePolicy(sizePolicy)
        self.translate_out.setBaseSize(QtCore.QSize(80, 80))
        self.translate_out.setStyleSheet("image: url(:/icons/translate_out.png);\n"
"background-image: url(:/icons/empty.png);")
        self.translate_out.setText("")
        self.translate_out.setObjectName("translate_out")
        self.gridLayout_2.addWidget(self.translate_out, 0, 0, 1, 1)
        self.translate_up = QtGui.QPushButton(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.translate_up.sizePolicy().hasHeightForWidth())
        self.translate_up.setSizePolicy(sizePolicy)
        self.translate_up.setBaseSize(QtCore.QSize(80, 80))
        self.translate_up.setStyleSheet("image: url(:/icons/translate_up.png);\n"
"background-image: url(:/icons/empty.png);")
        self.translate_up.setText("")
        self.translate_up.setObjectName("translate_up")
        self.gridLayout_2.addWidget(self.translate_up, 0, 1, 1, 1)
        self.translate_in = QtGui.QPushButton(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.translate_in.sizePolicy().hasHeightForWidth())
        self.translate_in.setSizePolicy(sizePolicy)
        self.translate_in.setBaseSize(QtCore.QSize(80, 80))
        self.translate_in.setStyleSheet("image: url(:/icons/translate_in.png);\n"
"background-image: url(:/icons/empty.png);")
        self.translate_in.setText("")
        self.translate_in.setObjectName("translate_in")
        self.gridLayout_2.addWidget(self.translate_in, 0, 2, 1, 1)
        self.translate_left = QtGui.QPushButton(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.translate_left.sizePolicy().hasHeightForWidth())
        self.translate_left.setSizePolicy(sizePolicy)
        self.translate_left.setBaseSize(QtCore.QSize(80, 80))
        self.translate_left.setStyleSheet("image: url(:/icons/translate_left.png);\n"
"background-image: url(:/icons/empty.png);")
        self.translate_left.setText("")
        self.translate_left.setObjectName("translate_left")
        self.gridLayout_2.addWidget(self.translate_left, 1, 0, 1, 1)
        self.translate_right = QtGui.QPushButton(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.translate_right.sizePolicy().hasHeightForWidth())
        self.translate_right.setSizePolicy(sizePolicy)
        self.translate_right.setBaseSize(QtCore.QSize(80, 80))
        self.translate_right.setStyleSheet("image: url(:/icons/translate_right.png);\n"
"background-image: url(:/icons/empty.png);")
        self.translate_right.setText("")
        self.translate_right.setObjectName("translate_right")
        self.gridLayout_2.addWidget(self.translate_right, 1, 2, 1, 1)
        self.translate_down = QtGui.QPushButton(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.translate_down.sizePolicy().hasHeightForWidth())
        self.translate_down.setSizePolicy(sizePolicy)
        self.translate_down.setBaseSize(QtCore.QSize(80, 80))
        self.translate_down.setStyleSheet("image: url(:/icons/translate_down.png);\n"
"background-image: url(:/icons/empty.png);")
        self.translate_down.setText("")
        self.translate_down.setObjectName("translate_down")
        self.gridLayout_2.addWidget(self.translate_down, 2, 1, 1, 1)
        self.frame = QtGui.QComboBox(Frame)
        self.frame.setGeometry(QtCore.QRect(420, 350, 261, 31))
        self.frame.setEditable(True)
        self.frame.setObjectName("frame")
        self.label_2 = QtGui.QLabel(Frame)
        self.label_2.setGeometry(QtCore.QRect(330, 350, 71, 31))
        self.label_2.setStyleSheet("font: 16pt;")
        self.label_2.setObjectName("label_2")
        self.widget = QtGui.QWidget(Frame)
        self.widget.setGeometry(QtCore.QRect(20, 20, 321, 311))
        self.widget.setObjectName("widget")
        self.gridLayout = QtGui.QGridLayout(self.widget)
        self.gridLayout.setObjectName("gridLayout")
        self.rotate_x_pos = QtGui.QPushButton(self.widget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rotate_x_pos.sizePolicy().hasHeightForWidth())
        self.rotate_x_pos.setSizePolicy(sizePolicy)
        self.rotate_x_pos.setBaseSize(QtCore.QSize(80, 80))
        self.rotate_x_pos.setStyleSheet("image: url(:/icons/roate_x_neg.png);\n"
"background-image: url(:/icons/empty.png);")
        self.rotate_x_pos.setText("")
        self.rotate_x_pos.setObjectName("rotate_x_pos")
        self.gridLayout.addWidget(self.rotate_x_pos, 0, 0, 1, 1)
        self.rotate_y_pos = QtGui.QPushButton(self.widget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rotate_y_pos.sizePolicy().hasHeightForWidth())
        self.rotate_y_pos.setSizePolicy(sizePolicy)
        self.rotate_y_pos.setBaseSize(QtCore.QSize(80, 80))
        self.rotate_y_pos.setStyleSheet("image: url(:/icons/rotate_y_pos.png);\n"
"background-image: url(:/icons/empty.png);")
        self.rotate_y_pos.setText("")
        self.rotate_y_pos.setObjectName("rotate_y_pos")
        self.gridLayout.addWidget(self.rotate_y_pos, 0, 1, 1, 1)
        self.rotate_x_neg = QtGui.QPushButton(self.widget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rotate_x_neg.sizePolicy().hasHeightForWidth())
        self.rotate_x_neg.setSizePolicy(sizePolicy)
        self.rotate_x_neg.setBaseSize(QtCore.QSize(80, 80))
        self.rotate_x_neg.setStyleSheet("image: url(:/icons/roate_x_pos.png);\n"
"background-image: url(:/icons/empty.png);")
        self.rotate_x_neg.setText("")
        self.rotate_x_neg.setObjectName("rotate_x_neg")
        self.gridLayout.addWidget(self.rotate_x_neg, 0, 2, 1, 1)
        self.rotate_z_neg = QtGui.QPushButton(self.widget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rotate_z_neg.sizePolicy().hasHeightForWidth())
        self.rotate_z_neg.setSizePolicy(sizePolicy)
        self.rotate_z_neg.setBaseSize(QtCore.QSize(80, 80))
        self.rotate_z_neg.setStyleSheet("image: url(:/icons/rotate_z_neg.png);\n"
"background-image: url(:/icons/empty.png);")
        self.rotate_z_neg.setText("")
        self.rotate_z_neg.setObjectName("rotate_z_neg")
        self.gridLayout.addWidget(self.rotate_z_neg, 1, 0, 1, 1)
        self.rotate_z_pos = QtGui.QPushButton(self.widget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rotate_z_pos.sizePolicy().hasHeightForWidth())
        self.rotate_z_pos.setSizePolicy(sizePolicy)
        self.rotate_z_pos.setBaseSize(QtCore.QSize(80, 80))
        self.rotate_z_pos.setStyleSheet("image: url(:/icons/rotate_z_pos.png);\n"
"background-image: url(:/icons/empty.png);")
        self.rotate_z_pos.setText("")
        self.rotate_z_pos.setObjectName("rotate_z_pos")
        self.gridLayout.addWidget(self.rotate_z_pos, 1, 2, 1, 1)
        self.rotate_y_neg = QtGui.QPushButton(self.widget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rotate_y_neg.sizePolicy().hasHeightForWidth())
        self.rotate_y_neg.setSizePolicy(sizePolicy)
        self.rotate_y_neg.setBaseSize(QtCore.QSize(80, 80))
        self.rotate_y_neg.setStyleSheet("image: url(:/icons/rotate_y_neg.png);\n"
"background-image: url(:/icons/empty.png);")
        self.rotate_y_neg.setText("")
        self.rotate_y_neg.setObjectName("rotate_y_neg")
        self.gridLayout.addWidget(self.rotate_y_neg, 2, 1, 1, 1)
        self.widget1 = QtGui.QWidget(Frame)
        self.widget1.setGeometry(QtCore.QRect(20, 350, 271, 111))
        self.widget1.setObjectName("widget1")
        self.horizontalLayout = QtGui.QHBoxLayout(self.widget1)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtGui.QLabel(self.widget1)
        self.label.setStyleSheet("font: 20pt;")
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.arm_left = QtGui.QPushButton(self.widget1)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.arm_left.sizePolicy().hasHeightForWidth())
        self.arm_left.setSizePolicy(sizePolicy)
        self.arm_left.setBaseSize(QtCore.QSize(80, 80))
        self.arm_left.setStyleSheet("image: url(:/icons/arm_left_off.png);\n"
"background-image: url(:/icons/empty.png);")
        self.arm_left.setObjectName("arm_left")
        self.horizontalLayout.addWidget(self.arm_left)
        self.arm_right = QtGui.QPushButton(self.widget1)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.arm_right.sizePolicy().hasHeightForWidth())
        self.arm_right.setSizePolicy(sizePolicy)
        self.arm_right.setBaseSize(QtCore.QSize(80, 80))
        self.arm_right.setStyleSheet("image: url(:/icons/arm_right_off.png);\n"
"background-image: url(:/icons/empty.png);")
        self.arm_right.setObjectName("arm_right")
        self.horizontalLayout.addWidget(self.arm_right)

        self.retranslateUi(Frame)
        QtCore.QMetaObject.connectSlotsByName(Frame)

    def retranslateUi(self, Frame):
        Frame.setWindowTitle(QtGui.QApplication.translate("Frame", "Frame", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Frame", "Frame", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Frame", "Arm", None, QtGui.QApplication.UnicodeUTF8))
        self.arm_left.setText(QtGui.QApplication.translate("Frame", "Left Arm", None, QtGui.QApplication.UnicodeUTF8))
        self.arm_right.setText(QtGui.QApplication.translate("Frame", "Right Arm", None, QtGui.QApplication.UnicodeUTF8))

import icons_rc