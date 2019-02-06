# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'plot_ui.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.checkBox_imu0_x = QtWidgets.QCheckBox(Dialog)
        self.checkBox_imu0_x.setGeometry(QtCore.QRect(60, 40, 87, 20))
        self.checkBox_imu0_x.setObjectName("checkBox_imu0_x")
        self.checkBox_imu0_y = QtWidgets.QCheckBox(Dialog)
        self.checkBox_imu0_y.setGeometry(QtCore.QRect(60, 60, 87, 20))
        self.checkBox_imu0_y.setObjectName("checkBox_imu0_y")
        self.checkBox_imu0_z = QtWidgets.QCheckBox(Dialog)
        self.checkBox_imu0_z.setGeometry(QtCore.QRect(60, 80, 87, 20))
        self.checkBox_imu0_z.setObjectName("checkBox_imu0_z")
        self.checkBox_imu1_y = QtWidgets.QCheckBox(Dialog)
        self.checkBox_imu1_y.setGeometry(QtCore.QRect(60, 120, 87, 20))
        self.checkBox_imu1_y.setObjectName("checkBox_imu1_y")
        self.checkBox_imu1_z = QtWidgets.QCheckBox(Dialog)
        self.checkBox_imu1_z.setGeometry(QtCore.QRect(60, 140, 87, 20))
        self.checkBox_imu1_z.setObjectName("checkBox_imu1_z")
        self.checkBox_imu1_x = QtWidgets.QCheckBox(Dialog)
        self.checkBox_imu1_x.setGeometry(QtCore.QRect(60, 100, 87, 20))
        self.checkBox_imu1_x.setObjectName("checkBox_imu1_x")
        self.checkBox_imu2_y = QtWidgets.QCheckBox(Dialog)
        self.checkBox_imu2_y.setGeometry(QtCore.QRect(60, 180, 87, 20))
        self.checkBox_imu2_y.setObjectName("checkBox_imu2_y")
        self.checkBox_imu2_z = QtWidgets.QCheckBox(Dialog)
        self.checkBox_imu2_z.setGeometry(QtCore.QRect(60, 200, 87, 20))
        self.checkBox_imu2_z.setObjectName("checkBox_imu2_z")
        self.checkBox_imu2_x = QtWidgets.QCheckBox(Dialog)
        self.checkBox_imu2_x.setGeometry(QtCore.QRect(60, 160, 87, 20))
        self.checkBox_imu2_x.setObjectName("checkBox_imu2_x")
        self.checkBox_EMG_1 = QtWidgets.QCheckBox(Dialog)
        self.checkBox_EMG_1.setGeometry(QtCore.QRect(60, 220, 121, 20))
        self.checkBox_EMG_1.setObjectName("checkBox_EMG_1")
        self.checkBox_buttons = QtWidgets.QCheckBox(Dialog)
        self.checkBox_buttons.setGeometry(QtCore.QRect(60, 20, 87, 20))
        self.checkBox_buttons.setObjectName("checkBox_buttons")
        self.checkBox_EMG_2 = QtWidgets.QCheckBox(Dialog)
        self.checkBox_EMG_2.setGeometry(QtCore.QRect(60, 240, 121, 20))
        self.checkBox_EMG_2.setObjectName("checkBox_EMG_2")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.checkBox_imu0_x.setText(_translate("Dialog", "x"))
        self.checkBox_imu0_y.setText(_translate("Dialog", "y"))
        self.checkBox_imu0_z.setText(_translate("Dialog", "z"))
        self.checkBox_imu1_y.setText(_translate("Dialog", "y"))
        self.checkBox_imu1_z.setText(_translate("Dialog", "z"))
        self.checkBox_imu1_x.setText(_translate("Dialog", "x"))
        self.checkBox_imu2_y.setText(_translate("Dialog", "y"))
        self.checkBox_imu2_z.setText(_translate("Dialog", "z"))
        self.checkBox_imu2_x.setText(_translate("Dialog", "x"))
        self.checkBox_EMG_1.setText(_translate("Dialog", "EMG Channel 1"))
        self.checkBox_buttons.setText(_translate("Dialog", "buttons"))
        self.checkBox_EMG_2.setText(_translate("Dialog", "EMG Channel 2"))

