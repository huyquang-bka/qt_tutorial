# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/Users/huyquang/huyquang/Company/thu-cuc/resources/uis/widget_camera.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_WidgetCamera(object):
    def setupUi(self, WidgetCamera):
        WidgetCamera.setObjectName("WidgetCamera")
        WidgetCamera.resize(400, 300)
        self.gridLayout = QtWidgets.QGridLayout(WidgetCamera)
        self.gridLayout.setObjectName("gridLayout")
        self.btn_choose_videopath = QtWidgets.QPushButton(WidgetCamera)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_choose_videopath.sizePolicy().hasHeightForWidth())
        self.btn_choose_videopath.setSizePolicy(sizePolicy)
        self.btn_choose_videopath.setObjectName("btn_choose_videopath")
        self.gridLayout.addWidget(self.btn_choose_videopath, 0, 0, 1, 1)
        self.qline_camera_path = QtWidgets.QLineEdit(WidgetCamera)
        self.qline_camera_path.setObjectName("qline_camera_path")
        self.gridLayout.addWidget(self.qline_camera_path, 0, 1, 1, 1)
        self.qlabel_frame = QtWidgets.QLabel(WidgetCamera)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.qlabel_frame.sizePolicy().hasHeightForWidth())
        self.qlabel_frame.setSizePolicy(sizePolicy)
        self.qlabel_frame.setStyleSheet("background: black;\n"
"border: 2px solid blue;\n"
"border-radius: 5px;")
        self.qlabel_frame.setScaledContents(True)
        self.qlabel_frame.setObjectName("qlabel_frame")
        self.gridLayout.addWidget(self.qlabel_frame, 1, 0, 1, 2)

        self.retranslateUi(WidgetCamera)
        QtCore.QMetaObject.connectSlotsByName(WidgetCamera)

    def retranslateUi(self, WidgetCamera):
        _translate = QtCore.QCoreApplication.translate
        WidgetCamera.setWindowTitle(_translate("WidgetCamera", "Form"))
        self.btn_choose_videopath.setText(_translate("WidgetCamera", "Choose Video Path"))
        self.qlabel_frame.setText(_translate("WidgetCamera", "TextLabel"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    WidgetCamera = QtWidgets.QWidget()
    ui = Ui_WidgetCamera()
    ui.setupUi(WidgetCamera)
    WidgetCamera.show()
    sys.exit(app.exec_())
