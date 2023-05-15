from main_app.views.controller.c_main_window import MainWindow
from PyQt5 import QtWidgets, QtCore, QtGui


if __name__ == "__main__":
    import sys
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
    