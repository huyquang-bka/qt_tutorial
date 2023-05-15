from PyQt5 import QtCore, QtGui, QtWidgets
from ..layouts.main_window import Ui_MainWindow
from .c_widget_camera import WidgetCamera
from ...utils.tools import recommend_row_col

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.widget_camera = WidgetCamera()
        self.list_camera = []
        
        self.setup_grid()
        # Connect signals
        self.connect_signal()

    def connect_signal(self):
        self.ui.btn_create_camera.clicked.connect(self.create_camera)
        self.ui.btn_start.clicked.connect(self.start_all_camera)

    def create_camera(self):
        self.remove_all_camera()
        try:
            num_camera = int(self.ui.qline_num_camera.text())
            self.add_all_camera(num_camera)
        except:
            QtWidgets.QMessageBox.critical(self, "Error", "Please enter a number")
            return

    def start_all_camera(self):
        for camera in self.list_camera:
            camera.start_all_threads()

    def setup_grid(self):
        self.grid_layout_cameras = QtWidgets.QGridLayout()
        self.grid_layout_cameras.setContentsMargins(0, 0, 0, 0)
        self.ui.frame.setLayout(self.grid_layout_cameras)
    
    def add_all_camera(self, num_camera=20):
        row, col = recommend_row_col(num_camera)
        for i in range(row):
            for j in range(col):
                camera = WidgetCamera()
                self.add_camera(camera, i, j)
        
    def add_camera(self, camera, row, column):
        self.list_camera.append(camera)
        self.grid_layout_cameras.addWidget(camera, row, column)
        
    def remove_all_camera(self):
        for camera in self.list_camera:
            self.grid_layout_cameras.removeWidget(camera)
        self.list_camera.clear()
    
    