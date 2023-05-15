from PyQt5 import QtCore, QtGui, QtWidgets
from ..layouts.widget_camera import Ui_WidgetCamera
from queue import Queue
from ...threads.thread_capture import CaptureThread
from ...threads.thread_tracking import TrackingThread


class WidgetCamera(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_WidgetCamera()
        self.ui.setupUi(self)
        
        # Connect signals
        self.create_queue()
        self.connect_signal()

    def create_queue(self):
        self.capture_queue = Queue()
        self.tracking_queue = Queue()
        
    def create_threads(self):
        self.capture_thread = CaptureThread(self.ui.qline_camera_path.text(), self.capture_queue)
        self.tracking_thread = TrackingThread(self.capture_queue, self.tracking_queue)
        
    def start_all_threads(self):
        self.create_threads()
        self.capture_thread.start()
        self.tracking_thread.start()

    def connect_signal(self):
        pass
    
    def paintEvent(self, e):
        if not self.tracking_queue.empty():
            frame = self.tracking_queue.get()
            qt_image = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
            self.ui.qlabel_frame.setPixmap(QtGui.QPixmap.fromImage(qt_image))
        self.update()
            