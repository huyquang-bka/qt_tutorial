from PyQt5 import QtCore
import cv2


class CaptureThread(QtCore.QThread):
    def __init__(self, camera_url, capture_queue):
        super().__init__()
        self.is_active = False
        self.camera_url = camera_url
        self.capture_queue = capture_queue
        
    def run(self):
        print("Thread started")
        self.is_active = True
        cap = cv2.VideoCapture(self.camera_url)
        while self.is_active:
            ret, frame = cap.read()
            if not ret:
                break
            if self.capture_queue.empty():
                self.capture_queue.put(frame)
            self.msleep(1)
    
    def stop(self):
        self.is_active = False
        print("Thread stopped")