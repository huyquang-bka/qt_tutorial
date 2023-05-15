import time
from PyQt5 import QtCore
import cv2
from ..utils.detect_yolov5 import OCTracker


class TrackingThread(QtCore.QThread):
    def __init__(self, capture_queue, tracking_queue):
        super().__init__()
        self.thread_activate = False
        self.capture_queue = capture_queue
        self.tracking_queue = tracking_queue
        self.tracker = OCTracker()
        self.tracker.weights = "resources/weights/face_n_320_openvino_model"
        self.tracker.imgsz = 320
        self.tracker.device = "cpu"
        self.tracker.conf_thres = 0.4
        self.tracker.classes = [0]
        self.tracker.agnostic_nms = True
        self.tracker.half = False
        self.tracker.load_model()

    def run(self):
        print("Start Tracking Thread")
        self.thread_activate = True
        fps = 0
        count = 0
        old_time = time.time()
        while self.thread_activate:
            if time.time() - old_time > 1:
                fps = count
                count = 0
                old_time = time.time()
            if self.capture_queue.empty():
                self.msleep(1)
                continue
            count += 1
            image = self.capture_queue.get()
            track_id = self.tracker.track(image)
            for key, value in track_id.items():
                x1, y1, x2, y2 = value[:4]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, str(key), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if self.tracking_queue.empty():
                self.tracking_queue.put(image)                

    def stop(self):
        self.thread_activate = False
        print("Stop Tracking Thread")
