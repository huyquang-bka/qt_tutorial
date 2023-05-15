import time
import cv2
import numpy as np
from .ocsort import OCSort


class Detection:
    def __init__(self):
        self.size = (320, 320)
        self.scale = 1 / 255.
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4
        self.weight_path = "Yolov4/lp.weights"
        self.cfg_path = "Yolov4/base_data_tiny.cfg"

    def setup_model(self):
        net = cv2.dnn.readNet(self.weight_path, self.cfg_path)
        # self.classes = self.get_classes()
        # self.output_layers = self.get_output_layers(self.net)
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(
            size=self.size, scale=self.scale, swapRB=True)

    def detect(self, frame):
        classes, scores, boxes = self.model.detect(
            frame, self.conf_threshold, self.nms_threshold)
        bboxes = []
        for (classid, score, box) in zip(classes, scores, boxes):
            x1, y1, w, h = box
            x2 = x1 + w
            y2 = y1 + h
            bboxes.append([x1, y1, x2, y2, classid, score])
        return bboxes


class OCTracking(Detection):
    def __init__(self, det_thresh=0.4, max_age=30, min_hits=3,
                 iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2, use_byte=False):
        super().__init__()
        self.tracker = OCSort(det_thresh=det_thresh, max_age=max_age, min_hits=min_hits,
                              iou_threshold=iou_threshold, delta_t=delta_t, asso_func=asso_func, inertia=inertia, use_byte=use_byte)

    def track(self, image):
        track_dict = {}
        bboxes = self.detect(image)
        dets_to_sort = np.empty((0, 6))
        for x1, y1, x2, y2, cls, conf in bboxes:
            dets_to_sort = np.vstack(
                (dets_to_sort, np.array([x1, y1, x2, y2, conf, cls])))

        tracked_det = self.tracker.update(dets_to_sort)
        if len(tracked_det) > 0:
            bbox_xyxy = tracked_det[:, :4]
            indentities = tracked_det[:, 4]
            categories = tracked_det[:, 5]
            for i in range(len(bbox_xyxy)):
                x1, y1, x2, y2 = list(
                    map(lambda x: max(0, int(x)), bbox_xyxy[i]))
                id = int(indentities[i])
                track_dict[id] = [x1, y1, x2, y2, categories[i]]
        return track_dict


if __name__ == "__main__":
    tracker = OCTracking()
    tracker.weight_path = "/Users/huyquang/huyquang/Company/desktop_dr/resources/weights/tiny_head_final_320.weights.weights"
    tracker.cfg_path = "/Users/huyquang/huyquang/Company/desktop_dr/resources/weights/tiny-custom.cfg"
    tracker.conf_threshold = 0.3
    tracker.setup_model()
    cap = cv2.VideoCapture(0)
    old_time = time.time()
    fps = 0
    while True:
        if time.time() - old_time > 1:
            old_time = time.time()
            print(fps)
            fps = 0
        ret, image = cap.read()
        if not ret:
            break
        fps += 1
        track_id = tracker.detect(image)
        for key, value in track_id.items():
            x1, y1, x2, y2, cls = value[:5]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{key}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Image", image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
