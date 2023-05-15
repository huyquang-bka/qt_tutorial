import cv2
import mediapipe as mp
import numpy as np
from .ocsort import OCSort


class FaceDetection:
    def __init__(self, min_detection_confidence=0.5, model_selection=1):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
    
    def setup_model(self):
        mp_face_detector = mp.solutions.face_detection
        self.detection = mp_face_detector.FaceDetection(
            min_detection_confidence=self.min_detection_confidence, model_selection=self.model_selection)

    def detect(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detection.process(rgb_image)
        bboxes = []
        print("Number of faces detected: ", len(results.detections))
        if results.detections:
            for id, detection in enumerate(results.detections):
                bbox = detection.location_data.relative_bounding_box

                H, W, _ = image.shape

                x, y, w, h = int(bbox.xmin * W), int(bbox.ymin *
                                                     H), int(bbox.width * W), int(bbox.height * H)
                bounding_box = [x, y, x + w, y + h, 0, detection.score[0]]
                bboxes.append(bounding_box)
        return bboxes


class OCTracking(FaceDetection):
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
    