from datetime import datetime
from threading import Thread
from dataclasses import dataclass, field
import time
import torch
import numpy as np
# from mmdet.apis import async_inference_detector, inference_detector
# from mmdet.apis.inference import init_detector
import threading
import os
from typing import List, Tuple
from torchvision.transforms import Compose
import cv2
import sys
import traceback
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.db import get_db
from pymysql.connections import Connection
from ensemble_boxes import *
from typing import List, Optional

class CustomThread(Thread):
    # constructor
    def __init__(self,model,frame):
        # execute the base constructor
        Thread.__init__(self)
        self.model=model
        self.frame=frame
        # self.tmp = self.inference(self.model,self.frame)
    def print_thread_id(self):
        thread_id = threading.get_ident()
        print("Thread ID:", thread_id)

    def inference(self, model, frame) -> Tuple[List[List[int]], List[float], List[str]]:
        box_list = []
        score_list = []
        label_list = []
        bboxes, labels, _ = model(frame)
        for bbox, label_id in zip(bboxes, labels):
            box, score = bbox[0:4].astype(int), bbox[4]
            if score >= 0.4:
                box_list.append(box)
                score_list.append(score)
                label_list.append(label_id)
        return box_list, score_list, label_list
    
    def run(self) -> None:
        start = round(time.time(), 3)
        # box_list, score_list, label_list = self.inference(self.model, self.frame)
        self.box,self.score,self.label = self.inference(self.model, self.frame)

@dataclass
class StreamingAIPipeline:
    CCTV_ID:str
    width : int
    height: int
    FPS: int = None                     ## 영상 보내지는 FPS 임의 지정
    version:str = None                  ## model Version
    def __post_init__(self)-> None:
        # self.conn = get_db.get_conn()
        self.db_manager = get_db.DBManager()
        self.device0="cuda"
        self.device1="cuda"
        ### model weights loads ###
        if self.version == "test":
            self.version:str = "test"
        elif self.version == "normal":
            pass
        else:
            self.version:str = "latest" 
        self.m1_path: str = os.getenv("TRT_ENGINE_PPE", "/tensorrt/ppe")
        self.m2_path: str = os.getenv("TRT_ENGINE_DL", "/tensorrt/dl")
        self.m3_path: str = os.getenv("TRT_ENGINE_PUBLIC", "/tensorrt/public")
        self.get_web_area1 = self.db_manager.GET_COORDINATES(cctv_id=self.CCTV_ID, dngtype="작업구역") ## 
        self.get_web_area2 = self.db_manager.GET_COORDINATES(cctv_id=self.CCTV_ID, dngtype="위험구역") ## 일은 가능
        print("get_web_area1",self.get_web_area1)
        try:
            self.working_area: List[Tuple[float, float]] = [(self.get_web_area1[i], self.get_web_area1[i+1]) for i in range(0, len(self.get_web_area1), 2)]
            print("working_area",self.working_area)
        except:
            pass
        try:
            self.danger_area: List[Tuple[float, float]] = [(self.get_web_area2[i], self.get_web_area2[i+1]) for i in range(0, len(self.get_web_area2), 2)]
        except:
            pass
        
    def loadmodel(self):
        from mmdeploy_runtime import Detector
        self.model1: str = Detector(self.m1_path, device_name=self.device0, device_id=0)
        self.model2: str = Detector(self.m2_path, device_name=self.device0, device_id=0)
        self.model3: str = Detector(self.m3_path, device_name=self.device0, device_id=0)
        return self.model1, self.model2, self.model3
        
    def detect(self, frame: np.ndarray) -> bool:
        self.frame = frame
        self.thread1 = CustomThread(model=self.model1,frame=self.frame)
        self.thread2 = CustomThread(model=self.model2,frame=self.frame)
        self.thread3 = CustomThread(model=self.model2,frame=self.frame)
        self.thread1.start()
        self.thread2.start()
        self.thread3.start()
        self.thread1.join()
        self.thread2.join()
        self.thread3.join()
        return True
    
    def filter_class(self, raw_bboxes: List[np.ndarray], raw_labels: List[int], raw_scores: List[int], class_names: List[int],prefix='worker') -> None:
        bboxes = []
        labels = []
        scores = []
        string_labels = []
        for i, box in enumerate(raw_bboxes):
            if raw_labels[i] in class_names:
                bboxes.append(box)
                labels.append(raw_labels[i])
                scores.append(raw_scores[i])
        return bboxes, labels, scores
    
    def compute_iou(self, box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
        x11, y11, x12, y12 = box1
        x21, y21, x22, y22 = box2
        xA = max(x11, x21)
        yA = max(y11, y21)
        xB = min(x12, x22)
        yB = min(y12, y22)
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
        boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    
    def extract_boxes_fusion(self, boxes: List[Tuple[float, float, float, float]], 
                         scores: List[float], 
                         labels: List[str], 
                         extracted_classes: List[str]) -> Tuple[List[Tuple[float, float, float, float]], List[float], List[int]]:
       
        extracted_boxes = []
        extracted_scores = []
        extracted_labels = []
        for i, box in enumerate(boxes):
            if labels[i] in extracted_classes:
                normalized_box = [box[0] / self.frame.shape[1], box[1] / self.frame.shape[0], box[2] / self.frame.shape[1], box[3] / self.frame.shape[0]]
                extracted_boxes.append(normalized_box)
                extracted_scores.append(scores[i])
                extracted_labels.append(labels[i])
        # extracted_labels = [int(label.split('_')[1]) for label in extracted_labels]
        return extracted_boxes, extracted_scores, extracted_labels
    
    def ensemble_bboxes(self, boxes_list: List[List[Tuple[float, float, float, float]]], 
                    scores_list: List[List[float]], 
                    labels_list: List[List[str]]) -> Tuple[List[Tuple[float, float, float, float]], List[float], List[str]]:
        weights = [1, 1]
        iou_thr = 0.5
        skip_box_thr = 0.0001
        sigma = 0.1
        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        # org boxes are normalized
        boxes = [[box[0] * self.frame.shape[1], box[1] * self.frame.shape[0], box[2] * self.frame.shape[1], box[3] * self.frame.shape[0]] for box in boxes]
        return boxes, scores, labels
  
    def is_point_inside_polygon(self, x: float, y: float, danger_points: List[Tuple[float, float]]) -> bool:
        n = len(danger_points)
        inside = False
        p1x, p1y = danger_points[0]
        for i in range(n + 1):
            p2x, p2y = danger_points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def detect_workers(self):
        public_worker_bboxes, public_worker_labels, public_worker_scores = self.filter_class(self.thread3.box, 
                                                                                        self.thread3.label, 
                                                                                        self.thread3.score,
                                                                                        [0], prefix='class')
        
        public_worker_bboxes, public_worker_scores, public_worker_labels = self.extract_boxes_fusion(public_worker_bboxes,
                                                                                            public_worker_scores,
                                                                                            public_worker_labels, [0])
        
        dl_worker_bboxes, dl_worker_labels, dl_worker_scores = self.filter_class(self.thread2.box, 
                                                                            self.thread2.label, 
                                                                            self.thread2.score,
                                                                            [0, 2], prefix='class')
        
        dl_worker_bboxes, dl_worker_scores, dl_worker_labels = self.extract_boxes_fusion(dl_worker_bboxes, dl_worker_scores, dl_worker_labels,
                                                                                [0,2])

        worker_bboxes, worker_scores, worker_labels = self.ensemble_bboxes([public_worker_bboxes, dl_worker_bboxes],
                                                                    [public_worker_scores, dl_worker_scores],
                                                                    [public_worker_labels, dl_worker_labels])
        
        # print("Workers:", worker_bboxes, worker_labels, worker_scores)
        return worker_bboxes, worker_labels, worker_scores
    
    def detect_signalman(self):
        # extract class_2: signalman
        signalman_bboxes, signalman_labels, signalman_scores = self.filter_class(self.thread2.box, 
                                                                            self.thread2.label, 
                                                                            self.thread2.score,
                                                                                [2], prefix='class')
        
        signalman_bboxes, signalman_scores, signalman_labels = self.extract_boxes_fusion(signalman_bboxes, signalman_scores, signalman_labels,
                                                                                [2])
        
        signalman_bboxes = [[box[0] * self.width, box[1] * self.height, box[2] * self.width, box[3] * self.height] for box in signalman_bboxes]
        # print("Signalman:", signalman_bboxes, signalman_labels, signalman_scores)
        return signalman_bboxes, signalman_labels, signalman_scores
    
    def detect_ppe(self):
        # extract class_3, class_4: ppe
        dl_ppe_bboxes, raw_dl_ppe_labels, dl_ppe_scores = self.filter_class(self.thread2.box, 
                                                                            self.thread2.label, 
                                                                            self.thread2.score,
                                                                            [3, 4], prefix='class')
        
        dl_ppe_bboxes, dl_ppe_labels, dl_ppe_scores = self.extract_boxes_fusion(dl_ppe_bboxes, dl_ppe_scores, raw_dl_ppe_labels,
                                                                           [3,4])
        
        # print("dl_ppe_bboxes, raw_dl_ppe_labels, dl_ppe_scores",dl_ppe_bboxes, raw_dl_ppe_labels, dl_ppe_scores)
        # dl_ppe_labels = [2 if x == 3 else 3 if x == 4 else x for x in raw_dl_ppe_labels]

        normaihub_ppe_bboxes, normaihub_ppe_labels, normaihub_ppe_scores = self.filter_class(self.thread1.box, 
                                                                                            self.thread1.label, 
                                                                                            self.thread1.score,
                                                                                            [0, 1, 2, 3], prefix='class')
        # print("normaihub_ppe_bboxes, normaihub_ppe_labels, normaihub_ppe_scores",normaihub_ppe_bboxes, normaihub_ppe_labels, normaihub_ppe_scores)
        normaihub_ppe_bboxes, normaihub_ppe_scores, normaihub_ppe_labels = self.extract_boxes_fusion(normaihub_ppe_bboxes,
                                                                                            normaihub_ppe_scores,
                                                                                            normaihub_ppe_labels,
                                                                                            [0, 1, 2, 3])
        
        # print("after normaihub_ppe_bboxes, normaihub_ppe_scores, normaihub_ppe_labels",normaihub_ppe_bboxes, normaihub_ppe_scores, normaihub_ppe_labels)
        
        ppe_bboxes, ppe_scores, ppe_labels = self.ensemble_bboxes([dl_ppe_bboxes, normaihub_ppe_bboxes],
                                                            [dl_ppe_scores, normaihub_ppe_scores],
                                                            [dl_ppe_labels, normaihub_ppe_labels])
        # print("PPE:", ppe_bboxes, ppe_labels, ppe_scores)
        return ppe_bboxes, ppe_labels, ppe_scores
    
    
    def check_number_worker_in_danger_area(self, detection_area: List[Tuple[float, float]],
                                          danger_area: Optional[List[Tuple[float, float]]]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:

        worker_bboxes, worker_labels, worker_scores = self.detect_workers()
        # CHECK NUMBER OF WORKER IN DANGER AREA
        workers_in_danger_area = []
        workers_in_detection_area = []
        for box in worker_bboxes:
            center_box_bottom = [(box[0] + box[2]) / 2, box[3]]
            if danger_area is not None:
                if self.is_point_inside_polygon(center_box_bottom[0], center_box_bottom[1], detection_area) and \
                        self.is_point_inside_polygon(center_box_bottom[0], center_box_bottom[1], danger_area):
                    workers_in_danger_area.append(box)
                elif self.is_point_inside_polygon(center_box_bottom[0], center_box_bottom[1], detection_area) and \
                        not self.is_point_inside_polygon(center_box_bottom[0], center_box_bottom[1], danger_area):
                    workers_in_detection_area.append(box)
            else:
                if self.is_point_inside_polygon(center_box_bottom[0], center_box_bottom[1], detection_area):
                    workers_in_detection_area.append(box)
                    
        return workers_in_danger_area, workers_in_detection_area
    
    def check_PPE(self, min_box_ppe: int, detection_area: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float, float, float]],
                                                                                              List[Tuple[float, float, float, float]],
                                                                                              List[Tuple[float, float, float, float]],
                                                                                              List[Tuple[float, float, float, float]],
                                                                                              List[Tuple[float, float, float, float]]]:
        # Detect worker
        worker_bboxes, worker_labels, worker_scores = self.detect_workers()
        # Detect PPE
        ppe_bboxes, ppe_labels, ppe_scores = self.detect_ppe()
        # dl_ppe_labels = []
        # print("ppe_bboxes, ppe_labels, ppe_scores",ppe_bboxes, ppe_labels, ppe_scores)
        # CHECK NUMBER OF WORKER IN DANGER AREA
        hardhat_on = []
        hardhat_off = []
        harness_on = []
        harness_off = []
        workers_box = []
        
        ppe_dict = {
            0: harness_on,
            0: harness_on,
            
            1: harness_off,
            2: hardhat_on,
            3: hardhat_off
        }
        min_box_ppe = 0
        if len(worker_bboxes):
            for index, box in enumerate(worker_bboxes):
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                min_box_ppe += box_area
            min_box_ppe = 0.06 * (min_box_ppe/(index+1) - 100)

        for box in worker_bboxes:
            center_box_bottom = [(box[0] + box[2]) / 2, box[3]]
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            # TODO: Check workers
            if self.is_point_inside_polygon(center_box_bottom[0], center_box_bottom[1], detection_area) and box_area > min_box_ppe:
                # print("check 2")
                workers_box.append(box)
                ## Check PPE
                for i, ppe_box in enumerate(ppe_bboxes):
                    # print("check 3")
                    x1_ppe, y1_ppe, x2_ppe, y2_ppe = ppe_box
                    ppe_box_area = (x2_ppe - x1_ppe) * (y2_ppe - y1_ppe)
                    iou = self.compute_iou(box, ppe_box)
                    # print("iou",iou)
                    if iou > 0:
                        ppe_dict.get(int(ppe_labels[i])).append(ppe_box)
                        
        return worker_bboxes, hardhat_on, hardhat_off, harness_on, harness_off
    
    def detect_vehicles(self):
        # extract class_7, class_1, class_5: vehicle
        public_vehicle_bboxes, public_vehicle_labels, public_vehicle_scores = self.filter_class( self.thread3.box, 
                                                                                            self.thread3.label, 
                                                                                            self.thread3.score,
                                                                                            [7], prefix='class')
        
        public_vehicle_bboxes, public_vehicle_labels, public_vehicle_scores = self.extract_boxes_fusion(public_vehicle_bboxes,
                                                                                            public_vehicle_scores,
                                                                                            public_vehicle_labels, [7])
        
        # print("public_vehicle_bboxes, public_vehicle_labels, public_vehicle_scores",public_vehicle_bboxes, public_vehicle_labels, public_vehicle_scores)
        dl_vehicle_bboxes, dl_vehicle_labels, dl_vehicle_scores = self.filter_class( self.thread2.box, 
                                                                            self.thread2.label, 
                                                                            self.thread2.score,
                                                                                [1, 5], prefix='class')

        dl_vehicle_bboxes, dl_vehicle_labels, dl_vehicle_scores = self.extract_boxes_fusion(dl_vehicle_bboxes,
                                                                                            dl_vehicle_scores,
                                                                                            dl_vehicle_labels, [1,5])

        vehicle_bboxes, vehicle_scores, vehicle_labels = self.ensemble_bboxes([public_vehicle_bboxes, dl_vehicle_bboxes],
                                                                        [public_vehicle_scores, dl_vehicle_scores],
                                                                        [public_vehicle_labels, dl_vehicle_labels])
        # print("Vehicles:", vehicle_bboxes, vehicle_labels, vehicle_scores)
        return vehicle_bboxes, vehicle_labels, vehicle_scores
    
    def check_signalman(self, min_box_signalman: int, detection_area: List[Tuple[float, float]],prev_vehicle_positions, new_vehicle_positions,no_signalman_count) -> Tuple[
            List[Tuple[float, float]],
            List[Tuple[float, float, float, float]],  # final_signalman_bboxes
            List[Tuple[float, float, float, float]],  # final_vehicle_bboxes
            int,  # no_signalman_count
            List[Tuple[int, int]],  # vehicle_positions
            List[Tuple[int, int]]  # new_vehicle_positions
        ]:
        # Detect workers
        worker_bboxes, worker_labels, worker_scores = self.detect_workers()
        # Detect vehicles
        vehicle_bboxes, vehicle_labels, vehicle_scores = self.detect_vehicles()
        # Detect signalman
        signalman_bboxes, signalman_scores, signalman_labels = self.detect_signalman()
        final_worker_bboxes = []
        final_signalman_bboxes = []
        final_vehicle_bboxes = []
        if len(worker_bboxes):
            for index, box in enumerate(worker_bboxes):
                x1, y1, x2, y2 = box
                box_area = (x2 - x1) * (y2 - y1)
                min_box_signalman += box_area
            min_box_signalman = min_box_signalman / (index + 1) - 100

        for box in worker_bboxes:
            x1, y1, x2, y2 = box
            box_area = (x2 - x1) * (y2 - y1)
            center_box_bottom = [(box[0] + box[2]) / 2, box[3]]
            if self.is_point_inside_polygon(center_box_bottom[0], center_box_bottom[1], detection_area)\
                    and box_area > min_box_signalman:
                
                for signalman_box in signalman_bboxes:
                    iou = self.compute_iou(box, signalman_box)
                    if iou > 0.1:
                        final_signalman_bboxes.append(signalman_box)
                        break
                final_worker_bboxes.append(box)

        for vehicle_box in vehicle_bboxes:
            # print("vehicle_box_all",vehicle_box)
            center_box_bottom = [(vehicle_box[0] + vehicle_box[2]) / 2, vehicle_box[3]]
            if self.is_point_inside_polygon(center_box_bottom[0], center_box_bottom[1], detection_area):
                final_vehicle_bboxes.append(vehicle_box)
                # Check if vehicle moving or not
                current_position = (int(center_box_bottom[0]), int(center_box_bottom[1]))
                x_center, y_center = current_position
                new_vehicle_positions.append(current_position)

                # Find the closest vehicle from the previous frame
                if prev_vehicle_positions:
                    distances = [math.sqrt((x - x_center) ** 2 + (y - y_center) ** 2) for x, y in prev_vehicle_positions]
                    min_distance_index = np.argmin(distances)
                    min_distance = distances[min_distance_index]

                    # Only consider the vehicle as moving if it has moved more than a threshold
                    # print("Vehicle moving: {}".format(min_distance))
                    if min_distance > 3:
                        has_signalman = any(signalman_bboxes)  # Check if the list is not empty
                        if not has_signalman:
                            no_signalman_count += 1
                        else:
                            no_signalman_count = 0
                    
        prev_vehicle_positions = new_vehicle_positions.copy()
        return final_worker_bboxes, final_signalman_bboxes, final_vehicle_bboxes, no_signalman_count, prev_vehicle_positions, new_vehicle_positions
    
    def visualize_detection_area(self, frame: np.ndarray, polygon_points: List[Tuple[int, int]], color: Tuple[int, int, int], thickness: int) -> np.ndarray:
        pts = np.array(polygon_points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, color, thickness)
        return frame
    
    def visualize_workers(self, frame: np.ndarray, workers_in_detection_area: List[Tuple[float, float, float, float]],
                        color: Tuple[int, int, int], text: str) -> np.ndarray:
        for box in workers_in_detection_area:
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 1)
            cv2.putText(frame, text, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1)
        return frame