from threading import Thread
import threading
import numpy as np
import cv2
import torch
import os
import shutil
from distutils.dir_util import copy_tree
from datetime import date, datetime
from dataclasses import dataclass, field
import multiprocessing
import logging
logger = logging.getLogger()
import httpx
from json import dumps
import time
import argparse
from fractions import Fraction
import numpy as np
from gstreamer import GstContext, GstPipeline, GstApp, Gst, GstVideo, GLib, GstVideoSink
import gstreamer.utils as gst_utils
####
from typing import Tuple, List
import sys
import os
import ai
from torchvision.transforms import Compose
from collections import Counter
import traceback

from utils.db import get_db
from utils.record import h264codec
import asyncio
from pymysql.connections import Connection
                
@dataclass
class VideoRecorder:
    save_location: str  # 비디오 저장 경로
    danger_type: str    # 위험 타입
    max_duration_sec: int  # 최대 녹화 지속 시간(초)
    fps: int  # 비디오 프레임 속도
    width: int  # 비디오 프레임 너비
    height: int  # 비디오 프레임 높이
    start_time: datetime = None  # 녹화 시작 시간
    frame_writer: cv2.VideoWriter = None  # 프레임 레코더 객체
    is_recording: bool = True  # 녹화 중 여부
    frame_count: int = 0  # 현재까지 녹화한 프레임 수
    
    def __post_init__(self):
        self.max_frame_count = self.max_duration_sec * self.fps
        self.creation_time =int(time.time())
        
    # @staticmethod
    def save_info(self, frame):
        self.videoname = f"{self.save_location}_{self.danger_type}"
        frame = cv2.imwrite(f"{self.videoname}.jpg", frame)
        frame = cv2.imread(f"{self.videoname}.jpg")
        frame = cv2.resize(frame, (self.width//2, self.height//2), interpolation=cv2.INTER_LINEAR)
        frame = cv2.imwrite(f"{self.videoname}_thumb.jpg",frame)

    def cleanup(self) -> None:
        if self.is_recording:
            self.stop_recording()
        if self.frame_writer is not None:
            self.frame_writer.release()

    def start_recording(self) -> None:
        if not self.is_recording:
            self.start_time = datetime.now()
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.frame_writer = cv2.VideoWriter(self.videoname+".mp4", self.fourcc, self.fps, (self.width, self.height))
            self.is_recording = True
            self.frame_count = 0

    def stop_recording(self) -> None:
        if self.is_recording:
            self.frame_writer.release()
            self.is_recording = False
            self.cleanup()

    def record(self, frame: np.array) -> None:
        if self.is_recording:
            self.frame_writer.write(frame)
            self.frame_count += 1
            if self.frame_count >= self.max_frame_count:
                self.stop_recording()
                asyncio.run(h264codec.convert(f"{self.videoname}"+".mp4"))
                
    def check_elapsed_time(self) -> int:
        current_time = int(time.time())
        elapsed_time = current_time - self.creation_time
        return elapsed_time
@dataclass
class Gstfactory:
    """ STREMMING FUNCTION """
    CCTV_ID: str # cctvid
    VIDEO_URL: str  # 비디오 URL
    ai_on: bool = True  # AI 사용 여부
    ts_len: int = 10  ### Save time unit of ts file
    playlist_dir: str = "/data/playlist/" ### Save video BaseDir
    VIDEO_FORMAT: str = "RGB"
    # SETTING
    public_threshold = 0.5
    dl_threshold = 0.5
    ppe_threshold = 0.1
    min_box_ppe = 10
    min_box_signalman = 100
    red_color = (0, 0, 255)
    green_color = (0, 255, 0)
    blue_color = (255, 0, 0)
    orange_color = (0,165,255)
    white_color = (255,255,255)

    hardhat_Record=None
    recordangerArea = None
    Fall_Record = None
    prev_vehicle_positions = []
    new_vehicle_positions = []
    no_signalman_count = 0
    status_check_time =None
    def __post_init__(self):
        print("Gstfactory init")
        self.db_manager = get_db.DBManager()
        self.streaming_dir, self.accident_dir = gst_utils.make_save_dir(self.CCTV_ID)
        # Real CCTV load
        self.tdof = int(self.ts_len / 5)
        self.cap = cv2.VideoCapture(self.VIDEO_URL)
        if not self.cap.isOpened():
            logger.info("Error opening video file.")
        self.GST_VIDEO_FORMAT = GstVideo.VideoFormat.from_string(self.VIDEO_FORMAT)
        # 실제 CCTV 해상도 크기
        self.WIDTH: int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.HEIGHT: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.WIDTH > 1280:                                         
            self.WIDTH = 1280
            self.HEIGHT = 720
        if self.ai_on:
            self.AImodel = ai.StreamingAIPipeline(CCTV_ID=self.CCTV_ID, width=self.WIDTH, height=self.HEIGHT, version="normal")
            self.model1, self.model2, self.model3 = self.AImodel.loadmodel()
            if self.AImodel.get_web_area1:
                self.working_area = [(x * self.WIDTH, y * self.HEIGHT) for x, y in self.AImodel.working_area]
            else:
                self.working_area = [(0,0),(self.WIDTH,0),(self.WIDTH,self.HEIGHT),(0,self.HEIGHT)]
            if self.AImodel.get_web_area2:
                self.danger_area = [(x * self.WIDTH, y * self.HEIGHT) for x, y in self.AImodel.danger_area]
            else:
                self.danger_area = None
            # load 시나리오
            # senario_list = self.db_manager.get_senario_list(self.CCTV_ID)
            # print("senario_list",senario_list)
            # if "PPE" in senario_list:
            #     self.check_ppe_module = True
            # if "DISTANCE" in senario_list:
            #     self.check_distance_module = True
            # if "FALL" in senario_list:
                # self.check_fall_module = True
            self.check_ppe_module = True
            self.check_distance_module = True
            self.check_fall_module = True
            self.check_signalman_module = True
            global hardhat_frame
            hardhat_frame = None
        
    ## 프레임 추론 
    def inf(self,frame):
        # logger.info("inf start")
        frame=cv2.resize(frame, (self.WIDTH, self.HEIGHT))
        save_location=f"{self.accident_dir}"+"/"f"{self.current_time_string}"
        self.org_img = frame.copy()
        self.AImodel.detect(frame)
        danger_bboxes, worker_bboxes = self.AImodel.check_number_worker_in_danger_area(self.working_area, self.danger_area)
        # frame = self.AImodel.visualize_workers(frame, worker_bboxes, self.white_color, 'worker')
        ## worker in recordangerArea
        # logger.info(f"danger_bboxes -> {danger_bboxes}")
        # logger.info(f"worker_bboxes -> {worker_bboxes}")
        ### 화재 감시 ###
        fire_bboxes= None
        if fire_bboxes:
            frame = self.AImodel.visualize_workers(frame, fire_bboxes, self.red_color, 'Fire')
            ## recording fire type
            if  self.recordFire == None or self.recordFire.check_elapsed_time() > 60:
                danger_type = "fire_detected"
                logger.info(f"recordFire -> {self.recordFire}")
                self.recordFire = VideoRecorder(save_location=save_location,danger_type="fire_detected",max_duration_sec=10, fps=10,width=self.WIDTH,height=self.HEIGHT)
                self.recordFire.save_info(frame)
                asyncio.run(self.db_manager.PUT_EVENT(event_time=self.current_time_string,
                                                          event_type="화재감지",
                                                          event_name=self.recordFire.videoname,
                                                          cctv_id=self.CCTV_ID))
        #######
        if danger_bboxes:
            # print(danger_bboxes)
            frame = self.AImodel.visualize_workers(frame, danger_bboxes, self.red_color, 'danger')
            ## recording recordangerarea type
            if  self.recordangerArea == None or self.recordangerarea.check_elapsed_time() > 60:
                danger_type = "danger"
                logger.info(f"recordangerArea -> {self.recordangerArea}")
                self.recordangerArea = VideoRecorder(save_location=save_location,danger_type="danger_area_entry",max_duration_sec=10, fps=10,width=self.WIDTH,height=self.HEIGHT)
                self.recordangerArea.save_info(frame)
                asyncio.run(self.db_manager.PUT_EVENT(event_time=self.current_time_string,
                                                          event_type="위험구역출입",
                                                          event_name=self.recordangerArea.videoname,
                                                          cctv_id=self.CCTV_ID))
            
        ## 안정장비 시나리오
        if self.check_ppe_module:
            # logger.info("check_ppe_module")
            worker_bboxes, hardhat_on, hardhat_off, harness_on, harness_off = self.AImodel.check_PPE(self.min_box_ppe,
                                                                                                     self.working_area)
            # print("worker_bboxes, hardhat_on, hardhat_off, harness_on, harness_off",worker_bboxes, hardhat_on, hardhat_off, harness_on, harness_off)
            # frame = self.AImodel.visualize_workers(frame, worker_bboxes, self.blue_color, 'worker')
            frame = self.AImodel.visualize_workers(frame, hardhat_on, self.green_color, 'hardhat_on')
            frame = self.AImodel.visualize_workers(frame, hardhat_off, self.red_color, 'hardhat_off')
            if len(hardhat_off):
                # if self.hardhat_Record is None and (not hasattr(self, 'nohardhat_log')   or (time.time() - self.nosignal_log) > 60):
                if self.hardhat_Record == None or self.hardhat_Record.check_elapsed_time() > 60:
                    self.hardhat_Record = VideoRecorder(save_location,"without_hardhat",max_duration_sec=10,fps=10,width=self.WIDTH,height=self.HEIGHT)
                    self.hardhat_Record.save_info(frame)
                    self.hardhat_Record.start_recording()
                    asyncio.run(self.db_manager.PUT_EVENT(event_time=self.current_time_string,
                                                          event_type="장비미착용",
                                                          event_name=self.hardhat_Record.videoname,
                                                          cctv_id=self.CCTV_ID))
            # if len(harness_off):
            #     if self.harness_Record is None:
            #         self.harnessRecorder = VideoRecorder(save_location=self.videoname,max_duration_sec=10,fps=10,width=self.WIDTH,height=self.HEIGHT)
            #         self.harnessRecorder.start_recording()
            #         asyncio.run(self.db_manager.PUT_EVENT(current_time_string, "장비미착용", f"{self.current_time_string}_without_harness.jpg", self.videoname, self.CCTV_ID))

        ## 신호수 시나리오
        if self.check_signalman_module:
            # logger.info("check_signalman_module")
            (final_worker_bboxes, final_signalman_bboxes, final_vehicle_bboxes,
            self.no_signalman_count, self.vehicle_positions, self.new_vehicle_positions) = self.AImodel.check_signalman(
                                                                                                    self.min_box_signalman,
                                                                                                    self.working_area,
                                                                                                    self.prev_vehicle_positions, 
                                                                                                    self.new_vehicle_positions, 
                                                                                                    self.no_signalman_count)
            # if self.no_signalman_count > 5:
            #     if time.time() - VideoRecorder.save_info.gen_time(frame, self.current_time_string,"danger",WIDTH,HEIGHT) > 60 or recordangerarea == None:
            #         no_signalman = VideoRecorder(save_location=self.current_time_string+"_danger.mp4",max_duration_sec=10, fps=10,width=self.WIDTH,height=self.HEIGHT)
            #     self.no_signalman_count = 0
            #     recording_list.append(no_signalman_reocrd)
            #     # print(f"Warning: Vehicle moving without a signalman!")
            #     cv2.putText(frame, f"Warning: Vehicle moving without a signalman!",
            #                 (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, self.red_color, 3)
            # visualize
            frame = self.AImodel.visualize_workers(frame, final_signalman_bboxes, self.orange_color, 'signalman')
            frame = self.AImodel.visualize_workers(frame, final_vehicle_bboxes, self.red_color, 'vehicle')
            
        if final_worker_bboxes:
            frame = self.AImodel.visualize_workers(frame, final_worker_bboxes, self.white_color, 'worker')
        else:
            frame = self.AImodel.visualize_workers(frame, worker_bboxes, self.white_color, 'worker')

        ###record hardhat_Record
        try:
            if not self.hardhat_Record.is_recording:
                self.hardhat_Record.cleanup()
                self.hardhat_Record=None
            else:
                self.hardhat_Record.record(frame)
                logger.info("hardhatRecord.record(frame)")
        except:
            pass
        ###record recordangerArea
        try:
            if not self.recordangerArea.is_recording:
                self.recordangerArea.cleanup()
                self.recordangerArea=None
            else:
                self.recordangerArea.record(frame)
                logger.info("recordangerArea.record(frame)")
        except:
            pass
        
        return frame

    def running(self) -> None:
        ## 영상 FPS 가져오기
        self.video_fps=int(self.cap.get(cv2.CAP_PROP_FPS))
        self.purpose_fps :int = 10
        frame_interval = int(round(self.video_fps / self.purpose_fps))
        self.FPS: int = Fraction(self.purpose_fps)
        ### 저장옵션
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = None
        FPS_STR = gst_utils.fraction_to_str(self.FPS)
        # 기본 설정
        # DEFAULT_CAPS:str = "video/x-h264,format={self.VIDEO_FORMAT},width={self.WIDTH},height={self.HEIGHT},framerate={FPS_STR}".format(**locals())
        DEFAULT_CAPS:str = "video/x-raw,format={self.VIDEO_FORMAT},width={self.WIDTH},height={self.HEIGHT},framerate={FPS_STR}".format(**locals())
        # 기본 파이프라인
        DEFAULT_PIPELINE = gst_utils.to_gst_string([
            "appsrc emit-signals=True is-live=True caps={DEFAULT_CAPS}".format(**locals()),
            "videoconvert",
            # "video/x-h264 format=I420",
            "video/x-raw, format=I420",
            "x264enc tune=zerolatency bitrate=1500 ",
            "mpegtsmux",
            f"hlssink location={self.streaming_dir}/segment-%05d.ts"
            ""
        ])
        command = DEFAULT_PIPELINE
        # 추가 설정 파이프라인에 등록
        command= command+f" playlist-location={self.streaming_dir}/index.m3u8 max-files=7 target-duration={self.tdof}"
        if gst_utils.parse_caps != None:
            args_caps = gst_utils.parse_caps(command)
        NUM_BUFFERS = 100
        WIDTH = int(args_caps.get("width", self.WIDTH))
        HEIGHT = int(args_caps.get("height", self.HEIGHT))
        FPS = Fraction(args_caps.get("framerate", self.FPS))
        GST_VIDEO_FORMAT = GstVideo.VideoFormat.from_string(
            args_caps.get("format",self.VIDEO_FORMAT))
        CHANNELS = gst_utils.get_num_channels(GST_VIDEO_FORMAT)
        DTYPE = gst_utils.get_np_dtype(GST_VIDEO_FORMAT)
        FPS_STR = gst_utils.fraction_to_str(self.FPS)
        CAPS = "video/x-raw,format={self.VIDEO_FORMAT},width={WIDTH},height={HEIGHT},framerate={FPS_STR}".format(**locals())
        # ip_address = self.get_ip_address()
        dbmanager = get_db.DBManager()
        server_ip = dbmanager.get_host_ip()
        URL = f"http://{server_ip}/"+self.streaming_dir.split("/data/")[-1]+"/index.m3u8"
        asyncio.run(self.db_manager.PUT_MONITOR(self.CCTV_ID,URL))
        logger.info("PUT_MONITOR")
        # in order to record First danger senario
        with GstContext():  # create GstContext (hides MainLoop)
            pipeline = GstPipeline(command)
            def on_pipeline_init(self):
                """Setup AppSrc element"""
                appsrc = self.get_by_cls(GstApp.AppSrc)[0]  # get AppSrc
                appsrc.set_property("format", Gst.Format.TIME)
                appsrc.set_property("block", True)
                appsrc.set_caps(Gst.Caps.from_string(CAPS))
            pipeline._on_pipeline_init = on_pipeline_init.__get__(pipeline)
            #try:
            pipeline.startup()
            appsrc = pipeline.get_by_cls(GstApp.AppSrc)[0]  # GstApp.AppSrc
            pts = 0  # buffers presentation timestamp
            duration = 10**9 / (self.FPS.numerator / self.FPS.denominator)  # frame duration
            target_margin = 20
            target_box = target_margin
            while True:
                try:
                    ret, frame =self.cap.read()
                    self.current_time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    self.number_frames = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if ret and frame_interval and self.number_frames:
                        if (self.number_frames % frame_interval == 0):
                            frame=cv2.putText(frame, f"{self.current_time_string}", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                            if self.ai_on:
                                frame = self.inf(frame)
                            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            # convert np.ndarray to Gst.Buffer
                            gst_buffer = gst_utils.ndarray_to_gst_buffer(frame)
                            # set pts and duration to be able to record video, calculate fps
                            pts += duration  # Increase pts by duration
                            gst_buffer.pts = pts
                            gst_buffer.duration = duration
                            # emit <push-butffer> event with Gst.Buffer
                            appsrc.emit("push-buffer", gst_buffer)
                            # emit <end-of-stream> event
                            if cv2.waitKey(1) > 0 :
                                break
                        else:
                            continue
                    else:
                        self.cap = cv2.VideoCapture(self.VIDEO_URL)
                except:
                    asyncio.run(self.db_manager.health_check(cctv_id=self.CCTV_ID))
            appsrc.emit("end-of-stream")
            logger.info("end-of-stream")
            while not pipeline.is_done:
                time.sleep(.1)
            pipeline.shutdown()
            