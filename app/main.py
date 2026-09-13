from fastapi import FastAPI,File, UploadFile, Depends, Path, HTTPException, BackgroundTasks,Request
from fastapi import status
from fastapi.staticfiles import StaticFiles
from functools import partial
from dataclasses import dataclass, field
from pydantic import BaseModel
import os
import sys

# legacy 디렉터리를 path에 넣어 utils, tasks 등 로컬 모듈 로드 가능하게
_app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _app_dir)

import asyncio
import requests
import subprocess
import shlex
import logging
logger = logging.getLogger(__name__)
from utils.db import get_db
from utils.common import parse_rtsp_url
## Thread ##
from threading import Thread, Event
import multiprocessing
## typehint ##
from typing import Optional, List
## post 허용 ##
from starlette.middleware.cors import CORSMiddleware 
from fastapi_utils.tasks import repeat_every
## 환경변수
from dotenv import load_dotenv
from pathlib import Path
env_path = Path("/app/.env") if Path("/app/.env").exists() else Path(_app_dir) / ".." / ".env"
load_dotenv(dotenv_path=env_path)
import json
app = FastAPI()
import run_appsrc
import traceback
import requests
import time
## redis
import redis as redis_lib
# redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
from celery import Celery
import uuid
from celery.result import AsyncResult
import socket
# from tasks import start_celery_task, start_all_celery_task, bg_celery,   # tasks.py에 정의한 작업 가져오기
from tasks import live_celery, test # tasks.py에 정의한 작업 가져오기  
## NOTE: start_all_background / start_all_celery_task / start_celery_task 는 원본에서 주석 처리된 상태여서 제외함.
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
import httpx
# 템플릿/정적 경로 (legacy/ 기준)
templates_directory = os.path.join(_app_dir, "templates")
static_directory = os.path.join(_app_dir, "templates", "static")
app.mount("/static", StaticFiles(directory=static_directory), name="static")
# Jinja2Templates 인스턴스 생성
templates = Jinja2Templates(directory=templates_directory)
origins = [
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Celery 애플리케이션을 초기화합니다.
celery_app = Celery('app', broker=os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/0'), backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/1'))
celery_app.conf.update(
    result_expires=3600,  # Set the time to keep work results (e.g. 1 hour)
    worker_pool='gevent', #eventlet, gevent
    worker_concurrency=100, 
    broker_pool_limit=None,
    task_acks_late=True,
    broker_heartbeat=None,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    worker_send_task_events=True,
    task_send_sent_event=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Seoul",
    enable_utc=False,
    # prefetch_multiplier=1
)

##CCTV ID CLASS
class Item(BaseModel):
    # START_STOP: str
    c_id: List[str]
    ai_on: List[bool]

class EventItem(BaseModel):
    # site_code: str
    # site_name: str
    cctv_id: str
    video_source_id: str
    updated_time: str
    record_video: str
    thumbnail: str
    ev_type: str
    # work_area: List[float]
    # danger_area: List[float]

## REDIS 정보 가져오기
async def get_redis():
    host = os.getenv("REDIS_HOST", "redis")
    redis_instance = redis_lib.Redis(host=host, port=6379, db=0, encoding="utf-8")
    return redis_instance

def get_host_ip():
    host_ip = os.getenv('HOST_IP')
    # host_ip = "test_ip"
    return host_ip

async def get_common_info():
    server_id=get_host_ip()
    common_info = {
        "server_id": server_id,
        "server_name":"AI CCTV 서버",
                }
    return common_info

## APP 첫시작
@app.on_event("startup")
async def startup_event():
    redis_instance = await get_redis()
    tasklist_dict = redis_instance.get("tasklist")
    if tasklist_dict is None:
        tasklist = {}
        serialized_tasklist = json.dumps(tasklist)
        redis_instance.set("tasklist",serialized_tasklist)

# app.mount("/static", StaticFiles(directory="fastapi/static"), name="static")
@app.get("/streaming/{video_id}")
async def read_root(request: Request,video_id: str,common_info: dict = Depends(get_common_info)):
    video_id = video_id  # 실제 비디오 ID로 변경
    return templates.TemplateResponse("index.html", {"request": request, "video_id": video_id, "common_info": common_info})

HLS_DIRECTORY = "/data/playlist/"
@app.get("/hls/{video_id}/playlist.m3u8")
async def get_playlist(video_id: str):
    playlist_path = os.path.join(HLS_DIRECTORY,"streaming", video_id, "index.m3u8")
    if os.path.exists(playlist_path):
        return FileResponse(playlist_path, media_type='application/x-mpegURL')
    raise HTTPException(status_code=404, detail="Playlist not found")

@app.get("/hls/{video_id}/{segment}")
async def get_segment(video_id: str, segment: str):
    segment_path = os.path.join(HLS_DIRECTORY,"streaming", video_id, segment)
    if os.path.exists(segment_path):
        return FileResponse(segment_path, media_type='video/MP2T')
    raise HTTPException(status_code=404, detail="Segment not found")

@app.get("/CCTVLIST/")
async def CCTVLIST(common_info: dict = Depends(get_common_info)):
    try:
        db_manager = get_db.DBManager()
        variable_dict = await db_manager.GET_CCTV_LIST()
        camera_info = []
        for item in variable_dict:
            cctv_id = f"{item[0]}"  
            origin_url = f"{item[1]}"
            admin, password, parsed_url = parse_rtsp_url(origin_url)
            ai_type = await db_manager.get_senario_list(cctv_id)
            ai_type_flat = [item for sublist in ai_type for item in sublist]
            stremaing_url = f"http://{common_info['server_id']}:1223/streaming/{cctv_id}",
            if item[2] == 1:
                monitoring_check = "연결됨"
            else:
                monitoring_check = "연결불가"
            update_at = f"{item[3]}"
            camera_info.append({
                "id": cctv_id,
                "status": monitoring_check,  # 상태, 실제 상황에 맞게 조정
                "update_at": update_at,  # 마지막 활성 시간, 실제 값 필요
                "origin_url": parsed_url,  # 원본 URL
                "cctvadmin": admin,
                "cctvpassword": password,
                "ai_type": ai_type_flat,
                "stremaing_url":stremaing_url,
            })
        return {"common": common_info, "camera_info": camera_info}, status.HTTP_200_OK
    except Exception as e:
        # Log the actual error message for debugging
        print(f"Error: {str(e)}")  # Consider using a logger in real applications
        raise HTTPException(status_code=500, detail="CCTV가 없습니다.")
    
## 현재 실행중인 CCTV 체크
@app.get("/CURRENTLY_WORKING/")
async def CURRENTLY_WORKING(redis_instance: redis_lib.Redis = Depends(get_redis)):
    try:
        db_manager = get_db.DBManager()
        tasklist_dict = redis_instance.get("tasklist")
        tasklist = json.loads(tasklist_dict)
        return {"message": "현재 작업중인 CCTVList.", "task_id_dict": tasklist}, status.HTTP_200_OK
    except Exception as e:
        raise HTTPException(status_code=500, detail="작업중인 CCTV가 없습니다.")
    
## CCTV START or STOP
@app.post("/live/{START_STOP}")
async def live(item: Item, START_STOP, redis_instance: redis_lib.Redis = Depends(get_redis)):
    try:
        tasklist_dict = redis_instance.get("tasklist")
        tasklist = json.loads(tasklist_dict)
        if tasklist_dict is None:
            raise TypeError
        tasklist= json.loads(tasklist_dict)
        dicted_item = dict(item)
        print("item",item)
        print("dicted_item",dicted_item)
        print("tasklist",tasklist)
        # START_STOP = dicted_item["START_STOP"]
        c_id = dicted_item["c_id"]
        ai_on = dicted_item["ai_on"]
        print("ai_on",ai_on)
        if START_STOP =="stop":
            # c_id = ["cctv" + str(item) if not str(item).startswith("cctv") else item for item in c_id]
            for cid in c_id:
                if cid in tasklist:
                    task_id = tasklist[cid]
                    result = celery_app.control.revoke(task_id, terminate=True)
                    del tasklist[cid]  # 태스크 ID를 tasklist에서 제거
                else:
                    print(f"Task ID for {cid} not found in tasklist.")
            tasklist = json.dumps(tasklist)
            redis_instance.set("tasklist",tasklist)
            return {"message": "Celery live 작업이 강제로 종료되었습니다."},status.HTTP_200_OK

            ## 시작 API 받으면
        elif START_STOP == "start":
            db_manager = get_db.DBManager()
            variable_dict = db_manager.GET_RTSP()
            print("variable_dict",variable_dict)
            print("c_id",c_id)
            # c_id = ["cctv" + str(item) if not str(item).startswith("cctv") else item for item in c_id]
            ## 이미 실행중인 CCTV 중지
            try:
                for cid in c_id:
                    task_id = tasklist[cid]
                    result = celery_app.control.revoke(task_id, terminate=True)
                    print(f"{cid} 끄고 재시작합니다...")
                time.sleep(0.2)
            except:
                pass
            # Celery 작업을 비동기로 실행
            req = list(zip(c_id, ai_on))
            print("req,",req)
            for cctv_id, ai_on in req:
                if cctv_id in variable_dict:
                    url=variable_dict[cctv_id]
                    result = live_celery.apply_async(args=[cctv_id,ai_on, url])
                    tasklist[cctv_id] = result.id
            tasklist = json.dumps(tasklist)
            print("tasklist start",tasklist)
            redis_instance.set("tasklist",tasklist)
        return {"message": "live 작업.", "task_id_dict": tasklist}, status.HTTP_200_OK
    except Exception as e:
        traceback_str = traceback.format_exc()
        line_number = traceback.extract_tb(e.__traceback__)[-1].lineno
        print(f"Exception occurred at line {line_number}:\n{traceback_str}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Exception occurred at line {line_number}:\n{traceback_str}")

##################################################    
## CCTV START or STOP
# @app.post("/VAControl//{START_STOP}")
# @app.get("/VAControl/{cctv_ipaddress}:{cctv_port}/cgi")
@app.get("/VAControl/cgi")
async def VAControl(cctv_ipaddress: str, cctv_port: int, analysis: str, redis_instance: redis_lib.Redis = Depends(get_redis),common_info: dict = Depends(get_common_info)):
    server_ip=common_info['server_id']
    print("cctv_ipaddress",cctv_ipaddress)
    print("cctv_port",cctv_port)
    print("analysis",analysis)
    print("server_ip",server_ip)
    try:
        tasklist_dict = redis_instance.get("tasklist")
        tasklist = json.loads(tasklist_dict)
        if tasklist_dict is None:
            raise TypeError
        tasklist= json.loads(tasklist_dict)
        print("tasklist",tasklist)
        db_manager = get_db.DBManager()
        variable_dict = await db_manager.GET_RTSP()
        cctv_url = cctv_ipaddress+":"+str(cctv_port)
        c_id = await db_manager.GET_CID_URL(cctv_url)
        c_id=str(c_id)
        url=variable_dict[c_id]
        result_json = {
            "camera_ip":f"{cctv_ipaddress}",
            "camera_port":f"{cctv_port}",
            "camera_analysis":f"{analysis}",
        }
        if analysis == "off":
            if c_id in tasklist:
                task_id = tasklist[c_id]
                result = celery_app.control.revoke(task_id, terminate=True)
                del tasklist[c_id]  # 태스크 ID를 tasklist에서 제거
            else:
                return {"message": f"{cctv_ipaddress}:{cctv_port} 실행중인 작업이 아닙니다."},status.HTTP_200_OK
            result = live_celery.apply_async(args=[c_id,analysis, url])
            tasklist[c_id] = result.id
            tasklist = json.dumps(tasklist)
            redis_instance.set("tasklist",tasklist)
            return result_json, status.HTTP_200_OK
            ## 시작 API 받으면
        elif analysis == "on":
            ## 이미 실행중인 CCTV 중지
            try:
                if c_id in tasklist:
                    task_id = tasklist[c_id]
                    result = celery_app.control.revoke(task_id, terminate=True)
                # Celery 작업을 비동기로 실행
                result = live_celery.apply_async(args=[c_id, url, analysis])
                tasklist[c_id] = result.id
                tasklist = json.dumps(tasklist)
                print("tasklist start",tasklist)
                redis_instance.set("tasklist",tasklist)
                return result_json, status.HTTP_200_OK
            except:
                pass
    except Exception as e:
        traceback_str = traceback.format_exc()
        line_number = traceback.extract_tb(e.__traceback__)[-1].lineno
        print(f"Exception occurred at line {line_number}:\n{traceback_str}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Exception occurred at line {line_number}:\n{traceback_str}")

@app.post("/request_event")
async def request_event(EventItem: EventItem, common_info: dict = Depends(get_common_info)):
    print("EventItem",EventItem)
    db_manager = get_db.DBManager()
    work_area = await db_manager.GET_COORDINATES(cctv_id=EventItem.cctv_id, dngtype="작업구역") ## 
    danger_area = await db_manager.GET_COORDINATES(cctv_id=EventItem.cctv_id, dngtype="위험구역") ## 

    event_item = {
        "common" : {
        "server_ip": common_info['server_id'],
        "server_name": common_info['server_name'],
        "site_code": "TEST",
        "site_name": "TEST",

        # "site_code": EventItem.site_code,
        # "site_name": EventItem.site_name,
        "cctv_id": EventItem.cctv_id,
        "video_source_id": EventItem.video_source_id,
        "updated_time": EventItem.updated_time,
        "record_video": EventItem.record_video,
        "thumbnail": f"{EventItem.thumbnail}"
        },
        "event": {
            "ev_type":  EventItem.ev_type, # 이벤트 타입
            "work_area": work_area,  # 작업 영역
            "danger_area": danger_area,  # 위험 영역
        }
    }
    ### 서비스 ID
    devId = f"ARTI-{EventItem.cctv_id}"
    if EventItem.ev_type == "PPE":
        serviceId= 100
    elif EventItem.ev_type == "FIRE":
        serviceId= 101
    elif EventItem.ev_type == "FALL_DOWN":
        serviceId= 102

    # 외부 관제 플랫폼 연동 엔드포인트는 환경변수로 주입 (형식: {base}?serviceId=&devId=)
    url = os.getenv("EXTERNAL_EVENT_API", "http://localhost")
    headers = {'Content-Type': 'application/json'}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=event_item, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return event_item
            # raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    return response.json()
