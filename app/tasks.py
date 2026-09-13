# tasks.py
import os
import sys

_app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _app_dir)

from celery import Celery
from celery.schedules import crontab
from utils.db import get_db
import run_appsrc
import traceback
# sys.path.append('../torchserve/serve/ts_scripts')  # 필요 시 경로 조정
import base64
app = Celery('app', broker=os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/0'), backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/1'))
app.conf.beat_schedule={
        'check_health_every_minute': {
            'task': 'legacy.tasks.check_health',  # 작업의 경로
            'schedule': crontab(minute='*/1'),  # 매 1분마다 실행
            'args': ()   # 작업에 전달할 인수
        },
}
import asyncio
import logging
from logging.handlers import RotatingFileHandler
# 파일 핸들러 추가
logger = logging.getLogger('celery')

file_handler = RotatingFileHandler(os.getenv('CELERY_LOG_FILE', '/var/log/celery/worker.log'), maxBytes=100000, backupCount=10)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

@app.task
def live_celery(cctv_id, ai_on,url):
    try:
        print(f"cctv_id: {cctv_id}, url: {url}")
        # cctv_id -> "cctv1"
        cctv_id = cctv_id.split("cctv")[-1]
        ai_on = ai_on.lower()
        if ai_on == "on":
            ai_on = True
        elif ai_on == "off":
            ai_on = False
        ch = run_appsrc.Gstfactory(CCTV_ID=f"{cctv_id}", ai_on=bool(ai_on),VIDEO_URL=url)
        ch.running()
        return {"message": "Started"}
    except Exception as e:
        traceback_str = traceback.format_exc()
        line_number = traceback.extract_tb(e.__traceback__)[-1].lineno
        print(f"Exception occurred at line {line_number}:\n{traceback_str}")
        return {"message": f"Exception occurred at line {line_number}:\n{traceback_str}"}    
        
@app.task
def test(cctv_no, cctv_id, url):
    try:
        print(f" cctv_no: {cctv_no}, cctv_id: {cctv_id}, url: {url}")
        # cctv_id -> "cctv1"
        cctv_id = cctv_id.split("cctv")[-1]
        ch = run_appsrc.Gstfactory(VIDEO_URL=url, CCTV_ID=f"{cctv_id}", CCTV_NO=cctv_no, is_process_running=True)
        ch.test()
        return {"message": "Started"}
    except Exception as e:
        traceback_str = traceback.format_exc()
        line_number = traceback.extract_tb(e.__traceback__)[-1].lineno
        print(f"Exception occurred at line {line_number}:\n{traceback_str}")
        return {"message": f"Exception occurred at line {line_number}:\n{traceback_str}"}    

@app.task
def check_health():
    db_manager = get_db.DBManager()
    async def perform_health_check():
        # URL 리스트를 한 번만 가져옵니다.
        cctv_list = await db_manager.GET_CCTV_LIST()
        if cctv_list is None:
            raise ValueError("Failed to retrieve CCTV list")
        url_list = [item[1] for item in cctv_list]  # 리스트에서 URL만 추출합니다.
        # health_check 함수에 url_list를 전달합니다.
        await db_manager.health_check(url_list)
    # 비동기 함수를 실행
    asyncio.run(perform_health_check())