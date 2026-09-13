import pymysql
import os
import asyncio
import traceback
## 환경변수
from dotenv import load_dotenv
from pathlib import Path
env_path = Path('/app/.env')
load_dotenv(dotenv_path=env_path)
import httpx
from urllib.parse import urlparse
import time
from dbutils.pooled_db import PooledDB
import logging
import hashlib
import re
import socket

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def exception_handler(func):
    async def wrapper(instance, *args, **kwargs):  
        try:
            return await func(instance, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            return None
    return wrapper

class DBManager:
    def __init__(self):
        self.pool = PooledDB(
            creator=pymysql,
            maxconnections=20,
            host=os.environ.get('DB_HOST'),
            port=int(os.environ.get('DB_PORT')),
            user=os.environ.get('DB_USER'),
            password=os.environ.get('DB_PASSWORD'),
            database=os.environ.get('DB_NAME'),
            charset='utf8'
        )
        self.last_health_check = 1

    def get_connection(self):
        return self.pool.connection()

    def execute(self, sql, params=None):
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                conn.commit()
                result = cursor.fetchall()
                return result
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error executing SQL: {e}")
        finally:
            conn.close()

    @exception_handler
    async def GET_COORDINATES(self, cctv_id, dngtype):
        sql = """SELECT ai_coordinate FROM dngnZone WHERE cctv_id = %s AND zone_type= %s;"""
        COORDINATEXY = self.execute(sql, (cctv_id, dngtype))
        coord_str = COORDINATEXY[0][0]
        coordinates_list = [float(coord) for coord in coord_str.split(',')]
        return coordinates_list
    
    @exception_handler
    async def GET_CCTV_LIST(self):
        sql = """SELECT id, origin_url, monitoring_check, updated_at FROM cctv;"""
        cctv_list = self.execute(sql)
        return cctv_list

    @exception_handler
    async def GET_RTSP(self):
        sql = """SELECT id, origin_url FROM cctv;"""
        URL = self.execute(sql)
        variable_dict = {}
        for result in URL:
            id_value, url = result
            variable_name = str(id_value)
            variable_dict[variable_name] = url
        return variable_dict

    @exception_handler
    async def PUT_MONITOR(self, cctv_id, URL):
        raise NotImplementedError
        select_query = "SELECT cctv_id FROM cctv_monitoring WHERE cctv_id = %s"
        row = self.execute(select_query, (cctv_id,))
        current_time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        if row:
            sql = """
                    UPDATE cctv_monitoring
                    SET updated_at = %s,
                    url = %s 
                    WHERE cctv_id = %s
                    """
            self.execute(sql, (current_time_string, URL, cctv_id))
        else:
            sql = """
                    INSERT INTO cctv_monitoring (cctv_id, created_at, updated_at, url, health_check)
                    VALUES (%s, %s, %s, %s, %s)
                    """
            health_check = 1
            self.execute(sql, (cctv_id, current_time_string, current_time_string, URL, health_check))

    @exception_handler
    async def GET_MTID_CCTVID(self, cctv_id):
        logger.info('GET_MTID_CCTVID')
        sql = """
        SELECT id
        FROM cctv_monitoring        
        WHERE cctv_id = %s;
        """
        mt_id = self.execute(sql, (cctv_id,))
        return mt_id

    @exception_handler
    async def GET_CID_URL(self, cctv_url):
        logger.info('GET_CID_URL')
        search_string = f"%{cctv_url}%"
        sql = """
        SELECT id
        FROM cctv     
        WHERE origin_url LIKE %s;
        """
        cid = self.execute(sql, (search_string,))
        cid = cid[0][0]
        return cid
    
    @exception_handler
    async def get_senario_list(self, cctv_id):
        logger.info('get_senario_list')
        sql = """
        SELECT ai_model.name
        FROM cctv_monitoring
        JOIN using_model ON cctv_monitoring.id = using_model.cctv_monitoring_id
        JOIN ai_model ON using_model.ai_model_id = ai_model.id
        WHERE cctv_monitoring.cctv_id = %s;
        """
        senario_list = self.execute(sql, (cctv_id,))
        return senario_list
    
    @exception_handler
    async def health_check(self, url_list):
        if not url_list:
            raise ValueError("URL list is empty")
        sql = """
        UPDATE cctv_monitoring
        JOIN cctv ON cctv_monitoring.cctv_id = cctv.id
        SET cctv_monitoring.health_check = %s
        WHERE cctv.origin_url = %s;
        """
        for url in url_list:
            try:
                parsed_url = urlparse(url)
                if not parsed_url.scheme or not parsed_url.netloc:
                    raise ValueError(f"Invalid URL format: {url}")
                host = parsed_url.hostname
                port = parsed_url.port if parsed_url.port else 554  # Default RTSP port is 554
                username = parsed_url.username
                password = parsed_url.password
                if username is None or password is None:
                    raise ValueError(f"Missing username or password in URL: {url}")
                is_healthy, response_message = await self.check_rtsp_connection(host, port, url, username, password)
                logger.debug(f"response_message, {response_message}")
    
                health_check = 1 if is_healthy else 0
            except Exception as e:
                health_check = 0  # Set health check to 0 in case of an exception
            
            self.execute(sql, (health_check, url))
            logger.debug(f"response_message, {response_message}")
        return True
    
    # @exception_handler
    # async def health_check(self, url_list):
    #     if not url_list:
    #         raise ValueError("URL list is empty")
    #     sql = """
    #     UPDATE cctv_monitoring
    #     JOIN cctv ON cctv_monitoring.cctv_id = cctv.id
    #     SET cctv_monitoring.health_check = %s
    #     WHERE cctv.origin_url = %s;
    #     """
    #     for url in url_list:
    #         try:
    #             parsed_url = urlparse(url)
    #             if not parsed_url.scheme or not parsed_url.netloc:
    #                 raise ValueError(f"Invalid URL format: {url}")
    #             host = parsed_url.hostname
    #             port = parsed_url.port if parsed_url.port else 554  # Default RTSP port is 554
    #             username = parsed_url.username
    #             password = parsed_url.password
    #             if username is None or password is None:
    #                 raise ValueError(f"Missing username or password in URL: {url}")

    #             is_healthy = await self.check_rtsp_connection(host, port, url, username, password)
    #             health_check = 1 if is_healthy else 0
    #         except Exception as e:
    #             logger.error(f"Exception during RTSP health check for URL {url}: {e}")
    #             health_check = 0  # Set health check to 0 in case of an exception
            
    #         self.execute(sql, (health_check, url))
    #     return True


    @exception_handler
    async def PUT_EVENT(self, event_time, event_type, event_name, cctv_id):
        try:
            event_name = event_name.split("/data")[-1]
            event_vid = f"{event_name}.mp4"
            event_img = f"{event_name}.jpg"
            event_thumb = f"{event_img.split('.')[0]}_thumb.jpg"
            ipt = (event_time, event_time, event_type, "1", event_vid, cctv_id, event_thumb, "0")
            sql = """INSERT INTO event (created_at, updated_at, event_type, number, event_vid, cctv_id, thumbnail, is_checked) VALUES (%s,%s,%s,%s,%s,%s,%s,%s);"""
            self.execute(sql, ipt)
            event_data = {
                # "site_code": "1",
                # "site_name": "1",
                "cctv_id": cctv_id,
                "video_source_id": "1",
                "updated_time": event_time,
                "record_video":event_vid,
                "thumbnail":event_thumb,
                "ev_type": event_type,
                # "work_area": work_area,
                # "danger_area": danger_area,
            }
            await self.send_event_request(event_data)
        except Exception as e:
            logger.error(f"Exception during PUT_EVENT: {e}")
            traceback.print_exc()

    @exception_handler
    async def get_realm_nonce(self, url):
        try:
            parsed_url = re.match(r"rtsp://([^/]+)(/.*)", url)
            if not parsed_url:
                raise ValueError("Invalid URL format")

            host_port = parsed_url.group(1)
            path = parsed_url.group(2)
            host, port = host_port.split(':')
            port = int(port)

            method = "DESCRIBE"
            initial_request = (f"{method} {url} RTSP/1.0\r\n"
                            f"CSeq: 1\r\n"
                            f"\r\n")
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            sock.send(initial_request.encode())
            
            # 서버 응답 받기
            response = sock.recv(4096).decode()
            sock.close()

            # realm과 nonce 추출
            realm = re.search(r'realm="([^"]+)"', response).group(1)
            nonce = re.search(r'nonce="([^"]+)"', response).group(1)
            return realm, nonce, response
        except Exception as e:
            return None, None, None  # 또는 적절한 값을 반환합니다.


    
    async def generate_digest_auth(self, username, password, realm, nonce, uri, method):
        a1 = f"{username}:{realm}:{password}"
        ha1 = hashlib.md5(a1.encode()).hexdigest()

        a2 = f"{method}:{uri}"
        ha2 = hashlib.md5(a2.encode()).hexdigest()

        response = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()

        digest_header = (f'Digest username="{username}", realm="{realm}", nonce="{nonce}", '
                        f'digest-uri="{uri}", response="{response}", algorithm=MD5')

        return digest_header

    async def check_rtsp_connection(self, host, port, url, username, password):
        sock = None  # 소켓 변수를 초기화합니다.
        try:
            realm, nonce, initial_response = await self.get_realm_nonce(url)
            if realm is None or nonce is None:
                raise ValueError("Failed to get realm and nonce")

            uri = re.match(r"rtsp://[^/]+(/.*)", url).group(1)
            if not uri:
                raise ValueError(f"URL path is empty for URL: {url}")

            method = "DESCRIBE"
            digest_header = await self.generate_digest_auth(username, password, realm, nonce, uri, method)

            rtsp_request = (f"{method} {url} RTSP/1.0\r\n"
                            f"CSeq: 1\r\n"
                            f"Authorization: {digest_header}\r\n"
                            f"\r\n")
            parsed_url = re.match(r"rtsp://([^/]+)(/.*)", url)
            host_port = parsed_url.group(1)
            host, port = host_port.split(':')
            port = int(port)

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            sock.send(rtsp_request.encode())

            # 서버 응답 받기
            response = sock.recv(4096).decode()
            if response == None:
                response = "No response from server"
            if '200 OK' in response.upper():
                logger.debug(f"200 OK!!!: {response}")
                return True, "Connection successful"
            
            else:
                logger.debug(f"????????: {response}")
                return False, response
        except Exception as e:
            logger.error(f"Error during authenticated connection: {response}")
            return False, str(e)
        finally:
            logger.debug(f"fin!!!: {response}")
            if sock:
                sock.close()  # 소켓을 닫습니다.
        #     reader, writer = await asyncio.open_connection(host, port)
        #     writer.write(rtsp_request.encode())
        #     await writer.drain()
            
        #     response = await reader.read(4096)
        #     writer.close()
        #     await writer.wait_closed()

        #     response = response.decode()
        #     status_code = re.search(r"RTSP/1.0 (\d{3})", response)
        #     if status_code:
        #         return status_code.group(1) == "200"
        #     else:
        #         return False
        # except ValueError as ve:
        #     logger.error(f"ValueError during RTSP connection check for URL {url}: {ve}")
        #     return False
        # except Exception as e:
        #     logger.error(f"Error checking RTSP connection: {e}")
        #     return False
    def get_host_ip(self):
        host_ip = os.getenv('HOST_IP')
        return host_ip
    
    async def send_event_request(event_data):
        host_ip = os.getenv('HOST_IP')
        url = f"http://{host_ip}:1223/request_event"
        headers = {'Content-Type': 'application/json'}
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=event_data, headers=headers)
            return response.json()
