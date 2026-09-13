# CCTV Safety Vision

건설 현장 CCTV 영상을 실시간 분석해 **안전보호구 미착용 · 위험구역 침입 · 신호수 미배치 · 추락**을
검출하고, 이벤트 발생 시 영상을 녹화해 외부 관제 플랫폼으로 전송하는 AI 영상분석 시스템이다.

다채널 RTSP 입력을 채널별 독립 프로세스로 처리하면서, 동시에 분석 결과가 오버레이된 화면을
HLS로 재송출한다. 실제 현장에 배포되어 운영된 시스템의 코드다.

> **성과 요약** *(아래 수치는 실제 운영값으로 채워 넣을 것)*
> - 동시 처리 채널: `<N>`채널 / GPU `<M>`장
> - 채널당 처리 FPS: `<N>` FPS (입력 `<N>` FPS 대비 `<N>`:1 샘플링)
> - 프레임당 추론 지연: `<N>` ms (3개 모델 병렬 추론 + 앙상블 포함)
> - 오탐 개선: `<도입 전 N건/일 → 도입 후 N건/일>`

---

## 1. 시스템 구성

```
                   ┌──────────────────────────────┐
   관제 웹/API ───▶ │ FastAPI Control Plane        │
                   │  /live/{start|stop}          │
                   │  /VAControl/cgi  (카메라 CGI)│
                   │  /CCTVLIST, /CURRENTLY_WORKING│
                   └──────────┬───────────────────┘
                              │ apply_async / revoke
                   ┌──────────▼───────────────────┐
                   │ Celery (gevent, Redis broker)│
                   │  채널 1개 = 장기 실행 태스크 1개 │
                   └──────────┬───────────────────┘
                              │
          ┌───────────────────▼────────────────────┐
          │ GStreamer 파이프라인 (채널당 1개)        │
          │  RTSP ─▶ OpenCV 디코드 ─▶ AI 추론       │
          │       ─▶ 오버레이 ─▶ appsrc ─▶ x264enc │
          │       ─▶ hlssink (.m3u8 / .ts)         │
          └───────────────────┬────────────────────┘
                              │ 이벤트 발생 시
                   ┌──────────▼───────────────────┐
                   │ 녹화(mp4) + 썸네일 생성        │
                   │  ─▶ 외부 관제 플랫폼 POST      │
                   └──────────────────────────────┘
```

| 구성요소 | 역할 |
|---|---|
| `app/main.py` | FastAPI 제어 API. 채널 시작/중지, 목록 조회, HLS 세그먼트 서빙, 이벤트 중계 |
| `app/tasks.py` | Celery 태스크. 채널별 스트리밍 실행, 1분 주기 RTSP 헬스체크 |
| `app/run_appsrc.py` | GStreamer 파이프라인 · 프레임 루프 · 시나리오 판정 · 이벤트 녹화 |
| `app/ai.py` | 3개 TensorRT 엔진 병렬 추론 + 박스 앙상블 + 시나리오별 검출 로직 |
| `app/aiutils.py` | 폴리곤 내부 판정, IoU, bbox 중심점 등 기하 연산 |
| `app/utils/db/get_db.py` | MySQL 커넥션 풀, 좌표/시나리오 조회, RTSP Digest 인증 헬스체크 |
| `app/utils/record/h264codec.py` | 녹화본 H.264 재인코딩 (브라우저 재생 호환) |

---

## 2. 설계에서 신경 쓴 지점

### 2.1 3개 모델 앙상블로 오탐 억제

단일 검출 모델로는 현장 조건(역광, 원거리, 가림)에서 작업자 검출이 불안정했다.
용도가 다른 3개 엔진을 **동시에** 돌리고 결과를 합친다.

| 엔진 | 담당 |
|---|---|
| `TRT_ENGINE_PPE` | 안전모/안전대 착용·미착용 |
| `TRT_ENGINE_DL` | 현장 특화 학습 모델 (작업자, 신호수, 차량) |
| `TRT_ENGINE_PUBLIC` | 범용 COCO 계열 (사람, 차량) |

세 추론을 스레드로 병렬 실행(`CustomThread`)한 뒤 `weighted_boxes_fusion`으로 병합한다.
NMS가 아니라 WBF를 쓴 이유는, 서로 다른 모델이 같은 객체에 대해 조금씩 다른 박스를 낼 때
**하나를 버리는 대신 가중 평균**하는 편이 경계 케이스에서 안정적이었기 때문이다.

### 2.2 거리에 따른 동적 임계값

원거리 작업자는 bbox가 작아 PPE 판정이 사실상 불가능한데, 그대로 두면 "미착용"으로
오탐이 쏟아진다. 프레임 내 작업자 bbox **평균 면적을 기준으로 최소 검사 면적을 매 프레임 산출**해
그보다 작은 객체는 PPE 판정 대상에서 제외한다.

```python
min_box_ppe = 0.06 * (sum(box_area) / len(boxes) - 100)
```

카메라 설치 높이·화각이 현장마다 달라 고정 픽셀 임계값이 통하지 않았던 문제의 대응이다.

### 2.3 판정 기준점은 bbox 하단 중심

위험구역 침입은 bbox 중심이 아니라 **하단 중심(발 위치)** 으로 판정한다.
중심점 기준이면 경계선 근처에서 상체만 걸쳐도 침입으로 잡힌다.
폴리곤 내부 판정은 ray casting을 직접 구현했다.

### 2.4 신호수 미배치: 차량이 "움직일 때만" 위반

차량이 주차만 되어 있어도 신호수를 요구하면 위반이 폭증한다.
이전 프레임 차량 위치와 최근접 매칭해 **이동량이 임계값을 넘은 경우에만** 신호수 존재를
확인하고, 없으면 카운터를 누적한다. 단발 프레임 오검출로 이벤트가 뜨지 않도록
연속 카운트 기반으로 판정한다.

### 2.5 RTSP 헬스체크에 Digest 인증 직접 구현

현장 카메라 상당수가 Digest 인증을 요구해 단순 TCP 연결 확인으로는 상태를 알 수 없었다.
`OPTIONS` 요청으로 realm/nonce를 받아 MD5 다이제스트를 만들어 재요청하는 흐름을
직접 구현해, Celery beat가 1분 주기로 전 채널 연결성을 DB에 기록한다.

---

## 3. 실행

### 요구사항

- Python 3.8+, NVIDIA GPU (TensorRT), GStreamer 1.x (`gst-plugins-bad`의 `hlssink` 포함)
- MySQL, Redis
- [`pygst-utils`](https://github.com/jackersson/pygst-utils) — [ATTRIBUTION.md](ATTRIBUTION.md) 참고
- `mmdeploy_runtime` (TensorRT 엔진 로딩)

```bash
cp .env.example .env          # DB/Redis/엔진 경로 설정
pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 1223
celery -A app.tasks worker --pool=gevent --concurrency=100
celery -A app.tasks beat
```

### 채널 시작/중지

```bash
curl -X POST http://localhost:1223/live/start \
  -H 'Content-Type: application/json' \
  -d '{"c_id": ["cctv1", "cctv2"], "ai_on": [true, true]}'

curl -X POST http://localhost:1223/live/stop \
  -H 'Content-Type: application/json' \
  -d '{"c_id": ["cctv1"], "ai_on": [false]}'
```

HLS 재생: `http://<host>:1223/hls/<cctv_id>/playlist.m3u8`

---

## 4. 설계 회고 — 지금이라면 다르게 할 부분

운영하면서 드러난 구조적 한계를 남겨 둔다.

**채널 소유권이 Redis 딕셔너리 하나에 있다.**
`tasklist`(cctv_id → task_id)를 읽고·수정하고·다시 쓰는 방식이라 read-modify-write 경합에
그대로 노출된다. 시작 요청이 동시에 들어오면 한쪽 task_id가 유실되고, 그 채널은
중지 API로 내릴 수 없는 고아 프로세스가 된다.

**워커가 죽으면 복구가 수동이다.**
중지는 `revoke(terminate=True)`에 의존하는데, 워커 프로세스 자체가 죽으면 그 채널은
아무도 인계받지 않는다. lease/heartbeat 기반 소유권과 takeover가 필요하다.

**제어 명령이 동기 HTTP다.**
`/live/start`가 DB 조회와 태스크 생성을 요청 안에서 처리해, 채널이 늘수록 응답이 느려진다.
명령을 이벤트로 분리하고 상태 머신(`desired_state` / `actual_state`)으로 수렴시키는 편이 맞다.

→ 이 세 가지를 Kafka + lease 기반으로 다시 설계한 것이
[streaming-pipeline](https://github.com/9imyong/streaming-pipeline) 이다.

---

## 5. 라이선스 / 공개 범위

현장 식별 정보, 카메라 자격증명, 발주처 연동 엔드포인트, 학습된 모델 가중치는 모두 제외했다.
설정값은 전부 환경변수로 분리되어 있으며 `.env`와 `config/config.yaml`은 커밋 대상이 아니다.
