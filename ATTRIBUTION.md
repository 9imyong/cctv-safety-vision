# 서드파티 출처

## gstreamer-python (pygst-utils)

GStreamer Python 바인딩 래퍼로 [jackersson/pygst-utils](https://github.com/jackersson/pygst-utils)
(author: LifeStyleTransfer) 를 사용한다.

원본 저장소에서는 이 라이브러리가 소스 트리에 통째로 벤더링되어 있었으나,
본 저장소에서는 **제거하고 의존성으로 분리**했다. `app/run_appsrc.py` 가 import 하는
`gstreamer`, `gstreamer.utils` 모듈이 여기서 온다.

설치:

```bash
git clone https://github.com/jackersson/pygst-utils.git
cd pygst-utils && pip install .
```

`app/` 아래의 코드는 전부 직접 작성한 것이다.
