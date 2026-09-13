FROM openmmlab/mmdeploy:ubuntu20.04-cuda11.8-mmdeploy
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y -f ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio libcairo2-dev libxt-dev libgirepository1.0-dev\
    && rm -rf /var/lib/apt/lists/* 
RUN apt-get update && apt-get install -y python3 python3-pip && apt-get clean
ARG DEBIAN_FRONTEND=noninteractive
RUN pip install -U pip
RUN pip install pycairo PyGObject onnxruntime mmdeploy_runtime
RUN apt-get install -y python3-gst-1.0
RUN pip install --ignore-installed  openmim==0.3.9 tensorrt
RUN mim install mmengine mmcv mmyolo mmdeploy==1.2.0
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
# GStreamer Python 래퍼 (ATTRIBUTION.md 참고)
RUN pip install git+https://github.com/jackersson/pygst-utils.git

COPY ./app /app/app
COPY ./config /app/config

EXPOSE 1223
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "1223"]
