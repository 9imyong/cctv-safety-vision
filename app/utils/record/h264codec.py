import subprocess
import asyncio
import shutil
import cv2
import os

async def convert(ipt):
    await asyncio.sleep(2)
    print("waiting convert")
    ipt_name=ipt.split(".mp4")[0]
    # if os.path.getsize(ipt) == 0:
    #     image = cv2.imread(f'{ipt_name}.jpg')
    #     height, width, _ = image.shape
    #     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    #     fps = 10
    #     video_duration = duration  # seconds
    #     out = cv2.VideoWriter(ipt, fourcc, fps, (width, height))
    #     for _ in range(fps * video_duration):
    #         out.write(image)
    #     out.release()
    opt=f'{ipt_name}_temp.mp4'
    
    cmd = [
        "gst-launch-1.0",
        "filesrc",
        "location=" + ipt,
        "!",
        "decodebin",
        "!",
        "videoconvert",
        "!",
        "x264enc",
        "!",
        "mp4mux",
        "!",
        "filesink",
        "location=" + f'{opt}',
    ]
    result = subprocess.run(cmd, capture_output=True)
    await asyncio.sleep(0.2)

    if result.returncode != 0:
        print("Error:", result.stderr)
    else:
        print("Conversion successful!")
        shutil.move(opt,ipt)