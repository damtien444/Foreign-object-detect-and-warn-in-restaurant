import time

import cv2
from yolo_tiny.firebase import Firebase
from scripts.openCamera import VideoCapture


def save_video(out_video: cv2.VideoWriter, frame_video):
    # To push final video to firebase storage

    for frame in frame_video:
        out_video.write(frame)
    out_video.release()
    push = Firebase()
    push.push_data()


cam = VideoCapture('https://10.10.57.42:8080/video')
start = time.time()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_video = cv2.VideoWriter('./video/ras.avi', fourcc, 5, (720, 480))

frames_video = []

while True:
    frame = cam.read()

    frames_video.append(frame)
    if time.time() - start > 10 and isSave == False:
        frames_video.pop(0)
    else:
        if time.time() - start > 10:
            print(1)
            for f in frames_video:
                out_video.write(f)
            out_video.release()
            # data = Firebase()
            # data.getdata()
            isSave = False
            # count = 0

    cv2.imshow("object detection", frame)
