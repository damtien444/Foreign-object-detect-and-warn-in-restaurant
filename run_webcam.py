import argparse
import logging
import time

import cv2, queue, threading
from scripts.openCamera import VideoCapture
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import scripts.label_image as label_img
from SinglePic import draw_human, wrap_human
from yolo_tiny.Main import *

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 '
                             'or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))

    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))

    logger.debug('cam read+')
    # cam = cv2.VideoCapture(args.camera)
    cam = VideoCapture('https://192.168.43.126:8080/video')
    image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    pre_state = "sitting"
    last = 0

    session = label_img.SessionRun()


    # count = 0
    while True:

        image = cam.read()

        # Call Openpose to export the human skeleton
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        if len(humans) == 0:
            cv2.imshow('tf-pose-estimation result', image)

            if cv2.waitKey(1) == 27:
                break

            continue

        img = draw_human(image, humans[0], imgcopy=False)

        # Getting only the skeletal structure (with white background) of the actual image
        image = np.zeros(image.shape, dtype=np.uint8)
        image.fill(255)
        image = draw_human(image, humans[0], imgcopy=False)

        # Classification, using MobileNet V2
        pose_class, poss = session.classify(image)
        print(pose_class)
        print(poss)

        # TODO: Tinh chỉnh thời gian interval lại cho chính xác hơn!

        # Compare to previous state
        inteval = time.time() - last
        print(inteval)
        if pose_class == "standing" and pre_state == "sitting" and inteval > 5:

            print("alerttttttttt")

            # TODO: Viết hàm gửi cảnh báo


            # TODO: Kích hoạt hàm ghi video chuẩn bị video để gửi đi

            # TODO: Kích hoạt mô hình nhận diện'

            last = time.time()
            img, center = wrap_human(img, humans[0], pose_class, imgcopy=False, color=(0,0,255))
        else:
            img, center = wrap_human(img, humans[0], pose_class, imgcopy=False, color=(255,0,0))

        pre_state = pose_class

        cv2.imshow('tf-pose-estimation result', img)

        if cv2.waitKey(1) == 27:
            break

        # For gathering training data
        # title = 'img'+str(count)+'.jpeg'
        # path = <enter any path you want>
        # cv2.imwrite(os.path.join(path , title), image)
        # count += 1

    cv2.destroyAllWindows()

def inni_video():
    video =[]
    return video

def build_video(video, arrived_frame, video_lenght, start_timestamp):
    video.append(arrived_frame)
    if time.time() - start_timestamp > 10:
        video.pop(0)



# =============================================================================
# For running the script simply run the following in the cmd prompt/terminal :
# python run_webcam.py --model=mobilenet_thin --resize=432x368 --camera=0
# =============================================================================
