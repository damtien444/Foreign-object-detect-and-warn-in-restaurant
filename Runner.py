import time

import cv2, queue, threading
from scripts.openCamera import VideoCapture
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import scripts.label_image as label_img
from SinglePic import draw_human, wrap_human
from yolo_tiny.Main import *
import matplotlib.pyplot as plt

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

IP_CAMERA_ADDRESS = 'http://192.168.43.126:8080/video'
CONFIDENCE_TIME = 1


def detect_human_leaving():

    # Load the model
    sess = label_img.SessionRun()
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))

    # cam = cv2.VideoCapture(args.camera)
    cam = VideoCapture(IP_CAMERA_ADDRESS)
    image = cam.read()

    pre_state = None
    last = time.time()
    last_has_human = False

    # count = 0
    while True:

        image = cam.read()

        humans = e.inference(image, upsample_size=args.resize_out_ratio)

        if len(humans) == 0:
            last_has_human = False
            continue

        img = draw_human(image, humans[0], imgcopy=False)

        # Getting only the skeletal structure (with white background) of the actual image
        image = np.zeros(image.shape, dtype=np.uint8)
        image.fill(255)
        image = draw_human(image, humans[0], imgcopy=False)

        # Classification
        pose_class, poss = sess.classify(image)
        print(pose_class)
        print(poss)

        # TODO: Tinh chỉnh thời gian interval lại cho chính xác hơn!

        # Compare to previous state
        inteval = time.time() - last
        print(inteval)
        if pose_class == "standing" and pre_state == "sitting" and inteval > 10 and last_has_human:

            # recheck the condition to make sure
            time.sleep(CONFIDENCE_TIME)
            image_check = cam.read()
            humans = e.inference(image_check, upsample_size=args.resize_out_ratio)

            img_check = draw_human(image_check, humans[0], imgcopy=False)

            # Getting only the skeletal structure (with white background) of the actual image
            image_check = np.zeros(image_check.shape, dtype=np.uint8)
            image_check.fill(255)
            image_check = draw_human(image_check, humans[0], imgcopy=False)

            # Classification
            pose_class, poss = sess.classify(image_check)
            print(pose_class)
            print(poss)
            print("False Alert!")

            if len(humans) == 0:
                last_has_human = False

                print("alerttttttttt")

                # TODO: Kích hoạt hàm ghi video chuẩn bị video để gửi đi
                event.set()

                # TODO: Kích hoạt mô hình nhận diện
                objects = detect_object_onTable(img_check)
                for value in objects.values():
                    plt.imshow(value)
                    plt.show()

                last = time.time()
                # img_check, center = wrap_human(img_check, humans[0], pose_class, imgcopy=False, color=(0, 0, 255))
                cv2.imshow('tf-pose-estimation result', img_check)

            event.clear()
        else:
            img, center = wrap_human(img, humans[0], pose_class, imgcopy=False, color=(255, 0, 0))
            cv2.imshow('tf-pose-estimation result', img)

        pre_state = pose_class
        last_has_human = True

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


def detect_object_onTable(frame):
    objects = ObjectDetect(frame, 0)
    for key in objects:
        print(key)
    # print(objects)
    return objects


# def saveVideoAndPush():
#     cam = VideoCapture(IP_CAMERA_ADDRESS)
#
#     start = time.time()
#     start_save = time.time()
#
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out_video = cv2.VideoWriter('./video/ras.avi', fourcc, 30, (640, 480))
#
#     frames_video = []
#     pre_state = True
#     is_save = False
#
#     while True:
#         frame = cam.read()
#
#         if event.is_set() and not pre_state:
#             start_save = time.time()
#             is_save = True
#
#         frames_video.append(frame)
#         if time.time() - start > 5 and not is_save:
#             frames_video.pop(0)
#         else:
#             if time.time() - start_save > 5 and is_save:
#                 print("Sending")
#                 frames_video.reverse()
#                 savevideo(out_video, frames_video)
#                 is_save = False
#
#         cv2.imshow("object detection", frame)
#
#         if cv2.waitKey(1) == 27:
#             break
#
#         pre_state = event.is_set()


event = threading.Event()

t1 = threading.Thread(target=detect_human_leaving)

# t2 = threading.Thread(target=saveVideoAndPush)

t1.start()

# t2.start()
