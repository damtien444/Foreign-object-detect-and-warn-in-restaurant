import threading
import time

import scripts.label_image as label_img
from SinglePic import draw_human, wrap_human
from detect import *
from getFlagFromSensor import flag
from scripts.openCamera import VideoCapture
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
from yolo_tiny.firebase import Firebase

parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
parser.add_argument('--camera', type=int, default=0)

parser.add_argument('--resize', type=str, default='0x0',
                    help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 '
                         'or 656x368 or 1312x736 ')
parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                    help='if provided, resize heatmaps before they are post-processed. default=1.0')

parser.add_argument('--models', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
parser.add_argument('--show-process', type=bool, default=False,
                    help='for debug purpose, if enabled, speed for inference is dropped.')
args = parser.parse_args()

IP_CAMERA_ADDRESS = 'http://192.168.1.7:8080/video'
# IP_CAMERA_ADDRESS = 0
CONFIDENCE_TIME = 1
SESSION_TIMEOUT = 2*60


def detect_human_leaving():

    flag()

    # Load the models
    sess = label_img.SessionRun()
    # cla = Classify()

    e = TfPoseEstimator(get_graph_path(args.models), target_size=(432, 368))

    # cam = cv2.VideoCapture(args.camera)
    cam = VideoCapture(IP_CAMERA_ADDRESS)
    image = cam.read()

    pre_state = None
    last = time.time()
    last_has_human = False
    has_human = False
    last_event_time = time.time()

    # count = 0
    while True:

        image = cam.read()

        humans = e.inference(image, upsample_size=args.resize_out_ratio)

        if len(humans) == 0:
            has_human = False
            last_has_human = has_human

            # TODO: Moi them nha
            if time.time() - last_event_time > SESSION_TIMEOUT:
                flag()

        else:
            last_event_time = time.time()
            has_human = True
            img = draw_human(image, humans[0], imgcopy=False)

            # Getting only the skeletal structure (with white background) of the actual image
            image = np.zeros(image.shape, dtype=np.uint8)
            image.fill(255)
            image = draw_human(image, humans[0], imgcopy=False)

            # Classification
            pose_class, poss = sess.classify(image)
            # pose_class = cla.classify(image)
            print(pose_class)
            print(poss)

            # TODO: Tinh chỉnh thời gian interval lại cho chính xác hơn!

            # Compare to previous state
            inteval = time.time() - last
            print(inteval)

            if (pose_class == "standing" and pre_state == "sitting" and inteval > 3) or (last_has_human and not has_human):

                # recheck the condition to make sure
                time.sleep(CONFIDENCE_TIME)
                image_check = cam.read()

                humans = e.inference(image_check, upsample_size=args.resize_out_ratio)

                if len(humans) != 0:
                    has_human = True
                    last_has_human = has_human
                    print("False Alert!")
                    continue

                if len(humans) == 0:
                    last_has_human = False

                    print("alerttttttttt")

                    # TODO: Kích hoạt hàm ghi video chuẩn bị video để gửi đi

                    # TODO: Kích hoạt mô hình nhận diện

                    objects = detect_object_onTable(image_check)
                    if len(objects) != 0:

                        print("There're something left!")
                        # for value in objects.values():
                        #     plt.imshow(value)
                        #     plt.show()
                        print(objects.keys())

                        event.set()

                        last = time.time()
                        # img_check, center = wrap_human(img_check, humans[0], pose_class, imgcopy=False, color=(0, 0, 255))
                        cv2.imshow('tf-pose-estimation result', image_check)
                        get_path_image(image_check)
                        if cv2.waitKey(1) == 27:
                            break

                        event.clear()
            else:
                img, center = wrap_human(img, humans[0], pose_class, imgcopy=False, color=(255, 0, 0))
                cv2.imshow('tf-pose-estimation result', img)

                pre_state = pose_class

                if cv2.waitKey(1) == 27:
                    break

    cv2.destroyAllWindows()


def get_path_image(image):
    filename = './image/ras.png'
    image = cv2.resize(image, (432, 368))
    cv2.imwrite(filename, image)


def detect_object_onTable(frame):
    print("Find things")
    objects = objectdetect(frame)
    print("Done finding")

# objects = ObjectDetect(frame,0)
    for key in objects:
        print(key)
    # print(objects)
    return objects


def savevideo(out_video, frames_video):
    for f in frames_video:
        out_video.write(f)
    out_video.release()
    data = Firebase()
    data.push_data()
    print("Sent successfully")


def saveVideoAndPush():
    cam = VideoCapture(IP_CAMERA_ADDRESS)

    start = time.time()
    start_save = time.time()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter('./video/ras.avi', fourcc, 24, (800, 600))

    frames_video = []
    is_save = False

    while True:
        frame = cam.read()

        if event.is_set():
            start_save = time.time()
            is_save = True

        frames_video.append(frame)
        if time.time() - start > 5 and not is_save:
            frames_video.pop(0)
        else:
            if time.time() - start_save > 5 and is_save:
                print("Sending")
                savevideo(out_video, frames_video)
                is_save = False
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out_video = cv2.VideoWriter('./video/ras.avi', fourcc, 30, (800, 600))

        cv2.imshow("object detection", frame)

        if cv2.waitKey(1) == 27:
            break


event = threading.Event()

t1 = threading.Thread(target=detect_human_leaving)

t2 = threading.Thread(target=saveVideoAndPush)

# t3 = threading.Thread(target=play)

t1.start()
# t3.start()
t2.start()
