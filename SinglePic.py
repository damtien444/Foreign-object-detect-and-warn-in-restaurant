import argparse
import logging
import time

import cv2, queue, threading
import numpy as np

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import scripts.label_image as label_img

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


def draw_human(npimg, human, imgcopy=False):
    if imgcopy:
        npimg = np.copy(npimg)
    image_h, image_w = npimg.shape[:2]
    centers = {}
    # for human in humans:
    # draw point
    # min_x, min_y, max_x, max_y = 0, 0, 0, 0

    for i in range(common.CocoPart.Background.value):
        if i not in human.body_parts.keys():
            continue

        body_part = human.body_parts[i]
        center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))

        centers[i] = center
        cv2.circle(npimg, center, 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)

    # draw line
    for pair_order, pair in enumerate(common.CocoPairsRender):
        if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
            continue

        # npimg = cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)
        cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)

    # thickness = 2
    # cv2.rectangle(npimg, (x1, y1), (x2, y2), (255,0,0), 2)
    # cv2.rectangle(npimg, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 0, 255), 2)

    return npimg


def wrap_human(npimg, human, pose_class_write, imgcopy=False, color=(255,0,0)):
    if imgcopy:
        npimg = np.copy(npimg)
    image_h, image_w = npimg.shape[:2]
    centers = {}
    # for human in humans:
    # draw point
    min_x, min_y, max_x, max_y = 0, 0, 0, 0

    for i in range(common.CocoPart.Background.value):
        if i not in human.body_parts.keys():
            continue

        body_part = human.body_parts[i]
        # center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))

        if min_x == 0:
            min_x = body_part.x * image_w + 0.5

        if min_y == 0:
            min_y = body_part.y * image_h + 0.5

        min_x = int(min(min_x, body_part.x * image_w + 0.5))
        min_y = int(min(min_y, body_part.y * image_h + 0.5))
        max_x = int(max(max_x, body_part.x * image_w + 0.5))
        max_y = int(max(max_y, body_part.y * image_h + 0.5))

    cv2.rectangle(npimg, (min_x, min_y), (max_x, max_y), color, 2)

    cv2.putText(npimg,
                "Predicted : %s" % pose_class_write,
                (min_x, min_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2)

    center = (int((min_x+max_x)/2), int((min_y+max_y)/2))
    cv2.circle(npimg, center, 3, [255,0,0], thickness=3, lineType=8, shift=0)

    return npimg, center


class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


# def center_of_human(npimg, human, imgcopy=False):
#     if imgcopy:
#         npimg = np.copy(npimg)
#     image_h, image_w = npimg.shape[:2]
#     center = (0,0)
#     for i in range(common.CocoPart.Background.value):
#         if i not in human.body_parts.keys():
#             continue
#
#         body_part = human.body_parts[i]
#         if center == (0,0):
#             center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
#
#         center = (int((body_part.x * image_w + 0.5+center[0])/2), int((body_part.y * image_h + 0.5+center[1])/2))
#
#     return npimg


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
    # cam = VideoCapture('https://192.168.31.170:8080/video')
    image = cv2.imread("D:\Downloads\PXL_20210402_113751913.MP.jpg", )
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    # count = 0
    # while True:

    logger.debug('+image processing+')

    logger.debug('+postprocessing+')
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    print(humans)
    i = 1

    print(humans[0].score)
    img = draw_human(image, humans[0], imgcopy=False)

    logger.debug('+classification+')
    # Getting only the skeletal structure (with white background) of the actual image
    image = np.zeros(image.shape, dtype=np.uint8)
    image.fill(255)
    image = draw_human(image, humans[0], imgcopy=False)

    # Classification
    pose_class = label_img.classify(image)
    print(pose_class)

    img = wrap_human(image, humans[0], imgcopy=False)

    logger.debug('+displaying+')
    cv2.putText(img,
                "Current predicted pose is : %s" % (pose_class),
                (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2)

    cv2.imshow('tf-pose-estimation result', img)

    fps_time = time.time()
    # if cv2.waitKey(1) == 27:
    #     break
    logger.debug('+finished+')

    # For gathering training data
    # title = 'img'+str(count)+'.jpeg'
    # path = <enter any path you want>
    # cv2.imwrite(os.path.join(path , title), image)
    # count += 1

    while True:
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

# =============================================================================
# For running the script simply run the following in the cmd prompt/terminal :
# python run_webcam.py --model=mobilenet_thin --resize=432x368 --camera=0
# =============================================================================
