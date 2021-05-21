import argparse
import logging
import os
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import scripts.label_image as label_img
from scripts.openCamera import VideoCapture

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
    # ret_val, image = cam.read()
    cam = VideoCapture('https://192.168.43.126:8080/video')
    image = cam.read()

    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    DIR = './data/object'
    dataNum = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    print("Number of data already collected: " + str(dataNum))
    count = dataNum

    while True:

        logger.debug('+image processing+')

        image = cam.read()

        # logger.debug('+postprocessing+')
        # humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        # img = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        logger.debug('+classification+')
        # Getting only the skeletal structure (with white background) of the actual image
        # image = np.zeros(image.shape, dtype=np.uint8)
        # image.fill(255)
        # image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        # Classification
        # pose_class = label_img.classify(image)
        #
        # logger.debug('+displaying+')
        # cv2.putText(img,
        #             "Current predicted pose is : %s" %(pose_class),
        #             (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (0, 255, 0), 2)
        #
        # cv2.imshow('with background', img)
        cv2.imshow('without Background', image)

        fps_time = time.time()

        while True:
            # Gather whenever we want
            if cv2.waitKey(0) == 27:
                break

        logger.debug('+finished+')
        print(count)

        # For gathering training data
        title = 'object-' + str(count) + '.jpeg'
        path = './data/object'
        cv2.imwrite(os.path.join(path, title), image)

        # title = 'imgWithout-' + str(count) + '.jpeg'
        # path = './data/withoutBackGround/stand'
        # cv2.imwrite(os.path.join(path, title), image)

        count += 1

    cv2.destroyAllWindows()

# =============================================================================
# For running the script simply run the following in the cmd prompt/terminal :
# python run_webcam.py --model=mobilenet_thin --resize=432x368 --camera=0
# =============================================================================
