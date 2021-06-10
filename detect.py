import argparse
import os

import cv2
import numpy as np
import torch
import yaml
from models.experimental import attempt_load
from numpy import random

from utils.datasets import letterbox
from utils.general import (
    check_img_size, non_max_suppression, scale_coords, plot_one_box)
from utils.torch_utils import select_device

os.environ['KMP_DUPLICATE_LIB_OK']='True'
parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='./models/weights/mymodel.pt', help='models.pt path(s)')
parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
parser.add_argument('--cfg', type=str, default='./models/yolov4-p5-update.yaml', help='models.yaml path')
parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--data', type=str, default='./data/mydata.yaml', help='data.yaml path')
opt = parser.parse_args()

source, weights, imgsz = opt.source, opt.weights, opt.img_size

# Initialize
device = select_device(opt.device)
half = device.type != 'cpu'  # half precision only supported on CUDA

with open(opt.data) as f:
    data_dict = yaml.load(f, Loader=yaml.FullLoader)

# Load models
model = attempt_load(weights, map_location=device)  # load FP32 models

imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
if half:
    model.half()  # to FP16

# Get names and colors
names = data_dict['names']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

def detect():
    # Run inference
    pipe = 0
    # pipe = 'http://192.168.1.82:8080/video'
    cap = cv2.VideoCapture(pipe)
    while True:
        ret_val, frame = cap.read()
        img = letterbox(frame, new_shape=imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes)
        for index, detect in enumerate(pred):
            if detect is not None and len(detect):
                # Rescale boxes from img_size to frame size
                detect[:, :4] = scale_coords(img.shape[2:], detect[:, :4], frame.shape).round()
                for *xyxy, conf, cls in detect:
                    label = names[int(cls)]
                    plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=2)
        cv2.imshow("ObjectDetect", frame)
        if cv2.waitKey(1) == ord('q'):  # q to quit
            raise StopIteration

def objectdetect(frame):
    dict_object = {}
    img = letterbox(frame, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img, augment=opt.augment)[0]
    print("inside: 1")
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes)
    for index, detect in enumerate(pred):
        if detect is not None and len(detect):
            # Rescale boxes from img_size to im0 size
            detect[:, :4] = scale_coords(img.shape[2:], detect[:, :4], frame.shape).round()
            for *xyxy, conf, cls in detect:
                label = names[int(cls)]
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                dict_object[label] = frame[y1:y2, x1:x2]
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=2)
    return dict_object
if __name__ == '__main__':
    detect()
    # img = cv2.imread('data/object/object-0.jpeg')
    # objectdetect(img)