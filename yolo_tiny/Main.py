import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Process
# Cai dat tham so doc weight, config va class name
from yolo_tiny.firebase import Firebase

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', default='path/yolov3.cfg',
                help='path to yolo config file')
ap.add_argument('-w', '--weights', default='path/yolov3.weights',
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', default='path/coco.names',
                help='path to text file containing class names')
args = ap.parse_args()
print(args)


def plot_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    plt.imshow(image)
    plt.show()


def get_path_image(image):
    filename = './image/ras.png'
    image = cv2.resize(image, (128, 128))
    cv2.imwrite(filename, image)


# Ham tra ve output layer
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


# Ham ve cac hinh chu nhat va ten class
def draw_prediction(img, class_id, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# Doc tu webcam
# cap  = VideoStream(src='http://192.168.1.4:8080/video').start()
# cap  = VideoStream(src='video/video8.mp4').start()


# Doc ten cac class
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet(args.weights, args.config)
Foreign_obj = ['cell phone']

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_video = cv2.VideoWriter('./video/ras.avi', fourcc, 5, (720, 480))


def run_v2():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow("Capturing", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def savevideo(out_video, frames_video):
    for f in frames_video:
        out_video.write(f)
    out_video.release()
    data = Firebase()
    data.getdata()


def run():
    isSave = False
    count = 0
    # cap = VideoStream(src=0).start()
    # cap = cv2.VideoCapture('https://10.10.57.41:8080/video')
    cap = cv2.VideoCapture(0)

    video = inni_video()

    # khởi tạo thời điểm ban đầu chạy chương trình
    start = time.time()

    # thời điểm lưu
    startsave = time.time()
    while (cap.isOpened()):
        ret, frame = cap.read()
        # Resize va dua khung hinh vao mang predict
        frame = cv2.resize(frame, (720, 480))
        Height, Width = frame.shape[:2]

        scale = 0.00392
        blob = cv2.dnn.blobFromImage(frame, scalefactor=scale, size=(416, 416), mean=(0, 0, 0), swapRB=True,
                                     crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))
        # Loc cac object trong khung hinh
        class_ids = []
        confidences = []
        boxes = []
        obj_threshold = 0.5
        nms_threshold = 0.2
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if (confidence > 0.5):
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, obj_threshold, nms_threshold)
        create_video(video, frame, start)

    # Ve cac khung chu nhat quanh doi tuong
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(frame, class_ids[i], round(x), round(y), round(x + w), round(y + h))
            print([round(x), round(y), round(x + w), round(y + h)])

            if classes[class_ids[i]] in Foreign_obj:
                if count == 0:
                    startsave = time.time()
                    isSave = save_video(video, isSave)
                    get_path_image(frame)
                    count = 1


        cv2.imshow("object detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out_video.release()
    cap.release()
    cv2.destroyAllWindows()


def inni_video():
    joint_video = []
    return joint_video


def create_video(joint_video, frame, start):
    joint_video.append(frame)
    if time.time() - start > 10:
        joint_video.pop(0)


def save_video(joint_video, isSave):
    if isSave:
        try:
            for f in joint_video:
                out_video.write(f)
            out_video.release()
            data = Firebase()
            data.getdata()
            isSave = False
            count = 0
        except():
            print("Save to Firebase Error")

    return isSave


def ObjectDetect(frame, count=0):
    dic_object = {}
    frame = cv2.resize(frame, (720, 480))
    Height, Width = frame.shape[:2]

    scale = 0.00392
    blob = cv2.dnn.blobFromImage(frame, scalefactor=scale, size=(416, 416), mean=(0, 0, 0), swapRB=True,
                                 crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    # Loc cac object trong khung hinh
    class_ids = []
    confidences = []
    boxes = []
    obj_threshold = 0.5
    nms_threshold = 0.2
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if (confidence > 0.5):
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, obj_threshold, nms_threshold)
    # Ve cac khung chu nhat quanh doi tuong
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        print([round(x), round(y), round(x + w), round(y + h)])
        plt.imshow(frame[x:x + w, y:y + h:, ])
        plt.show()
        if classes[class_ids[i]] in Foreign_obj:
            draw_prediction(frame, class_ids[i], round(x), round(y), round(x + w), round(y + h))
            dic_object[classes[class_ids[i]]] = frame[round(y):round(y + h), round(x):round(x + w)]
            if count == 0:
                startsave = time.time()
                isSave = True
                get_path_image(frame)
                count = 1
    return dic_object


if __name__ == "__main__":
    run()
