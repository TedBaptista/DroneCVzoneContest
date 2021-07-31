from absl import flags
import sys
from absl import app

# Addresses `UnrecognizedFlagError: Unknown command line flag 'f'`
sys.argv = sys.argv[:1]

# `app.run` calls `sys.exit`
try:
    app.run(lambda argv: None)
except:
    pass

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet


# Addresses `UnrecognizedFlagError: Unknown command line flag 'f'`
sys.argv = sys.argv[:1]

# `app.run` calls `sys.exit`
try:
  app.run(lambda argv: None)
except:
  pass


FLAGS = flags.FLAGS
FLAGS(sys.argv)

class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('./weights/yolov3.tf')

# Definition of the parameters
max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8

# initialize deep sort
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)

# calculate cosine distance metric
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
# initialize tracker
tracker = Tracker(metric)

#begin video capture
vid = cv2.VideoCapture('VIDEO.mp4')

#preparing output video
codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./data/video/results.avi', codec, vid_fps, (vid_width, vid_height))

#from a collections for past centers
from _collections import deque
pts = [deque(maxlen=30) for _ in range(1000)]

#initialize counter
counter = []

# while running the video
while True:
    _, img = vid.read()
    if img is None:
        print('Video has ended or failed')
        break

    #convert to RGB
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #expand dimensions on batch size and resize
    img_in = tf.expand_dims(img_in, 0)
    img_in = transform_images(img_in, 416)

    t1 = time.time()    #declare time in order to calculate fps

    boxes, scores, classes, nums = yolo.predict(img_in) #use the predict function
    # boxes, 3D shape ( 1,100,4)
    # scores, 2D shape (1,100)
    # classes, 2D shape (1,100)
    # nums, 1D shape(1,)

    classes = classes[0]
    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    names = np.array(names)
    converted_boxes = convert_boxes(img, boxes[0])  #using function convert_boxes to convert boxes to a list
    features = encoder(img, converted_boxes)

    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores[0], names, features)]
    #bbox in tlwh format, scores, classname and the feature (type of object detected)
    #we need to put that information in np.arrays
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    #predict positions with kalman filter and update its conditions
    tracker.predict()
    tracker.update(detections)

    #create color map
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

    #
    current_count = int(0)

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update >1:  #if kalman doesnt assign a track and there is no update we skip this track
            continue
        bbox = track.to_tlbr()
        class_name= track.get_class()
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]

        #draw rectangle
        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[3]+20)), (int(bbox[0]) + (len(class_name)
                + len(str(track.track_id))) * 20, int(bbox[3])), color, -1) #box for the text
        #put Text with track id
        cv2.putText(img, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[3]+14)), 0, 0.75,
                    (255, 255, 255), 2)

        center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
        pts[track.track_id].append(center)  #know the center points

        for j in range(1, len(pts[track.track_id])):    #draw line of previous centers
            if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                continue
            thickness = int(np.sqrt(64/float(j+1))*2)
            cv2.line(img, (pts[track.track_id][j-1]), (pts[track.track_id][j]), color, thickness)

        height, width, _ = img.shape    #generalize the detection zone
        cv2.line(img, (0, int(3*height/6+height/20)), (width, int(3*height/6+height/20)), (0, 255, 0), thickness=2) #down line
        cv2.line(img, (0, int(3*height/6-7*height/20)), (width, int(3*height/6-7*height/20)), (0, 255, 0), thickness=2) #upper line

        center_y = center[1]
        #if the center is between the lines then we count it
        if center_y <= int(3*height/6+height/20) and center_y >= int(3*height/6-7*height/20):
            if class_name == 'car':
                counter.append(int(track.track_id))
                current_count += 1
    #measure the length of the count
    total_count = len(set(counter))

    #display name, count
    cv2.putText(img, "Ted Baptista", (850, 100), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 2)
    cv2.putText(img, str(total_count), (1700,100), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), 2)

    #calculate and display fps while it runs on the cpu
    fps = 1./(time.time()-t1)
    cv2.putText(img, "FPS: {:.2f}".format(fps), (0,190), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
    cv2.imshow('output_drone', img)
    out.write(img)

    if cv2.waitKey(1) == ord('q'):  #stop if I press q
        break
#release Video input and output and close all windows
vid.release()
out.release()
cv2.destroyAllWindows()