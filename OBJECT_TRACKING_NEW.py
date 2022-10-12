from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)

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


class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('./weights/yolov3.tf') #Chek where this file is 

max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8

model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# Video capture
vid = cv2.VideoCapture('./data/video/01.avi')

codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps = int(vid.get(cv2.CAP_PROP_FPS))
vid_width, vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('/data/videos/results.avi', codec, vid_fps, (vid_width, vid_height)) #Output new video

while True:

    _, img = vid.read()
    if img is None:
        print("Completed")
        break
    
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Adding dimension
    img_in = tf.expand_dims(img_in, 0)
    # Reshaping imgase
    img_in = transform_images(img_in, 416)
    
    t1 = time.time()

    # boxes, 3D shape (1, 100, 4)
    # scores, 2D shape (1, 100)
    # classed, 2D shape (1, 100)
    # nums, 1D shape (1,)

    boxes, scores, classes, nums = yolo.predict(img_in)

    classes = classes[0]
    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    
    names = np.array(names)
    converted_boxes = convert_boxes(img, boxes[0])
    features = encoder(img, converted_boxes)

    #Add coment
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in 
                 zip(converted_boxes, scores[0], names, features)]

    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxes, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    tracker.predict()
    tracker.update(detections)


    #Vizualization of results
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

    for track in tracker.tracks:
        # If there is no updates in trakc skip track
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        bbox = track.to_tlbr()
        class_name = track.get_class()
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color] # To return to RGB scale


        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)
                    +len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        cv2.putText(img, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
                    (255, 255, 255), 2)
                    

    fps = 1./(time.time() - t1)
    cv2.putText(img, "FPS: {:.2f}".format(fps), (0,30), 0, 1, (0,0,255), 2)
    #cv2.resizeWindow('output', 1024, 768)
    cv2.imshow('output', img)

    out.write(img)

    if cv2.waitKey(1) == ord('q'):
        break

vid.release()
out.release()
cv2.destroyAllWindows()


        

        




    

    



    









