FOR OBJECT TRACKING:

1. Download weights from links

# yolov3 - for Windows
wget https://pjreddie.com/media/files/yolov3.weights -O weights/yolov3.weights

# yolov3-tiny - for Linux
wget https://pjreddie.com/media/files/yolov3-tiny.weights -O weights/yolov3-tiny.weights

2. After that run run convert.py to convert weihgts to .tf files

3. Set correct paths in OBJECT_TRACKING SCRIPT.py for coco.names, yolo3.tf file, and video.


FOR OBJECT DETECTINON:

1.Same as step 1 for object tracking

2. Download cfg fodler and object_detection_video.py script

3. Set correct paths in object_detection_video.py script for coco.names file, yolo3.weights file, yolo3.cfg file, and video.
