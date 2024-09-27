# author : ayush-thakur02
# bio.link/ayush_thakur02
# It's an open source project so make sure you give it a like and share it with other. 
# Eductaional Purpose Only
#https://github.com/ayush-thakur02/object-tracking-opencv/tree/main

import cv2
import numpy as np
from object_detection import ObjectDetection
import math
import configparser

config = configparser.ConfigParser()
try:
    config.read('config.ini')
    CUT_IMAGE = int(config['DEFAULT']['CUT_IMAGE'])
    CUT_IMAGE_V = int(config['DEFAULT']['CUT_IMAGE_V'])
    DISTANCIA = int(config['DEFAULT']['DISTANCIA'])
except Exception as e:
    CUT_IMAGE = 3 #Quanto menor, menos corta a imagem. Ex: 2 corta pela metade. 3 corta 1/3 da imagem.
    CUT_IMAGE_V = 1
    DISTANCIA = 80  

# Initialize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture("simulador.mp4")

# Initialize count
count = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    # Point current frame
    center_points_cur_frame = []
    if CUT_IMAGE > 1:
        frame = frame[frame.shape[0]//CUT_IMAGE:frame.shape[0]]
    if CUT_IMAGE_V > 1:
        frame = frame[:,frame.shape[1]//CUT_IMAGE_V:]
    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)
    for box,class_id in zip(boxes, class_ids):
        (x, y, w, h) = box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        if (cy > frame.shape[0]//3) and (cy < (frame.shape[0]*2)//3):
            center_points_cur_frame.append((cx, cy))
            #print("FRAME N°", count, " ", x, y, w, h)
            #cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #print(class_id,od.classes[class_id])
    
    # Only at the beginning we compare previous and current frame
    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < DISTANCIA:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:

        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # Update IDs position
                if distance < DISTANCIA:
                    tracking_objects[object_id] = pt
                    #if (pt2[1] > frame.shape[0]//3) and (pt2[1] < (frame.shape[0]*2)//3):
                        #if (pt[1]-pt2[1]) < 0:
                        #    cv2.putText(frame, "Subindo...", (pt[0], pt[1] - 20), 0, 1, (0, 0, 255), 2)
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue

            # Remove IDs lost
            if not object_exists:
                tracking_objects.pop(object_id)

        # Add new IDs found
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1

    for object_id, pt in tracking_objects.items():
        if (pt[1] > frame.shape[0]//3) and (pt[1] < (frame.shape[0]*2)//3):
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
            cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

    #print("Tracking objects")
    #print(tracking_objects)


    #print("CUR FRAME LEFT PTS")
    #print(center_points_cur_frame)

    cv2.imshow("Frame", frame)

    # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
