from collections import defaultdict

import cv2

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

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

# Dictionary to store tracking history with default empty lists
track_history = defaultdict(lambda: [])

# Load the YOLO model with segmentation capabilities
model = YOLO("yolov8n-seg.pt")

# Open the video file
cap = cv2.VideoCapture("cars2.mp4")

# Retrieve video properties: width, height, and frames per second
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Initialize video writer to save the output video with the specified properties
out = cv2.VideoWriter("instance-segmentation-object-tracking.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

while True:
    # Read a frame from the video
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Create an annotator object to draw on the frame
    annotator = Annotator(im0, line_width=2)

    # Perform object tracking on the current frame
    results = model.track(im0, persist=True)
    
    # Check if tracking IDs and masks are present in the results
    if results[0].boxes.id is not None and results[0].masks is not None:
        # Extract masks and tracking IDs
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xywh.cpu()
        names = results[0].names
        classes = results[0].boxes.cls.int().cpu().tolist()
        # Annotate each mask with its corresponding tracking ID and color
        for mask, track_id,box,classe in zip(masks, track_ids,boxes,classes):
            x,y,w,h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            #print(names[classe])
            if (len(track) > 1): #Verificando a direção do veículo
                lasts = track[-2:]
                if (len(lasts) > 1):
                    x1, y1 = lasts[-2]
                    x2, y2 = lasts[-1]
                    if (y1-y2 <= 0):
                        annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), label=str(track_id))
            
    # Write the annotated frame to the output video
    out.write(im0)
    # Display the annotated frame
    cv2.imshow("instance-segmentation-object-tracking", im0)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video writer and capture objects, and close all OpenCV windows
out.release()
cap.release()
cv2.destroyAllWindows()