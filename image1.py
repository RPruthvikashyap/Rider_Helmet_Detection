import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import tkinter as tk
from tkinter import filedialog

# create folder for storing violation images
if not os.path.exists('C:/Users/pruth/Downloads/dataSet/violations'):
    os.makedirs('violations')

# provide the path for testing cofing file and tained model form colab
net = cv2.dnn.readNetFromDarknet("yolov3_custom.cfg", r"yolov3_video_4000.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

### Change here for custom classes for trained model 
classes = ['Wearing Helmet', 'Not Wearing Helmet']

# Set colors for bounding boxes
colors = [(0, 255, 0), (0, 0, 255)]

root = tk.Tk()
root.withdraw()

# Open file dialog to select image file
file_path = filedialog.askopenfilename()

# Check if file was selected
if file_path:
    # Read the image file
    img = cv2.imread(file_path)
    # Resize the image if needed
    img = cv2.resize(img, (1280, 720))
else:
    print("No file selected")

hight, width, _ = img.shape
blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

net.setInput(blob)

output_layers_name = net.getUnconnectedOutLayersNames()

layerOutputs = net.forward(output_layers_name)

boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
    for detection in output:
        score = detection[5:]
        class_id = np.argmax(score)
        confidence = score[class_id]
        if confidence > 0.7:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * hight)
            w = int(detection[2] * width)
            h = int(detection[3] * hight)
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)


indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)

boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
    for detection in output:
        score = detection[5:]
        class_id = np.argmax(score)
        confidence = score[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * hight)
            w = int(detection[2] * width)
            h = int(detection[3] * hight)

            x = int(center_x - w/2)
            y = int(center_y - h/2)

            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, .8, .4)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(boxes), 3))

#Draw bounding boxes and labels on the image
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        confidence = confidences[i]
        color = colors[i]
cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
cv2.putText(img, f"{label} {confidence:.2f}", (x, y-5), font, 1, color, 2)

#Show the image with bounding boxes and labels
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
