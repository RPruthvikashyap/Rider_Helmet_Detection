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

# Open file dialog to select video file
file_path = filedialog.askopenfilename()

# Check if file was selected
if file_path:
    # Update cap variable with selected video file
    cap = cv2.VideoCapture(file_path)
else:
    print("No file selected")

while 1:
    ret, img = cap.read()
    if not ret:
        break
    img = cv2.resize(img, (1280, 720))
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

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            accuracy = str(round(confidences[i]*100, 2)) + "%"
            color = (0, 255, 0) # Default color is green for wearing helmet
            
            if label == 'Not Wearing Helmet':
                color = (0, 0, 255) # Change color to red for not wearing helmet
                
                # extract face region
                face_img = img[y:y+h, x:x+w]
                
                # save image to folder
                count = len(os.listdir("violations"))
                if face_img.size > 0:
                    cv2.imwrite("violations/violation_{}.jpg".format(count), face_img)

            
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + accuracy, (x, y-10), font, 2, color, 2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows() 



