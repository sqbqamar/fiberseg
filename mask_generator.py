# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:29:21 2024

@author: drsaq
"""

import os
import cv2
from ultralytics import YOLO

# Load the image
img = cv2.imread('C:/Users/drsaq/Downloads/131.jpg')

# Create the YOLOv8 model
model = YOLO("C:/Users/drsaq/Downloads/best.pt")

import numpy as np
from PIL import Image


results = model.predict(img, overlap_mask=False)

bboxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")

i = 1
for x in results[0].masks.cpu().data.numpy():
    # Convert mask to single channel image
    #mask_raw = results[0].masks[3].cpu().data.numpy().transpose(1, 2, 0)
    # Convert single channel grayscale to 3 channel image
    mask_3channel = cv2.merge((x, x, x))
    # Get the size of the original image (height, width, channels)
    h2, w2, c2 = results[0].orig_img.shape
    # Resize the mask to the same size as the image
    mask = cv2.resize(mask_3channel, (w2, h2))
    # Convert BGR to HSV
    hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
    # Define range of brightness in HSV
    lower_black = np.array([0,0,0])
    upper_black = np.array([0,0,1])
    # Create a mask. Threshold the HSV image to get everything black
    mask = cv2.inRange(mask, lower_black, upper_black)
    # Invert the mask to get everything but black
    mask = cv2.bitwise_not(mask)
    # Apply the mask to the original image
    masked = cv2.bitwise_or(results[0].orig_img, results[0].orig_img, mask=mask)

    # Save the images
    mask_filename = f"{i}.png"
    mask_path = os.path.join("C:/Users/drsaq/Downloads/Fiber/cmasks", mask_filename)
    cv2.imwrite(mask_path, mask)
    
    i += 1
    #cv2.imwrite("binmask.jpg", mask)
    #cv2.imwrite("clrmask.jpg", masked)
    #cv2.destroyAllWindows()