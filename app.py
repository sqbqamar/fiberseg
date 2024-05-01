# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:20:26 2023

@author: SAQIBQ
"""

from flask import Flask, render_template, Response, request, send_file
import tempfile
import json
import pandas as pd
import os
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import colorsys
import random
from datetime import datetime
import shutil
from flask import current_app
from fil_finder import FilFinder2D
import astropy.units as u

#from segment import measure

# Create flask app
app = Flask(__name__)


def detect(model, img, imgsz):
        # Get img shape
   # height, width, channels = img.shape
    results = model.predict(source=img.copy(), project="Result", name="pred", overlap_mask=False, imgsz=imgsz, save=True, iou=0.8, conf=0.6, save_txt=False)
    result = results[0]
        
    # Extract Masks, bounding boxes, class IDs, and scores from the result
    segment = result.masks.cpu().data.numpy()
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
    scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
    
    # Return the detected bounding boxes, class IDs, segments, and scores
    return bboxes, class_ids, segment, scores

def draw_mask(img, pts, color, alpha=0.5):
    h, w, _ = img.shape

    overlay = img.copy()
    output = img.copy()

    pts_list = [np.array(pts, dtype=np.int32)]  # Convert the input `pts` to the correct format
    cv2.fillPoly(overlay, pts_list, color)
    output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 255 if bright else 180
    hsv = [(i / N + 1, 1, brightness) for i in range(N + 1)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors 

class_names = ['Fibre', 'Vessel']
colors = random_colors(len(class_names))

model = YOLO("C:/Users/drsaq/Downloads/best.pt")

def measure(model, path):
    img = cv2.imread(path)
    height, width, channels = img.shape
        
    imgsz = None
    if 500 < width < 2000:
        imgsz = 1024
    elif 2500 < width < 5000:
        imgsz = 3000
    elif 5000 < width < 9000:
        imgsz = 6000
    if imgsz is None:
        imgsz = 2048  
        
    list2=[]
    bboxes, classes, segmentations, scores = detect(model, img, imgsz)
    #bboxes, classes, segmentations, scores, area = detect(model, img, imgsz)
    for i, (bbox, class_id, seg, score) in enumerate(zip(bboxes, classes, segmentations, scores)):
    # print("bbox:", bbox, "class id:", class_id, "seg:", seg, "score:", score)
        color = colors[i]
        (x, y, x2, y2) = bbox 
        h, w = seg.shape
        mask_3channel = cv2.merge((seg, seg, seg))
        # Get the size of the original image (height, width, channels)
        h2, w2, c2 = img.shape
        # Resize the mask to the same size as the image
        x = cv2.resize(seg, (w2, h2)).astype('uint8')
        # Find contours in the mask
        d = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if d:    
            cntsSorted = sorted(d, key=lambda x: cv2.contourArea(x) , reverse =  True)
            
        if d: 
            largest_contour = cntsSorted[0] 
        mask = cv2.resize(mask_3channel, (w2, h2)).astype(int)

        ### Length Calculation 

        # Create a blank image with the same dimensions as the mask, but with 3 channels (RGB)
        drawing = np.zeros((mask.shape[0], mask.shape[1], 3))
        # Draw the largest contour from the mask onto the drawing image and fill it with green color
        cv2.drawContours(drawing, [cntsSorted[0]] , -1 , color = (0,255,0) , thickness = cv2.FILLED)
        drawing = drawing.astype(np.uint8)
        drawing = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
        # Apply the Zhang-Suen thinning algorithm to the grayscale drawing to obtain a skeleton
        thinned = cv2.ximgproc.thinning(drawing, thinningType = cv2.ximgproc.THINNING_ZHANGSUEN)
        
        skeleton = thinned
        
        # Initialize an instance of the FilFinder2D class with the skeleton, a distance parameter, and the skeleton as the mask
        fil = FilFinder2D(skeleton, distance=250 * u.pc, mask=skeleton)
        # Preprocess the image or skeleton by flattening the intensity values
        fil.preprocess_image(flatten_percent=85)
        fil.create_mask(border_masking=True, verbose=False,
        use_existing_mask=True)
        fil.medskel(verbose=False)
        fil.analyze_skeletons(branch_thresh=40* u.pix, skel_thresh=10 * u.pix, prune_criteria='length')
        # Assign the longest path or skeleton from the FilFinder2D instance to the 'mask' variable
        mask = fil.skeleton_longpath
        Length = np.sum(mask)

        #print(f' Object Length is: {length:.2f}')
        
        ### Width Calculation 

        # Compute the distance transform
        dist_transform = cv2.distanceTransform(drawing, cv2.DIST_L2, 5)

        # Find the maximum value in the distance transform
        max_dist = np.max(dist_transform)

        # The maximum value corresponds to the thickness of the thickest part of the fiber
        fiber_thickness = max_dist * 2
        Width = round(max_dist * 2, 2)

        #print(f"Object Width is: {fiber_thickness:.2f} pixels")
        
        Area = cv2.contourArea(cntsSorted[0])
        list1 = class_names[class_id], Length, Width, Area
        list2.append(list1)
        
        cnt = cntsSorted[0].reshape(-1, 2)

        img = draw_mask(img, [cnt], color)
    
    #cv2.drawContours(img, [box1], 0, (0, 0, 255), 2)
        #img = draw_mask(img, [seg], colors[class_id])
    #cv2.rectangle(img, (a, b), (c, d), (255, 0, 0), 2)
        #cv2.drawContours(img,[box1],0,(255,0,0),2)
    df = pd.DataFrame(list2)
    df1 = df.rename(columns={0: 'Class Name',1: 'Length', 2: 'Width', 3: 'Area'})
   # df1.to_excel("static/xls_file/data.xlsx")
    return img, df1


def find_most_recent_folder(directory):
    folders = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    if not folders:
        return None
    most_recent_folder = max(folders, key=os.path.getctime)

    # Remove all other folders except the most recent one
    for folder in folders:
        if folder != most_recent_folder:
            try:
                shutil.rmtree(folder)
            except OSError:
                pass

    return most_recent_folder

def find_most_recent_image_in_folder(folder):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    recent_image = None
    recent_time = 0

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(image_extensions):
                file_path = os.path.join(root, file)
                file_time = os.path.getctime(file_path)
                if file_time > recent_time:
                    recent_time = file_time
                    recent_image = file_path

    return recent_image



image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload')


@app.route('/', methods=['GET', 'POST'])
def application():
    if request.method == 'POST':
        # Take uploaded image
        upload_file = request.files['image_name']
        nowTime = datetime.now().strftime("%Y%m%d%H%M%S")
        
        filename = nowTime + '_' + upload_file.filename
        path_save = os.path.join(UPLOAD_PATH, filename)
        # Store image in upload directory
        upload_file.save(path_save)
        image_base_name = os.path.splitext(os.path.basename(path_save))[0]
        # Take image and perform OCR
        img = cv2.imread(path_save)
        if path_save:
            # Now, delete other images
            for root, dirs, files in os.walk(UPLOAD_PATH):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file_path != path_save and file.lower().endswith(image_extensions):
                        try:
                            os.remove(file_path)
                        except OSError:
                            pass
            
        img, df1 = measure(model, path_save)
        directory_path = 'static/Result' # Replace with the actual directory path
        recent_folder = find_most_recent_folder(directory_path)
        image_path = find_most_recent_image_in_folder(recent_folder)
       # image1_path = 'static/prediction/image0.jpg'
       # image1_name = os.path.basename(image_path)
        
       # print(text_roi + '\n' + text_thresh)
        xls_filename= image_base_name + "_summary.xlsx"
        df1.to_excel(os.path.join(current_app.static_folder, "xls_file", xls_filename))
        static_folder = os.path.join(current_app.static_folder, "xls_file")
        for f in os.listdir(static_folder):
            if f.endswith(".xlsx") and f != xls_filename:
                os.remove(os.path.join(static_folder, f))
        #df1.to_excel("static/xls_file/" + xls_filename)
        #xls_filename = 'static/xls_file/data.xlsx'

        return render_template('index.html', upload = True, upload_image = filename, image1 = image_path, xls_filename=xls_filename)

    return render_template('index.html', upload = False)


if __name__ == "__main__":
    #app.run()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)