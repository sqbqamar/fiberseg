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

#from segment import measure

# Create flask app
app = Flask(__name__)


def detect(model, img, imgsz):
        # Get img shape
   # height, width, channels = img.shape
    results = model.predict(source=img.copy(), project="static/Result", name="prediction", imgsz=imgsz, save=True, iou=0.8, conf=0.6, save_txt=False)
    result = results[0]
    segments = []
    total_area= []
    h2, w2, c2 = results[0].orig_img.shape
    for x in result.masks.cpu().data.numpy().astype('uint8'):
        x = cv2.resize(x, (w2, h2))
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contour_area = 0
        for contour in c:
            area = cv2.contourArea(contour)
            contour_area += area
        c = np.concatenate([x.reshape(-1, 2) for x in c])
        segments.append(c.astype('float32'))
        total_area.append(contour_area)
        

    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        # Get class ids
    class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        # Get scores
    scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
    #print(class_ids)
    return bboxes, class_ids, segments, scores, total_area

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

model = YOLO("model/best.pt")

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
    bboxes, classes, segmentations, scores, area = detect(model, img, imgsz)
    for bbox, class_id, seg, score, area in zip(bboxes, classes, segmentations, scores, area):
    # print("bbox:", bbox, "class id:", class_id, "seg:", seg, "score:", score)
        (x, y, x2, y2) = bbox
        rect = cv2.minAreaRect(seg)
        box1 = cv2.boxPoints(rect)
        box1 = np.int0(box1)
    #print(box1)
        a,b,c,d=box1
        dst1= ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5
        dst2 = ((b[0] - c[0])**2 + (b[1] - c[1])**2)**0.5
        dst1 = dst1 * 0.65
        dst2 = dst2 * 0.65
        if dst1 > dst2:
            Length = dst1
        else:
            Length = dst2
    #area_px = cv2.contourArea(seg)
        area_px = area * 0.65
        Width = round(area_px/Length, 2)
        list1 = class_names[class_id], Length, Width, area_px
        list2.append(list1)
    
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