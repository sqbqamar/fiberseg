# Yolov8 Flask API for Fiber Segmentation

This code is based on the YOLOv8 from Ultralytics and it has all the functionalities that the original code has:
- Source: images
- Supported weight: Pytorch

## Requirements

Python 3.8 or later with all [requirements.txt](requirements.txt) dependencies installed, including `torch>=1.7`. To install run:

```bash
$ pip install -r requirements.txt
```
## Train YOLOv8 with Custom Dataset
Just open the Train_custom_data.ipynb file on Google Colab or the local system and follow the instructions as written in the file.  
[Open Training file in Google Colab] (https://colab.research.google.com/github/sqbqamar/fiberseg/blob/main/Train_custom_data.ipynb)

## Prediction by trained model

`prediction.py` can deal with images and can run into the CPU, but it is highly recommendable to run in GPU.

```bash
Usage - sources:
    $ python prediction.py --weights best.pt --source 'image/131.jpg'                          
 ```   
[Open Prediction file in Google Colab] (https://colab.research.google.com/github/sqbqamar/fiberseg/blob/main/prediction_file.ipynb)

## Interactive implementation

You can deploy the API to label in an interactive way.

Run:

```bash
$ python app.py 
```
Open the application in any browser 0.0.0.0:5000 and upload your image. The API will return the image labeled.



## Run code with Binder

Just click on the binder link, and it automatically installs all the required libraries and opens the prediction_file.ipynb. 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sqbqamar/fiberseg/master?labpath=prediction_file.ipynb)




In the prediction_file.ipynb file:

1- Load the trained model by specifying the weight path:

model = YOLO('path/to/your/trained/model') 


 

2- Pass the input to the model by adding the image path in the following code:

input_data = cv2.imread("Path/to/your image")


## Conversion scale 
We set the conversion ratio of each pixel 
1 px = 0.425 mm

