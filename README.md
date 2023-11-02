# Yolov8 Flask API for detection and segmentation

This code is based on the YOLOv5 from Ultralytics and it has all the functionalities that the original code has:
- Source: images.
- Weights are supported: Pytorch, Onnx.

## Requirements

Python 3.8 or later with all [requirements.txt](requirements.txt) dependencies installed, including `torch>=1.7`. To install run:

```bash
$ pip install -r requirements.txt


## Instance segmentation API

`app.py` can deal with images and can run into the CPU, but it is highly recommendable to run in GPU.

```bash
Usage - sources:
    $ python app.py --weights best.pt --source 'image/131.jpg'                          
    


## Interactive implementation

You can deploy the API to label in an interactive way.

Run:

```bash
$ python app.py --device cpu # to run into CPU (by default is GPU)
```
Open the application in any browser 0.0.0.0:5000 and upload your image.


## How to use the API

### Interactive way
Just open your favorite browser and go to 0.0.0.0:5000 and intuitevely load the image you want to label and press the buttom "submit".

The API will return the image labeled.


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
1 px = 0.65 mm

