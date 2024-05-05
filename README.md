# Deep Learning for Fiber Segmentation

This code is based on the YOLOv8 from Ultralytics and it has all the functionalities that the original code has:
- Source: images
- Supported weight: Pytorch

## Requirements
First, create a conda environment to install the required libraries. Use the 'Fiberseg_env.yml' file to install the necessary libraries.
```bash
$ conda env create -f Fiberseg_env.yml

$ conda activate fiberseg
```
## Train YOLOv8 with plant dataset
Just open the Train_custom_data.ipynb file on Google Colab or the local system and follow the instructions as written in the file.  
[Open Training file in Google Colab] (https://colab.research.google.com/github/sqbqamar/fiberseg/blob/main/Train_custom_data.ipynb)

Specially, set `overlap_mask=False` to get the individual mask for each instance, even when instances overlap with each other.

## Prediction by trained model

`prediction.py` can deal with images and can run into the CPU, but it is highly recommendable to run in GPU.

```bash
Usage - sources:
    $ python prediction.py --weights best.pt --source 'image/131.jpg' --mask --conversion 0.65                       
 ```
In command line arguments, we load the model with the specified weights using the `--weight` argument and `--source` to specify the input image variable. The `--mask` argument is used to generate the masks and save them in the folder. The `--conversion` argument is used as the conversion factor when converting pixel values to standard units.

[Open Prediction file in Google Colab] (https://colab.research.google.com/github/sqbqamar/fiberseg/blob/main/prediction_file.ipynb)

## Generate individual masks and save them to the designated folder

`mask_generator.py` generates individual mask instances for the objects detected by the model. Each mask instance is positioned at the actual location of the corresponding object in the image. The size of the mask image is equal to the size of the original image. This will help in accurately calculating the length and width of the masks in pixels.

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
We did not set the conversion ratio of each pixel because the pixel size varies across different microscopy images.
In our case, we found that 1 pixel = 0.65 mm, but we could not use this ratio for calculations in the original manuscript.

You can use the `conversion.py` file to change the values in the Excel file according to your microscopic standard. Just update the conversion value on Line 13 in the code.
 

