# fiberseg
Interactive zone for Fiber Segmentation


[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sqbqamar/fiberseg/master?labpath=prediction_file.ipynb)





In the prediction_file.ipynb file:

1- Load the trained model by specifying the weight path:

model = YOLO('path/to/your/trained/model') 


 

2- Pass the input to the model by adding the image path in the following code:

input_data = cv2.imread("Path/to/your image")


3- Conversion scale :

1 px = 0.65 mm

