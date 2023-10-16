# fiberseg
Interactive zone for Fiber Segmentation
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sqbqamar/fiberseg/master?labpath=Single_image_prediction.ipynb)



In your notebook file:

Load the trained model by specifying the weight path:

model = YOLO('path/to/your/trained/model') 


 

Pass the input to the model by adding the following code:

input_data = cv2.imread("Path/to/your image")

