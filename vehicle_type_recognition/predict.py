import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from flask import current_app


# Create an index of class names
class_names = ['Ambulance', 'Barge', 'Bicycle', 'Boat', 'Bus', 'Car', 'Cart', 'Caterpillar',
    'Helicopter', 'Limousine', 'Motorcycle', 'Segway', 'Snowmobile', 'Tank', 'Taxi', 'Truck', 'Van']

model1 = load_model(os.path.join(current_app.config["MODELS_PATH"], 'mobilenet_v2.h5'))
model2 = load_model(os.path.join(current_app.config["MODELS_PATH"], 'inception_v3.h5'))
model3 = load_model(os.path.join(current_app.config["MODELS_PATH"], 'densenet.h5'))

models = [model1, model2, model3]

def ensemble_predictions(members, testX):
    # make predictions
	yhats = [model.predict(testX) for model in members]
	yhats = np.array(yhats)
	# sum across ensemble members
	summed = np.sum(yhats, axis=0)
	# argmax across classes
	result = np.argmax(summed, axis=1)
	return result


def predict(numpy_img):
    # Resize it to the net input size:
    numpy_img = cv2.resize(numpy_img, (224, 224))
    numpy_img = numpy_img[np.newaxis, ...]

    # Convert the data to float:
    numpy_img = numpy_img.astype(np.float32)

    # Predict class by picking the highest probability index
    # then add 1 (due to indexing behavior)
    class_index = ensemble_predictions(models, numpy_img)[0]

    # Convert class id to name
    label = class_names[class_index]

    return label


if __name__ == "__main__":
    img_folder = "photos"
    img_list = os.listdir(img_folder)
    for img_name in img_list:
        numpy_img = plt.imread(os.path.join(img_folder, img_name))
        print(predict(numpy_img))
