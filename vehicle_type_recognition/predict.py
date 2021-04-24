import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model


# Create an index of class names
class_names = ['Ambulance', 'Barge', 'Bicycle', 'Boat', 'Bus', 'Car', 'Cart', 'Caterpillar',
    'Helicopter', 'Limousine', 'Motorcycle', 'Segway', 'Snowmobile', 'Tank', 'Taxi', 'Truck', 'Van']

model1 = load_model('mobilenet_v2.h5')
model2 = load_model('inception_v3.h5')
model3 = load_model('densenet.h5')

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


def predict(filename):
    # Load the image
    img = plt.imread(filename)
    # Resize it to the net input size:
    img = cv2.resize(img, (224, 224))
    img = img[np.newaxis, ...]

    # Convert the data to float:
    img = img.astype(np.float32)

    # Predict class by picking the highest probability index
    # then add 1 (due to indexing behavior)
    class_index = ensemble_predictions(models, img)[0]

    # Convert class id to name
    label = class_names[class_index]

    return label


if __name__ == "__main__":
    images = os.listdir("photos")
    for filename in images:
        print(predict("photos/"+filename))
