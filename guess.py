# import the necessary packages
import json
import os
import random
import numpy as np

import cv2 as cv
import keras.backend as K
import numpy as np
import scipy.io

from PIL import Image
import requests
from io import BytesIO

from resnets_utils import load_model

if __name__ == '__main__':
    img_width, img_height = 224, 224
    model = load_model_and_weights()
    model.load_weights('./weights.best.hdf5')

    cars_meta = scipy.io.loadmat('devkit/cars_meta')
    class_names = cars_meta['class_names']  # shape=(1, 196)
    class_names = np.transpose(class_names)

    url = "https://picolio.auto123.com/12photo/bmw/2012-bmw-m3_2.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    results = []
    print('Processing image')
    img = cv.resize(np.float32(img), (img_width, img_height), cv.INTER_CUBIC)
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    rgb_img = np.expand_dims(rgb_img, 0)
    preds = model.predict(rgb_img)
    prob = np.max(preds)
    class_id = np.argmax(preds)
    text = ('Predict: {}, prob: {}'.format(class_names[class_id][0][0], prob))
    results.append({'label': class_names[class_id][0][0], 'prob': '{:.4}'.format(prob)})
    cv.imwrite('images/{}_out.png', img)

    print(results)
    with open('results.json', 'w') as file:
        json.dump(results, file, indent=4)

    K.clear_session()
