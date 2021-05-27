from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

import numpy as np


def translate_image2feature_with_vgg16(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    vgg16_model = VGG16(weights='imagenet', include_top=False)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = vgg16_model.predict(x).flatten()
    return features
