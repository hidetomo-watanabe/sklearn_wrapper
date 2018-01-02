import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


class ImageController(object):
    def __init__(self):
        self.vgg16_model = VGG16(weights='imagenet', include_top=False)

    def extract_features_with_vgg16(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = self.vgg16_model.predict(x).flatten()
        return features


if __name__ == '__main__':
    pass
