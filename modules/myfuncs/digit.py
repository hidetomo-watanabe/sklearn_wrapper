import os
import gc
import numpy as np
from PIL import Image
from ImageController import ImageController

N = 28


def _save_tmp_png(org, label):
    data = []
    for i in range(N):
        data.append(org[i * N: (i + 1) * N])
    data = np.uint8(np.array(data))
    img = Image.fromarray(data, 'L')
    img_path = '/tmp/tmp_%s.png' % label
    img.save(img_path)
    return img_path


def _rm_tmp_png(img_path):
    os.remove(img_path)


def extract_features(train_df, test_df):
    image_controller_obj = ImageController()
    for df in [train_df, test_df]:
        for i in range(len(df)):
            pixels = []
            for j in range(N * N):
                pixels.append(df['pixel%s' % j].values[i])
            img_path = _save_tmp_png(pixels, i)
            features = image_controller_obj.extract_features_with_vgg16(
                img_path)
            for j in range(len(features)):
                if 'feature%s' % j not in df.columns:
                    df['feature%s' % j] = [0] * len(df)
                df['feature%s' % j].values[i] = features[j]
            _rm_tmp_png(img_path)
            # gc
            del pixels
            del features
            gc.collect()
    return train_df, test_df
