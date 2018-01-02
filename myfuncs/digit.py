import os
import numpy as np
from PIL import Image
from ImageController import ImageController

N = 28


def _save_tmp_png(ndarray, label):
    data = []
    for i in range(N):
        data.append(ndarray[i * N: (i + 1) * N])
    data = np.uint8(np.array(data))
    img = Image.fromarray(data, 'L')
    img_path = '/tmp/tmp_%s.png' % label
    img.save(img_path)
    return img_path


def _rm_tmp_png(img_path):
    os.remove(img_path)


def extract_features(dfs, _):
    image_controller_obj = ImageController()
    for df in dfs:
        for i in range(len(df)):
            ndarray = []
            for j in range(784):
                ndarray.append(df['pixel%s' % j].values[i])
            ndarray = np.array(ndarray)
            img_path = _save_tmp_png(ndarray, i)
            features = image_controller_obj.extract_features_with_vgg16(
                img_path)
            for j in range(len(features)):
                if 'feature%s' % j not in df.columns:
                    df['feature%s' % j] = [0] * len(df)
                df['feature%s' % j].values[i] = features[j]
            _rm_tmp_png(img_path)
        for j in range(784):
            del df['pixel%s' % j]
    return dfs


if __name__ == '__main__':
    pass
