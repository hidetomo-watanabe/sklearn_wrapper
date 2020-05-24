import math
from logging import getLogger

import cv2

from keras.preprocessing import image

import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = getLogger('predict').getChild('ImageDataTranslater')
if 'BaseDataTranslater' not in globals():
    from .BaseDataTranslater import BaseDataTranslater


class ImageDataTranslater(BaseDataTranslater):
    def __init__(self, kernel=False):
        self.kernel = kernel
        self.configs = {}

    def _calc_base_train_data(self):
        img_config = self.configs['pre']['image']

        def _translate_image2array(img_path):
            img = cv2.imread(img_path)
            # resize
            resize_param = img_config['resize']
            if resize_param:
                img = cv2.resize(
                    img, dsize=(resize_param['x'], resize_param['y']))
            # array
            img = image.img_to_array(img)
            img = img / 255
            # for memory reduction
            img = img.astype(np.float32)
            return img

        train_df = self.train_df
        test_df = self.test_df
        pred_df = self.pred_df
        train_img_dir = img_config.get('train_img_dir', '.')
        test_img_dir = img_config.get('test_img_dir', '.')
        img_path_col = img_config['img_path_col']
        img_extension = img_config.get('img_extension', '')
        # Y_train
        self.Y_train = pred_df.to_numpy()
        if self.configs['pre']['train_mode'] == 'reg':
            self.Y_train = self.Y_train.astype(np.float32)
        # X_train
        self.train_ids = train_df[self.id_col].to_numpy()
        self.X_train = []
        for img_path in train_df[img_path_col].to_numpy():
            img_path = f'{train_img_dir}/{img_path}{img_extension}'
            self.X_train.append(_translate_image2array(img_path))
        self.X_train = np.array(self.X_train)
        # X_test
        self.test_ids = test_df[self.id_col].to_numpy()
        self.X_test = []
        for img_path in test_df[img_path_col].to_numpy():
            img_path = f'{test_img_dir}/{img_path}{img_extension}'
            self.X_test.append(_translate_image2array(img_path))
        self.X_test = np.array(self.X_test)
        # feature_columns
        self.feature_columns = [img_path_col]
        return

    def _normalize_data_for_model(self):
        # y
        if self.configs['pre']['train_mode'] == 'reg':
            # pre
            y_pre = self.configs['pre'].get('y_pre')
            if y_pre:
                logger.info('translate y_train with %s' % y_pre)
                if y_pre == 'log':
                    self.Y_train = np.array(list(map(math.log, self.Y_train)))
                    self.Y_train = self.Y_train.reshape(-1, 1)
                else:
                    logger.error('NOT IMPLEMENTED FIT Y_PRE: %s' % y_pre)
                    raise Exception('NOT IMPLEMENTED')
            # scaler
            y_scaler = self.configs['pre'].get('y_scaler')
            logger.info(f'normalize y data: {y_scaler}')
            if y_scaler:
                if y_scaler == 'standard':
                    self.y_scaler = StandardScaler()
                elif y_scaler == 'minmax':
                    self.y_scaler = MinMaxScaler()
                self.y_scaler.fit(self.Y_train)
                self.Y_train = self.y_scaler.transform(self.Y_train)
            else:
                self.y_scaler = None
        return

    def calc_train_data(self):
        self._calc_raw_data()
        self._calc_base_train_data()
        self._normalize_data_for_model()
        return
