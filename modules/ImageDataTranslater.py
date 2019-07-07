import os
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import cv2
from keras.preprocessing import image
from IPython.display import display
from logging import getLogger

logger = getLogger('predict').getChild('ImageDataTranslater')
try:
    from .ConfigReader import ConfigReader
except Exception:
    logger.warn('IN FOR KERNEL SCRIPT, ConfigReader import IS SKIPPED')


class ImageDataTranslater(ConfigReader):
    def __init__(self, kernel=False):
        self.kernel = kernel
        self.BASE_PATH = '%s/..' % os.path.dirname(os.path.abspath(__file__))
        if self.kernel:
            self.OUTPUT_PATH = '.'
        else:
            self.OUTPUT_PATH = '%s/outputs' % self.BASE_PATH
        self.configs = {}

    def display_data(self):
        # to do
        if self.configs['pre']['train_mode'] == 'clf':
            logger.info('train pred counts')
            for pred_col in self.pred_cols:
                logger.info('%s:' % pred_col)
                if pred_col not in self.train_df.columns:
                    logger.warn('NOT %s IN TRAIN DF' % pred_col)
                    continue
                display(self.train_df[pred_col].value_counts())
                display(self.train_df[pred_col].value_counts(normalize=True))
        elif self.configs['pre']['train_mode'] == 'reg':
            logger.info('train pred mean, std')
            for pred_col in self.pred_cols:
                logger.info('%s:' % pred_col)
                display('mean: %f' % self.train_df[pred_col].mean())
                display('std: %f' % self.train_df[pred_col].std())
        else:
            logger.error('TRAIN MODE SHOULD BE clf OR reg')
            raise Exception('NOT IMPLEMENTED')
        for label, df in [('train', self.train_df), ('test', self.test_df)]:
            logger.info('%s:' % label)
            display(df.head())
            display(df.describe(include='all'))
        return

    def get_data_for_view(self):
        output = {
            'train_df': self.train_df,
            'test_df': self.test_df,
        }
        return output

    def create_data_for_view(self):
        train_path = self.configs['data']['train_path']
        test_path = self.configs['data']['test_path']
        delim = self.configs['data'].get('delimiter')
        if delim:
            self.train_df = pd.read_csv(train_path, delimiter=delim)
            self.test_df = pd.read_csv(test_path, delimiter=delim)
        else:
            self.train_df = pd.read_csv(train_path)
            self.test_df = pd.read_csv(test_path)
        return

    def translate_data_for_view(self):
        pass

    def write_data_for_view(self):
        savename = self.configs['pre'].get('savename')
        if savename:
            logger.warn('WRITE DATA FOR VIEW OF IMAGE IS NOT IMPLEMENTED')
            return

    def get_data_for_model(self):
        output = {
            'feature_columns': self.feature_columns,
            'test_ids': self.test_ids,
            'X_train': self.X_train,
            'Y_train': self.Y_train,
            'X_test': self.X_test,
        }
        return output

    def get_pre_processers(self):
        output = {
        }
        if hasattr(self, 'y_scaler'):
            output['y_scaler'] = self.y_scaler
        return output

    def get_post_processers(self):
        output = {}
        if hasattr(self, 'y_scaler'):
            output['y_scaler'] = self.y_scaler
        return output

    def create_data_for_model(self):
        img_config = self.configs['pre']['image']

        def _translate_image2array(img_path):
            img = cv2.imread(img_path)
            resize_param = img_config['resize']
            if resize_param:
                img = cv2.resize(
                    img, dsize=(resize_param['x'], resize_param['y']))
            img = image.img_to_array(img)
            img = img / 255
            return img

        train_df = self.train_df
        test_df = self.test_df
        img_path_col = img_config['img_path']
        # Y_train
        self.Y_train = train_df[self.pred_cols].values
        # X_train
        self.X_train = []
        for img_path in train_df[img_path_col].values:
            self.X_train.append(_translate_image2array(img_path))
        self.X_train = np.array(self.X_train)
        # X_test
        self.test_ids = test_df[self.id_col].values
        self.X_test = []
        for img_path in test_df[img_path_col].values:
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

    def translate_data_for_model(self):
        self._normalize_data_for_model()
        return
