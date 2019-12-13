import math
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import cv2
from keras.preprocessing import image
from skimage import transform, util
from logging import getLogger

logger = getLogger('predict').getChild('ImageDataTranslater')
try:
    from .CommonDataTranslater import CommonDataTranslater
except ImportError:
    logger.warning(
        'IN FOR KERNEL SCRIPT, CommonDataTranslater import IS SKIPPED')


class ImageDataTranslater(CommonDataTranslater):
    def __init__(self, kernel=False):
        self.kernel = kernel
        self.configs = {}

    def translate_data_for_view(self):
        pass

    def write_data_for_view(self):
        savename = self.configs['pre'].get('savename')
        if savename:
            logger.warning('WRITE DATA FOR VIEW OF IMAGE IS NOT IMPLEMENTED')
            return

    def create_data_for_model(self):
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
            img = img.astype('float32')
            return img

        train_df = self.train_df
        test_df = self.test_df
        pred_df = self.pred_df
        img_path_col = img_config['img_path']
        # Y_train
        self.Y_train = pred_df.values
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
        # for data augmentation
        self.org_X_train = self.X_train
        self.org_Y_train = self.Y_train
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

    def _augment_data_for_model_with_horizontal_flip(self):
        aug_conf = self.configs['pre']['image'].get('augmentation')
        if not aug_conf:
            return
        if not aug_conf.get('flip'):
            return

        def _flip(array):
            return array[:, ::-1, :]

        logger.info('data augmentation with horizontal flip')
        aug_array = np.array(list(map(_flip, self.org_X_train)))
        self.X_train = np.append(self.X_train, aug_array, axis=0)
        self.Y_train = np.append(self.Y_train, self.org_Y_train, axis=0)
        return

    def _augment_data_for_model_with_rotation(self):
        aug_conf = self.configs['pre']['image'].get('augmentation')
        if not aug_conf:
            return
        if not aug_conf.get('rotate'):
            return

        def _rotate(array):
            np.random.seed(seed=42)
            return transform.rotate(
                array, angle=np.random.randint(-15, 15),
                resize=False, center=None)

        logger.info('data augmentation with rotation')
        aug_array = np.array(list(map(_rotate, self.org_X_train)))
        self.X_train = np.append(self.X_train, aug_array, axis=0)
        self.Y_train = np.append(self.Y_train, self.org_Y_train, axis=0)
        return

    def _augment_data_for_model_with_noize(self):
        aug_conf = self.configs['pre']['image'].get('augmentation')
        if not aug_conf:
            return
        if not aug_conf.get('noize'):
            return

        def _add_noize(array):
            return util.random_noise(array)

        logger.info('data augmentation with noize')
        aug_array = np.array(list(map(_add_noize, self.org_X_train)))
        self.X_train = np.append(self.X_train, aug_array, axis=0)
        self.Y_train = np.append(self.Y_train, self.org_Y_train, axis=0)
        return

    def _augment_data_for_model_with_invertion(self):
        aug_conf = self.configs['pre']['image'].get('augmentation')
        if not aug_conf:
            return
        if not aug_conf.get('invert'):
            return

        def _invert(array):
            # float32 => int => float32
            output = (np.invert((array * 255).astype(int)) / 255)
            output = output.astype('float32')
            return output

        logger.info('data augmentation with invertion')
        aug_array = np.array(list(map(_invert, self.org_X_train)))
        self.X_train = np.append(self.X_train, aug_array, axis=0)
        self.Y_train = np.append(self.Y_train, self.org_Y_train, axis=0)
        return

    def translate_data_for_model(self):
        self._normalize_data_for_model()
        self._augment_data_for_model_with_horizontal_flip()
        self._augment_data_for_model_with_rotation()
        self._augment_data_for_model_with_noize()
        self._augment_data_for_model_with_invertion()
        return
