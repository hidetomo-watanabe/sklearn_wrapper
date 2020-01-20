from logging import getLogger

logger = getLogger('predict').getChild('DataTranslater')
try:
    from .ConfigReader import ConfigReader
except ImportError:
    logger.warning('IN FOR KERNEL SCRIPT, ConfigReader import IS SKIPPED')


class DataTranslater(ConfigReader):
    def __init__(self, kernel=False):
        self.kernel = kernel

    def get_translater(self):
        data_type = self.configs['data']['type']
        if data_type == 'table':
            if not self.kernel:
                from .TableDataTranslater import TableDataTranslater
            self.translater = TableDataTranslater(self.kernel)
        elif data_type == 'image':
            if not self.kernel:
                from .ImageDataTranslater import ImageDataTranslater
            self.translater = ImageDataTranslater(self.kernel)
        else:
            logger.error('DATA MODE SHOULD BE table OR image')
            raise Exception('NOT IMPLEMENTED')

        # take over instance variable
        self.translater.__dict__.update(self.__dict__)
        return

    def get_raw_data(self):
        return self.translater.get_raw_data()

    def calc_train_data(self):
        return self.translater.calc_train_data()

    def write_train_data(self):
        return self.translater.write_train_data()

    def get_train_data(self):
        return self.translater.get_train_data()

    def get_pre_processers(self):
        return self.translater.get_pre_processers()

    def get_post_processers(self):
        return self.translater.get_post_processers()
