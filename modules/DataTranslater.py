from logging import getLogger

logger = getLogger('predict').getChild('DataTranslater')
try:
    from .ConfigReader import ConfigReader
except Exception:
    logger.warn('IN FOR KERNEL SCRIPT, ConfigReader import IS SKIPPED')
try:
    from .TableDataTranslater import TableDataTranslater
except Exception:
    logger.warn('IN FOR KERNEL SCRIPT, TableDataTranslater import IS SKIPPED')
try:
    from .ImageDataTranslater import ImageDataTranslater
except Exception:
    logger.warn('IN FOR KERNEL SCRIPT, ImageDataTranslater import IS SKIPPED')


class DataTranslater(ConfigReader):
    def __init__(self, kernel=False):
        self.kernel = kernel

    def get_translater(self):
        data_type = self.configs['data']['type']
        if data_type == 'table':
            self.translater = TableDataTranslater(self.kernel)
        elif data_type == 'image':
            self.translater = ImageDataTranslater(self.kernel)
        else:
            logger.error('DATA MODE SHOULD BE table OR image')
            raise Exception('NOT IMPLEMENTED')

        # take over instance variable
        self.translater.__dict__.update(self.__dict__)
        return

    def display_data(self):
        self.translater.display_data()
        return

    def get_data_for_view(self):
        return self.translater.get_data_for_view()

    def create_data_for_view(self):
        self.translater.create_data_for_view()
        return

    def translate_data_for_view(self):
        self.translater.translate_data_for_view()
        return

    def write_data_for_view(self):
        return self.translater.write_data_for_view()

    def get_data_for_model(self):
        return self.translater.get_data_for_model()

    def get_pre_processers(self):
        return self.translater.get_pre_processers()

    def get_post_processers(self):
        return self.translater.get_post_processers()

    def create_data_for_model(self):
        self.translater.create_data_for_model()
        return

    def translate_data_for_model(self):
        self.translater.translate_data_for_model()
        return
