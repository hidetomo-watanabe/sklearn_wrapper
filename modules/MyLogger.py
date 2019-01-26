from logging import getLogger, StreamHandler, Formatter, DEBUG


class MyLogger(object):
    def __init__(self):
        pass

    def get_logger(self, name):
        logger = getLogger(name)
        handler = StreamHandler()
        formatter = Formatter(
            '%(asctime)s:%(lineno)d:%(levelname)s:%(message)s')
        handler.setLevel(DEBUG)
        handler.setFormatter(formatter)
        logger.setLevel(DEBUG)
        logger.addHandler(handler)
        logger.propagate = False
        return logger
