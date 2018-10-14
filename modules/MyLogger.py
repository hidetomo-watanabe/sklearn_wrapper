from logging import getLogger, StreamHandler, Formatter, INFO


class MyLogger(object):
    def __init__(self):
        pass

    def get_logger(self):
        logger = getLogger(__name__)
        handler = StreamHandler()
        formatter = Formatter(
            '%(asctime)s:%(lineno)d:%(levelname)s:%(message)s')
        handler.setLevel(INFO)
        handler.setFormatter(formatter)
        logger.setLevel(INFO)
        logger.addHandler(handler)
        logger.propagate = False
        return logger


if __name__ == '__main__':
    pass
