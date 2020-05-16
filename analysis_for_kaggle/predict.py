import logging.config
import os
import sys
import traceback
from logging import getLogger

from memory_profiler import profile

from modules.DataTranslaters.DataTranslater import DataTranslater
from modules.Notifier import Notifier
from modules.Outputer import Outputer
from modules.Trainers.Trainer import Trainer


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
logging.config.fileConfig(
    f'{BASE_PATH}/configs/logging.conf',
    disable_existing_loggers=False)
logger = getLogger('predict')


@profile
def main(config_path):
    # translate
    translater_obj = DataTranslater()
    translater_obj.read_config_file(config_path)
    translater_obj.get_translater()

    logger.info('### TRANSLATE')
    translater_obj.calc_train_data()

    logger.info('### WRITE TRAIN DATA')
    translater_obj.write_train_data()

    logger.info('### GET TRAIN DATA')
    train_data = translater_obj.get_train_data()
    post_processers = translater_obj.get_post_processers()

    # train
    trainer_obj = Trainer(**train_data)
    trainer_obj.read_config_file(config_path)

    logger.info('### FIT')
    trainer_obj.calc_estimator_data()

    logger.info('### WRITE ESTIMATOR DATA')
    trainer_obj.write_estimator_data()

    logger.info('### GET ESTIMATOR DATA')
    estimator_data = trainer_obj.get_estimator_data()

    # output
    outputer_obj = Outputer(
        **train_data, **estimator_data, **post_processers)
    outputer_obj.read_config_file(config_path)

    logger.info('### PREDICT')
    outputer_obj.calc_predict_data()

    logger.info('### WRITE PREDICT DATA')
    outputer_obj.write_predict_data()
    return outputer_obj.get_predict_data()


if __name__ == '__main__':
    logger.info('# START')

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = f'{BASE_PATH}/configs/config.json'

    try:
        main(config_path)
    except Exception as e:
        logger.error('%s' % e)
        traceback.print_exc()
    finally:
        notifier_obj = Notifier()
        notifier_obj.read_config_file(config_path)
        notifier_obj.notify_slack()

    logger.info('# FINISHED')
