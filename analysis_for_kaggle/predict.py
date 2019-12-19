import os
import sys
import traceback
from memory_profiler import profile
import logging.config
from logging import getLogger
from modules.DataTranslater import DataTranslater
from modules.Trainer import Trainer
from modules.Outputer import Outputer
from modules.Notifier import Notifier


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
logging.config.fileConfig(
    f'{BASE_PATH}/configs/logging.conf',
    disable_existing_loggers=False)
logger = getLogger('predict')


@profile
def main(config_path):
    # data translate
    translater_obj = DataTranslater()
    translater_obj.read_config_file(config_path)
    translater_obj.get_translater()

    logger.info('### DATA FOR VIEW')
    translater_obj.create_data_for_view()
    # translater_obj.display_data()

    logger.info('### TRANSLATE DATA FOR VIEW')
    translater_obj.translate_data_for_view()
    # translater_obj.display_data()

    logger.info('### WRITE DATA FOR VIEW')
    translater_obj.write_data_for_view()

    logger.info('### DATA FOR MODEL')
    translater_obj.create_data_for_model()

    logger.info('### TRANSLATE DATA FOR MODEL')
    translater_obj.translate_data_for_model()

    logger.info('### GET DATA FOR MODEL')
    data_for_model = translater_obj.get_data_for_model()
    post_processers = translater_obj.get_post_processers()

    # train
    trainer_obj = Trainer(**data_for_model)
    trainer_obj.read_config_file(config_path)

    logger.info('### FIT')
    trainer_obj.calc_estimator()

    logger.info('### WRITE ESTIMATOR DATA')
    trainer_obj.write_estimator_data()

    logger.info('### GET ESTIMATOR DATA')
    estimator_data = trainer_obj.get_estimator_data()

    # output
    outputer_obj = Outputer(
        **data_for_model, **estimator_data, **post_processers)
    outputer_obj.read_config_file(config_path)

    logger.info('### PREDICT')
    outputer_obj.predict_y()
    outputer_obj.calc_predict_df()

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
