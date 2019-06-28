import sys
import traceback
import logging.config
from logging import getLogger
from modules.TableDataTranslater import TableDataTranslater
from modules.Predicter import Predicter
from modules.Notifier import Notifier

if __name__ == '__main__':
    logging.config.fileConfig(
        './configs/logging.conf', disable_existing_loggers=False)
    logger = getLogger('predict')

    logger.info('# START')

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = './configs/config.json'
    try:
        # data translate
        translater_obj = TableDataTranslater()
        translater_obj.read_config_file(config_path)

        logger.info('### DATA FOR VIEW')
        translater_obj.create_data_for_view()
        translater_obj.display_data()

        logger.info('### TRANSLATE DATA FOR VIEW')
        translater_obj.translate_data_for_view()
        translater_obj.display_data()

        logger.info('### WRITE DATA FOR VIEW')
        translater_obj.write_data_for_view()

        logger.info('### DATA FOR MODEL')
        translater_obj.create_data_for_model()

        logger.info('### TRANSLATE DATA FOR MODEL')
        translater_obj.translate_data_for_model()

        logger.info('### GET DATA FOR MODEL')
        data_for_model = translater_obj.get_data_for_model()
        post_processers = translater_obj.get_post_processers()

        # predict
        predicter_obj = Predicter(**data_for_model, **post_processers)
        predicter_obj.read_config_file(config_path)

        logger.info('### FIT')
        predicter_obj.calc_ensemble_model()

        logger.info('### WRITE ESTIMATOR DATA')
        predicter_obj.write_estimator_data()

        logger.info('### PREDICT')
        predicter_obj.predict_y()
        predicter_obj.calc_predict_df()

        logger.info('### WRITE PREDICT DATA')
        predicter_obj.write_predict_data()

    except Exception as e:
        logger.error('%s' % e)
        traceback.print_exc()
    finally:
        notifier_obj = Notifier()
        notifier_obj.read_config_file(config_path)
        notifier_obj.notify_slack()
    logger.info('# FINISHED')
