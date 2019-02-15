import sys
import traceback
import logging.config
from logging import getLogger
from modules.DataTranslater import DataTranslater
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
        translater_obj = DataTranslater()
        translater_obj.read_config_file(config_path)

        logger.info('### RAW DATA')
        translater_obj.create_raw_data()
        translater_obj.display_data()

        logger.info('### TRANSLATE RAW DATA')
        translater_obj.translate_raw_data()
        translater_obj.display_data()

        logger.info('##### DATA FOR MODEL')
        translater_obj.create_data_for_model()
        translater_obj.normalize_data_for_model()
        translater_obj.reduce_dimension_of_data_for_model()
        feature_columns, test_ids, X_train, Y_train, X_test = \
            translater_obj.get_data_for_model()
        scaler_y = translater_obj.get_scaler_y()

        # predict
        predicter_obj = Predicter(
            feature_columns, test_ids, X_train, Y_train, X_test, scaler_y)
        predicter_obj.read_config_file(config_path)

        # logger.info('### VISUALIZE TRAIN DATA')
        # predicter_obj.visualize_train_data()

        logger.info('### VALIDATE')
        predicter_obj.is_ok_with_adversarial_validation()

        logger.info('### FIT')
        predicter_obj.calc_ensemble_model()

        # logger.info('### VISUALIZE LEARNING CURVES')
        # predicter_obj.visualize_learning_curves()

        logger.info('### PREDICT')
        predicter_obj.calc_output()

        # logger.info('### VISUALIZE TRAIN PRED DATA')
        # predicter_obj.visualize_train_pred_data()

        logger.info('### OUTPUT')
        predicter_obj.write_output()

    except Exception as e:
        logger.error('%s' % e)
        traceback.print_exc()
    finally:
        notifier_obj = Notifier()
        notifier_obj.read_config_file(config_path)
        notifier_obj.notify_slack()
    logger.info('# FINISHED')
