import sys
import os
import traceback
import logging.config
from logging import getLogger
import pandas as pd
BASE_PATH = '%s/../..' % os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_PATH)
from modules.DataTranslater import DataTranslater
from modules.Predicter import Predicter
from modules.Notifier import Notifier

if __name__ == '__main__':
    logging.config.fileConfig(
        '../../../configs/logging.conf',
        disable_existing_loggers=False)
    logger = getLogger('predict')

    logger.info('# START')

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = '%s/configs/config.json' % BASE_PATH
    try:
        # data translate
        translater_obj = DataTranslater()
        translater_obj.read_config_file(config_path)

        logger.info('### DATA FOR VIEW')
        translater_obj.create_data_for_view()
        translater_obj.display_data()
        data_for_view = translater_obj.get_data_for_view()

        def _calc_proba_df(translater_obj):
            logger.info('### TRANSLATE DATA FOR VIEW')
            translater_obj.translate_data_for_view()
            translater_obj.display_data()

            logger.info('### DATA FOR MODEL')
            translater_obj.create_data_for_model()
            translater_obj.normalize_data_for_model()
            translater_obj.reduce_dimension_of_data_for_model()
            data_for_model = translater_obj.get_data_for_model()

            # predict
            predicter_obj = Predicter(**data_for_model)
            predicter_obj.read_config_file(config_path)

            logger.info('### VALIDATE')
            predicter_obj.is_ok_with_adversarial_validation()

            logger.info('### FIT')
            predicter_obj.calc_ensemble_model()

            logger.info('### PREDICT')
            predicter_obj.predict_y()
            _, Y_pred_proba_df = predicter_obj.calc_predict_df()

            return Y_pred_proba_df

        def _fill_proba(proba):
            all_classes = [
                'class_6', 'class_15', 'class_16',
                'class_42', 'class_52', 'class_53',
                'class_62', 'class_64', 'class_65',
                'class_67', 'class_88', 'class_90',
                'class_92', 'class_95', 'class_99'
            ]
            for label in all_classes:
                if label not in proba.columns:
                    proba[label] = 0
            return proba

        logger.info('### DATA SEPARATION')
        train_df = data_for_view['train_df']
        test_df = data_for_view['test_df']

        logger.info('##### ISNULL DISTMOD')
        translater_obj.train_df = train_df[train_df['distmod'].isnull()]
        translater_obj.test_df = test_df[test_df['distmod'].isnull()]
        isnull_proba_df = _calc_proba_df(translater_obj)

        logger.info('##### NOTNULL DISTMOD')
        translater_obj.train_df = train_df[train_df['distmod'].notnull()]
        translater_obj.test_df = test_df[test_df['distmod'].notnull()]
        notnull_proba_df = _calc_proba_df(translater_obj)

        logger.info('### DATA INTEGRATION')
        all_proba_df = pd.concat(
            [_fill_proba(isnull_proba_df), _fill_proba(notnull_proba_df)],
            ignore_index=True)

        # predict
        predicter_obj = Predicter(**{
            'feature_columns': [],
            'test_ids': [],
            'X_train': [],
            'Y_train': [],
            'X_test': [],
        })
        predicter_obj.read_config_file(config_path)
        predicter_obj.Y_pred_proba_df = all_proba_df

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
