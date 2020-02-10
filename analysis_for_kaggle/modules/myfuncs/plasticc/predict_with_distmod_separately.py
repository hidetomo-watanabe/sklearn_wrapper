import sys
import os
import traceback
import logging.config
from logging import getLogger
import pandas as pd
BASE_PATH = '%s/../..' % os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_PATH)
from modules.DataTranslater import DataTranslater
from modules.Trainer import Trainer
from modules.Outputer import Outputer
from modules.Notifier import Notifier

if __name__ == '__main__':
    logging.config.fileConfig(
        f'{BASE_PATH}/configs/logging.conf',
        disable_existing_loggers=False)
    logger = getLogger('predict')

    logger.info('# START')

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = f'{BASE_PATH}/configs/config.json'
    try:
        # data translate
        translater_obj = DataTranslater()
        translater_obj.read_config_file(config_path)
        translater_obj.get_translater()

        logger.info('### DATA FOR VIEW')
        translater_obj.create_data_for_view()
        # translater_obj.display_data()
        data_for_view = translater_obj.get_data_for_view()

        def _calc_proba_df(translater_obj):
            logger.info('### TRANSLATE DATA FOR VIEW')
            translater_obj.translate_data_for_view()
            # translater_obj.display_data()

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
            trainer_obj.calc_estimator_data()

            logger.info('### GET ESTIMATOR DATA')
            estimator_data = trainer_obj.get_estimator_data()

            # output
            outputer_obj = Outputer(
                **data_for_model, **estimator_data, **post_processers)
            outputer_obj.read_config_file(config_path)

            logger.info('### PREDICT')
            _, Y_pred_proba_df = outputer_obj.calc_predict_data()

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

        # output
        outputer_obj = Outputer(**{
            'feature_columns': [],
            'test_ids': [],
            'X_train': [],
            'Y_train': [],
            'X_test': [],
            'cv': [],
            'scorer': None,
            'classes': [],
            'single_estimators': [],
            'estimator': None,
        })
        outputer_obj.read_config_file(config_path)
        outputer_obj.Y_pred_proba_df = all_proba_df

        logger.info('### WRITE PREDICT DATA')
        outputer_obj.write_predict_data()

    except Exception as e:
        logger.error('%s' % e)
        traceback.print_exc()
    finally:
        notifier_obj = Notifier()
        notifier_obj.read_config_file(config_path)
        notifier_obj.notify_slack()
    logger.info('# FINISHED')
