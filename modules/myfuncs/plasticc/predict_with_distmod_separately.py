import sys
import os
import traceback
import logging.config
from logging import getLogger
import pandas as pd
BASE_PATH = '%s/../..' % os.path.dirname(os.path.abspath(__file__))
sys.path.append('%s/modules' % BASE_PATH)
from Predicter import Predicter
from Notifier import Notifier

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
        predicter_obj = Predicter()
        predicter_obj.read_config_file(config_path)

        logger.info('### INIT')
        predicter_obj.get_data_for_view()
        predicter_obj.display_data()

        def _calc_proba(predicter_obj):
            logger.info('### TRANSLATE')
            predicter_obj.trans_data_for_view()
            logger.info('##### NORMALIZE')
            predicter_obj.get_fitting_data()
            predicter_obj.normalize_fitting_data()
            predicter_obj.reduce_dimension()
            predicter_obj.display_data()

            # logger.info('### VISUALIZE TRAIN DATA')
            # predicter_obj.visualize_train_data()

            logger.info('### VALIDATE')
            predicter_obj.is_ok_with_adversarial_validation()

            logger.info('### FIT')
            predicter_obj.calc_ensemble_model()

            # logger.info('### VISUALIZE TRAIN PRED DATA')
            # predicter_obj.visualize_train_pred_data()

            logger.info('### OUTPUT')
            _, Y_pred_proba = predicter_obj.calc_output()
            # predicter_obj.write_output()

            return Y_pred_proba

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
        train_df = predicter_obj.train_df
        test_df = predicter_obj.test_df

        logger.info('##### ISNULL DISTMOD')
        predicter_obj.train_df = train_df[train_df['distmod'].isnull()]
        predicter_obj.test_df = test_df[test_df['distmod'].isnull()]
        isnull_proba = _calc_proba(predicter_obj)

        logger.info('##### NOTNULL DISTMOD')
        predicter_obj.train_df = train_df[train_df['distmod'].notnull()]
        predicter_obj.test_df = test_df[test_df['distmod'].notnull()]
        notnull_proba = _calc_proba(predicter_obj)

        all_proba = pd.concat(
            [_fill_proba(isnull_proba), _fill_proba(notnull_proba)],
            ignore_index=True)
        predicter_obj.Y_pred_proba = all_proba
        predicter_obj.write_output()

    except Exception as e:
        logger.error('%s' % e)
        traceback.print_exc()
    finally:
        notifier_obj = Notifier()
        notifier_obj.read_config_file(config_path)
        notifier_obj.notify_slack()
    logger.info('# FINISHED')
