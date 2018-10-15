import sys
import os
import traceback
BASE_PATH = '%s/..' % os.path.dirname(os.path.abspath(__file__))
sys.path.append('%s/modules' % BASE_PATH)
from SingleAnalyzer import SingleAnalyzer
from Notifier import Notifier
from MyLogger import MyLogger

logger = MyLogger().get_logger('analyze')

if __name__ == '__main__':
    logger.info('# START')

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = '%s/scripts/config.json' % BASE_PATH
    try:
        sa_obj = SingleAnalyzer()
        sa_obj.read_config_file(config_path)

        logger.info('### INIT')
        sa_obj.get_raw_data()
        sa_obj.display_data()

        logger.info('### TRANSLATE')
        sa_obj.trans_raw_data()
        sa_obj.display_data()

        # logger.info('### VISUALIZE TRAIN DATA')
        # sa_obj.visualize_train_data()

        logger.info('### NORMALIZE')
        sa_obj.get_fitting_data()
        sa_obj.normalize_fitting_data()

        logger.info('### VALIDATE')
        sa_obj.is_ok_with_adversarial_validation()

        logger.info('### FIT')
        sa_obj.calc_best_model('tmp.pickle')

        # logger.info('### VISUALIZE TRAIN PRED DATA')
        # sa_obj.visualize_train_pred_data()

        logger.info('### OUTPUT')
        sa_obj.calc_output()
        sa_obj.write_output('tmp.csv')

    except Exception as e:
        logger.error('%s' % e)
        traceback.print_exc()
    finally:
        notifier_obj = Notifier()
        notifier_obj.read_config_file(config_path)
        notifier_obj.notify_slack()
    logger.info('# FINISHED')
