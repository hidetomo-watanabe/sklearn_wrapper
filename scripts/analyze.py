import sys
import os
import traceback
BASE_PATH = '%s/..' % os.path.dirname(os.path.abspath(__file__))
sys.path.append('%s/modules' % BASE_PATH)
from SingleAnalyzer import SingleAnalyzer
from Notifier import Notifier

if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = '%s/scripts/config.json' % BASE_PATH
    try:
        sa_obj = SingleAnalyzer()
        sa_obj.read_config_file(config_path)

        print('### INIT')
        sa_obj.get_raw_data()
        sa_obj.display_data()
        print('')

        print('### TRANSLATE')
        sa_obj.trans_raw_data()
        sa_obj.display_data()
        print('')

        # print('### VISUALIZE TRAIN DATA')
        # sa_obj.visualize_train_data()
        # print('')

        print('### NORMALIZE')
        sa_obj.get_fitting_data()
        sa_obj.normalize_fitting_data()
        print('')

        print('### VALIDATE')
        sa_obj.is_ok_with_adversarial_validation()
        print('')

        print('### FIT')
        sa_obj.calc_best_model('tmp.pickle')

        # print('### VISUALIZE TRAIN PRED DATA')
        # sa_obj.visualize_train_pred_data()
        # print('')

        print('### OUTPUT')
        sa_obj.calc_output()
        sa_obj.write_output('tmp.csv')

    except Exception as e:
        print('[ERROR] %s' % e)
        traceback.print_exc()
    finally:
        notifier_obj = Notifier()
        notifier_obj.read_config_file(config_path)
        notifier_obj.notify_slack()
