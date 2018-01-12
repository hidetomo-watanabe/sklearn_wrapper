import sys
from Analyzer import Analyzer
from Notifier import Notifier

if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = './config.json'
    try:
        analyzer_obj = Analyzer()
        analyzer_obj.read_config_file(config_path)

        analyzer_obj.get_raw_data()
        print('### INIT OVERVIEW')
        analyzer_obj.display_data()
        analyzer_obj.trans_raw_data()

        print('### TRANSLATION OVERVIEW')
        analyzer_obj.display_data()
        # analyzer_obj.visualize()

        analyzer_obj.get_fitting_data()
        analyzer_obj.normalize_fitting_data()
        analyzer_obj.is_ok_with_adversarial_validation()

        analyzer_obj.calc_best_model('tmp.pickle')
        analyzer_obj.calc_output('tmp.csv')
    except Exception as e:
        print('[ERROR] %s' % e)
    finally:
        notifier_obj = Notifier()
        notifier_obj.read_config_file(config_path)
        notifier_obj.notify_slack()
