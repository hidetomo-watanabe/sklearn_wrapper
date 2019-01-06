import sys
import os
import traceback
BASE_PATH = '%s/..' % os.path.dirname(os.path.abspath(__file__))
sys.path.append('%s/modules' % BASE_PATH)
from Integrater import Integrater
from Notifier import Notifier
from MyLogger import MyLogger

logger = MyLogger().get_logger('integrate')

if __name__ == '__main__':
    logger.info('# START')

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = '%s/configs/config.json' % BASE_PATH
    try:
        integrater_obj = Integrater()
        integrater_obj.read_config_file(config_path)

        integrater_obj.calc_average()
        integrater_obj.write_output()
    except Exception as e:
        logger.error('%s' % e)
        traceback.print_exc()
    finally:
        notifier_obj = Notifier()
        notifier_obj.read_config_file(config_path)
        notifier_obj.notify_slack()
    logger.info('# FINISHED')
