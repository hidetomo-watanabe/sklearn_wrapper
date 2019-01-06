import sys
import os
import traceback
from modules.Integrater import Integrater
from modules.Notifier import Notifier
from modules.MyLogger import MyLogger

logger = MyLogger().get_logger('integrate')

if __name__ == '__main__':
    logger.info('# START')

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = './configs/config.json'
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
