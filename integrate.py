import sys
import os
import traceback
import logging.config
from logging import getLogger
from modules.Integrater import Integrater
from modules.Notifier import Notifier

if __name__ == '__main__':
    logging.config.fileConfig(
        './configs/logging.conf', disable_existing_loggers=False)
    logger = getLogger('integrate')

    logger.info('# START')

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = './configs/config.json'
    try:
        integrater_obj = Integrater()
        integrater_obj.read_config_file(config_path)

        logger.info('### DISPLAY CORRELATIONS')
        integrater_obj.display_correlations()

        logger.info('### INTEGRATE')
        integrater_obj.integrate()

        logger.info('### WRITE OUTPUT')
        integrater_obj.write_output()
    except Exception as e:
        logger.error('%s' % e)
        traceback.print_exc()
    finally:
        notifier_obj = Notifier()
        notifier_obj.read_config_file(config_path)
        notifier_obj.notify_slack()
    logger.info('# FINISHED')
