import json
from logging import getLogger

import requests

logger = getLogger('predict').getChild('Notifier')
if 'ConfigReader' not in globals():
    from .ConfigReader import ConfigReader


class Notifier(ConfigReader):
    def __init__(self):
        self.configs = {}

    def notify_slack(self):
        mode = self.configs['notify'].get('mode')
        if not mode:
            logger.warning('NO NOTIFICATION')
            return

        logger.info('notification: %s' % mode)
        if mode == 'slack':
            text = 'Finished.'
            requests.post(
                self.configs['notify'][mode],
                data=json.dumps({'text': text}))
        else:
            logger.error('NOT IMPLEMENTED NOTIFICATION: %s' % mode)
            raise Exception('NOT IMPLEMENTED')
