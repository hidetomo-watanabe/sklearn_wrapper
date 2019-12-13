import json
import requests
from logging import getLogger

logger = getLogger('predict').getChild('Notifier')


class Notifier(object):
    def __init__(self):
        self.configs = {}

    def read_config_file(self, path):
        with open(path, 'r') as f:
            self.configs = json.loads(f.read())

    def read_config_text(self, text):
        self.configs = json.loads(text)

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
