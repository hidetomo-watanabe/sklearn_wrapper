import os
import json
import requests
BASE_PATH = '%s/..' % os.path.dirname(os.path.abspath(__file__))


class Notifier(object):
    def __init__(self):
        self.configs = {}

    def read_config_file(self, path='%s/scripts/config.json' % BASE_PATH):
        with open(path, 'r') as f:
            self.configs = json.loads(f.read())

    def notify_slack(self):
        text = 'Finished Analysis.'
        requests.post(
            self.configs['notify']['slack'],
            data=json.dumps({'text': text}))


if __name__ == '__main__':
    pass
