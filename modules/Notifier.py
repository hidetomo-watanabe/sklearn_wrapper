import json
import requests


class Notifier(object):
    def __init__(self):
        self.configs = {}

    def read_config_file(self, path):
        with open(path, 'r') as f:
            self.configs = json.loads(f.read())

    def read_config_text(self, text):
        self.configs = json.loads(text)

    def notify_slack(self):
        text = 'Finished.'
        requests.post(
            self.configs['notify']['slack'],
            data=json.dumps({'text': text}))
