import json
import requests
import configparser


class Notifier(object):
    def __init__(self):
        self.cp = configparser.SafeConfigParser()

    def read_config_file(self, path='./config.ini'):
        self.cp.read(path)

    def notify_slack(self):
        text = 'Finished Analysis.'
        requests.post(
            self.cp.get('notify', 'slack'),
            data=json.dumps({'text': text}))


if __name__ == '__main__':
    pass
