import os
import json
import pandas as pd
from logging import getLogger

BASE_PATH = '%s/..' % os.path.dirname(os.path.abspath(__file__))
logger = getLogger('predict').getChild('Integrater')


class Integrater(object):
    def __init__(self):
        self.configs = {}

    def read_config_file(self, path='%s/configs/config.json' % BASE_PATH):
        with open(path, 'r') as f:
            self.configs = json.loads(f.read())
        self.filenames = self.configs['integrate']['filenames']
        if 'weights' in self.configs['integrate']:
            self.weights = self.configs['integrate']['weights']
        else:
            self.weights = [1] * len(self.filenames)

    def calc_average(self):
        self.output = pd.DataFrame()
        for i, filename in enumerate(self.filenames):
            if len(self.output) == 0:
                self.output = pd.read_csv(filename) * self.weights[i]
            else:
                self.output += pd.read_csv(filename) * self.weights[i]
        self.output /= sum(self.weights)
        return self.output

    def write_output(self, filename=None):
        if not filename:
            filename = '%s.csv' % self.configs['integrate']['output']
        self.output.to_csv(
            '%s/outputs/%s' % (BASE_PATH, filename), index=False)


if __name__ == '__main__':
    pass
