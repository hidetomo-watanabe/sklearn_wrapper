import os
import json
import pandas as pd
from logging import getLogger

BASE_PATH = '%s/..' % os.path.dirname(os.path.abspath(__file__))
logger = getLogger('predict').getChild('Integrater')


class Integrater(object):
    def __init__(self):
        self.configs = {}

    def read_config_file(self, path='%s/scripts/config.json' % BASE_PATH):
        with open(path, 'r') as f:
            self.configs = json.loads(f.read())
        self.filenames = self.configs['integrate']['filenames']

    def calc_average(self):
        dfs = []
        for filename in self.filenames:
            dfs.append(pd.read_csv(filename))
        self.output = sum(dfs) / len(dfs)
        return self.output

    def write_output(self, filename):
        self.output.to_csv(
            '%s/outputs/%s' % (BASE_PATH, filename), index=False)


if __name__ == '__main__':
    pass
