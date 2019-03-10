import os
import pandas as pd
from logging import getLogger
from .ConfigReader import ConfigReader

logger = getLogger('predict').getChild('Integrater')


class Integrater(ConfigReader):
    def __init__(self):
        self.BASE_PATH = '%s/..' % os.path.dirname(os.path.abspath(__file__))
        self.OUTPUT_PATH = '%s/outputs' % self.BASE_PATH
        self.configs = {}

    def calc_average(self):
        filenames = self.configs['integrate']['filenames']
        if 'weights' in self.configs['integrate']:
            weights = self.configs['integrate']['weights']
        else:
            weights = [1] * len(self.filenames)

        self.output = pd.DataFrame()
        for i, filename in enumerate(filenames):
            df = pd.read_csv(filename)
            id_df = df[self.id_col]
            val_df = df.drop(self.id_col, axis=1)
            if len(self.output) == 0:
                self.output = val_df * weights[i]
            else:
                self.output += val_df * weights[i]
        self.output /= sum(weights)
        self.output[self.id_col] = id_df
        return self.output

    def write_output(self, filename=None):
        if not filename:
            filename = '%s.csv' % self.configs['integrate']['output']
        self.output.to_csv(
            '%s/%s' % (self.OUTPUT_PATH, filename), index=False)
