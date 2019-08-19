import os
import pandas as pd
from itertools import combinations
from IPython.display import display
from logging import getLogger
from .ConfigReader import ConfigReader

logger = getLogger('predict').getChild('Integrater')


class Integrater(ConfigReader):
    def __init__(self):
        self.BASE_PATH = '%s/..' % os.path.dirname(os.path.abspath(__file__))
        self.OUTPUT_PATH = '%s/outputs' % self.BASE_PATH
        self.configs = {}

    def display_correlations(self):
        filenames = self.configs['integrate']['filenames']
        for filename1, filename2 in combinations(filenames, 2):
            logger.info(f'{filename1} vs {filename2}')
            df1 = pd.read_csv(filename1).drop(self.id_col, axis=1)
            df2 = pd.read_csv(filename2).drop(self.id_col, axis=1)
            display(df1.corrwith(df2))
        return

    def integrate(self):
        mode = self.configs['integrate']['mode']
        logger.info(f'mode: {mode}')
        if mode == 'average':
            return self._calc_average()
        else:
            logger.error('NOT IMPLEMENTED INTEGRATE MODE: %s' % mode)
            raise Exception('NOT IMPLEMENTED')

    def _calc_average(self):
        filenames = self.configs['integrate']['filenames']
        weights = self.configs['integrate'].get('weights')
        if not weights:
            weights = [1] * len(self.filenames)

        self.output = pd.DataFrame()
        for filename, weight in zip(filenames, weights):
            df = pd.read_csv(filename)
            id_df = df[self.id_col]
            val_df = df.drop(self.id_col, axis=1)
            if len(self.output) == 0:
                self.output = val_df * weight
            else:
                self.output += val_df * weight
        self.output /= sum(weights)
        self.output[self.id_col] = id_df
        return self.output

    def write_output(self, filename=None):
        if not filename:
            filename = '%s.csv' % self.configs['integrate']['output']
        self.output.to_csv(
            '%s/%s' % (self.OUTPUT_PATH, filename), index=False)
