import os
import numpy as np
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
        elif mode == 'vote':
            return self._calc_vote()
        else:
            logger.error('NOT IMPLEMENTED INTEGRATE MODE: %s' % mode)
            raise Exception('NOT IMPLEMENTED')

    def _get_id_df(self, dfs):
        id_dfs = []
        for df in dfs:
            id_dfs.append(df[self.id_col])
        id_df = id_dfs[0]
        for id_df_tmp in id_dfs:
            if not id_df.equals(id_df_tmp):
                logger.error('NOT SAME IDS OVER ALL FILES')
                raise Exception('DATA INCOSISTENCY')
        return id_df

    def _get_val_columns(self, dfs):
        cols_list = []
        for df in dfs:
            cols_list.append(df.drop(self.id_col, axis=1).columns)
        cols = cols_list[0]
        for cols_tmp in cols_list:
            if set(cols) != set(cols_tmp):
                logger.error('NOT SAME COLUMNS OVER ALL FILES')
                raise Exception('DATA INCOSISTENCY')
        return cols

    def _calc_vote(self):
        filenames = self.configs['integrate']['filenames']
        weights = self.configs['integrate'].get('weights')
        if not weights:
            weights = [1] * len(self.filenames)
        dfs = [pd.read_csv(filename) for filename in filenames]

        def _calc_results(val_column):
            vals = []
            for df, weight in zip(dfs, weights):
                val_df = df[val_column]
                vals.append(val_df.values.flatten())
            vals = np.array(vals)
            labels = list(set(vals.flatten()))

            vote_table = np.array([
                np.dot(
                    np.array(weights).T,
                    (np.vstack(vals) == i).astype(int)) for i in labels
            ])
            results = []
            for i in np.argmax(vote_table, axis=0):
                results.append(labels[i])
            return results

        self.output = pd.DataFrame()
        self.output[self.id_col] = self._get_id_df(dfs)
        for val_column in self._get_val_columns(dfs):
            self.output[val_column] = _calc_results(val_column)
        return self.output

    def _calc_average(self):
        filenames = self.configs['integrate']['filenames']
        weights = self.configs['integrate'].get('weights')
        if not weights:
            weights = [1] * len(self.filenames)
        dfs = [pd.read_csv(filename) for filename in filenames]

        self.output = pd.DataFrame()
        for df, weight in zip(dfs, weights):
            val_df = df.drop(self.id_col, axis=1)
            if len(self.output) == 0:
                self.output = val_df * weight
            else:
                self.output += val_df * weight
        self.output /= sum(weights)
        self.output[self.id_col] = self._get_id_df(dfs)
        return self.output

    def write_output(self, filename=None):
        if not filename:
            filename = '%s.csv' % self.configs['integrate']['output']
        self.output.to_csv(
            '%s/%s' % (self.OUTPUT_PATH, filename), index=False)
