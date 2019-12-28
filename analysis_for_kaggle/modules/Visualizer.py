import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from dtreeviz.trees import dtreeviz
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from logging import getLogger


logger = getLogger('predict').getChild('Visualizer')
try:
    from .ConfigReader import ConfigReader
except ImportError:
    logger.warning('IN FOR KERNEL SCRIPT, ConfigReader import IS SKIPPED')


class Visualizer(ConfigReader):
    def __init__(self):
        self.configs = {}

    def display_data(self, train_df, test_df, pred_df):
        if self.configs['pre']['train_mode'] == 'clf':
            logger.info('train pred counts')
            for pred_col in self.pred_cols:
                logger.info('%s:' % pred_col)
                display(pred_df[pred_col].value_counts())
                display(pred_df[pred_col].value_counts(normalize=True))
        elif self.configs['pre']['train_mode'] == 'reg':
            logger.info('train pred mean, std')
            for pred_col in self.pred_cols:
                logger.info('%s:' % pred_col)
                display('mean: %f' % pred_df[pred_col].mean())
                display('std: %f' % pred_df[pred_col].std())
        else:
            logger.error('TRAIN MODE SHOULD BE clf OR reg')
            raise Exception('NOT IMPLEMENTED')
        for label, df in [('train', train_df), ('test', test_df)]:
            logger.info('%s:' % label)
            display(df.head())
            display(df.describe(include='all'))

    def plot_train_pred_histogram(self, train_df, pred_df):
        for key in train_df.keys():
            if key == self.id_col:
                continue
            ax = plt.subplot()
            ax.set_title(key)
            if self.configs['fit']['train_mode'] == 'clf':
                cmap = plt.get_cmap('tab10')
                for pred_col in self.pred_cols:
                    for i, pred_val in enumerate(
                        np.unique(pred_df[pred_col].to_numpy())
                    ):
                        ax.hist(
                            train_df[
                                pred_df[pred_col] == pred_val
                            ][key].dropna(),
                            bins=20, color=cmap(i), alpha=0.5,
                            label='%s: %d' % (pred_col, pred_val))
            elif self.configs['fit']['train_mode'] == 'reg':
                ax.hist(train_df[key].dropna(), bins=20, alpha=0.5)
            ax.legend()
            plt.show()

    def plot_train_test_histogram(self, train_df, test_df):
        for key in train_df.keys():
            if key == self.id_col:
                continue
            ax = plt.subplot()
            ax.set_title(key)
            cmap = plt.get_cmap('tab10')
            for i, (label, df) in enumerate(
                [('train', train_df), ('test', test_df)]
            ):
                ax.hist(
                    df[key].dropna(),
                    bins=20, color=cmap(i), alpha=0.5,
                    label='%s' % (label))
            ax.legend()
            plt.show()

    def plot_train_scatter_matrix(self, X_train, Y_train, feature_columns):
        pd.plotting.scatter_matrix(
            pd.DataFrame(
                np.concatenate([X_train, Y_train], 1),
                columns=(feature_columns + self.pred_cols)),
            figsize=(10, 10))

    def plot_test_scatter_matrix(self, X_test, feature_columns):
        pd.plotting.scatter_matrix(
            pd.DataFrame(
                X_test,
                columns=feature_columns),
            figsize=(10, 10))

    def plot_train_corrcoef(self, X_train, Y_train, feature_columns):
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(
            np.corrcoef(X_train, Y_train, rowvar=False),
            xticklabels=feature_columns + self.pred_cols,
            yticklabels=feature_columns + self.pred_cols,
            fmt="1.2f", annot=True, lw=0.7, cmap='YlGnBu')
        ax.set_ylim(len(feature_columns + self.pred_cols), 0)

    def plot_test_corrcoef(self, X_test, feature_columns):
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(
            np.corrcoef(X_test, rowvar=False),
            xticklabels=feature_columns,
            yticklabels=feature_columns,
            fmt="1.2f", annot=True, lw=0.7, cmap='YlGnBu')
        ax.set_ylim(len(feature_columns), 0)

    def visualize_decision_tree(
        self, X_train, Y_train, feature_names, max_depth=3
    ):
        Y_train = Y_train.ravel()
        clf = DecisionTreeClassifier(max_depth=max_depth)
        clf.fit(X_train, Y_train)
        for pred_col in self.pred_cols:
            viz = dtreeviz(
                clf, X_train, Y_train, target_name=pred_col,
                feature_names=feature_names,
                class_names=list(set([str(i) for i in Y_train])))
            display(viz)

    def plot_learning_curve(
        self, title, estimator, X_train, Y_train, scorer, cv, n_jobs=-1
    ):
        ylim = (0.7, 1.01)
        train_sizes = np.linspace(.1, 1.0, 5)

        plt.figure()
        plt.title(title)
        plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X_train, Y_train, scoring=scorer,
            cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(
            train_sizes, train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std, alpha=0.1,
            color="r")
        plt.fill_between(
            train_sizes, test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(
            train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
        plt.plot(
            train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")

        plt.legend(loc="best")

    def plot_confusion_matrix(self, Y_train, Y_train_pred):
        g = sns.jointplot(Y_train_pred, Y_train, kind='kde')
        g.set_axis_labels('Y_train_pred', 'Y_train')
        g.fig.suptitle('estimator')
