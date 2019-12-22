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

    def visualize_train_df_histogram(self, train_df, pred_df):
        for key in train_df.keys():
            if key == self.id_col:
                continue
            ax = plt.subplot()
            ax.set_title(key)
            if self.configs['fit']['train_mode'] == 'clf':
                cmap = plt.get_cmap("tab10")
                for pred_col in self.pred_cols:
                    logger.info('%s:' % pred_col)
                    for i, pred_val in enumerate(
                        np.unique(pred_df[pred_col].to_numpy())
                    ):
                        ax.hist(
                            train_df[pred_df[pred_col] == pred_val][key],
                            bins=20, color=cmap(i), alpha=0.5,
                            label='%s: %d' % (pred_col, pred_val))
            elif self.configs['fit']['train_mode'] == 'reg':
                ax.hist(train_df[key], bins=20, alpha=0.5)
            ax.legend()
            plt.show()

    def visualize_train_df_heatmap(self, train_df, pred_df):
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            pd.merge(
                train_df.drop(self.id_col, axis=1),
                pred_df, left_index=True, right_index=True
            ).corr(),
            fmt="1.2f", annot=True, lw=0.7, cmap='YlGnBu')

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

    def visualize_learning_curve(
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
        return plt

    def visualize_y_train_pred_data(self, Y_train, Y_train_pred):
        g = sns.jointplot(Y_train, Y_train_pred, kind='kde')
        g.set_axis_labels('Y_train', 'Y_train_pred')
        g.fig.suptitle('estimator')
