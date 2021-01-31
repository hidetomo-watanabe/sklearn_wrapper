from logging import getLogger

from IPython.display import display

from dtreeviz.trees import dtreeviz

import eli5
from eli5.sklearn import PermutationImportance

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier


logger = getLogger('predict').getChild('Visualizer')
if 'ConfigReader' not in globals():
    from .ConfigReader import ConfigReader
if 'LikeWrapper' not in globals():
    from .commons.LikeWrapper import LikeWrapper


class Visualizer(ConfigReader, LikeWrapper):
    def __init__(self, sample_frac=1.0, with_xlog=False):
        self.configs = {}
        self.sample_frac = sample_frac
        self.with_xlog = with_xlog
        self.hist_params = {
            'alpha': 0.5,
            'density': True,
            'stacked': True,
        }

    def _show_plt(self, ax, plt):
        if self.with_xlog:
            plt.xscale('log')

        ax.legend(loc="best")
        plt.tick_params(colors='white')
        plt.show()

    def display_dfs(self, train_df, test_df, pred_df):
        train_df = self.sample_like(train_df, frac=self.sample_frac)
        test_df = self.sample_like(test_df, frac=self.sample_frac)
        pred_df = self.sample_like(pred_df, frac=self.sample_frac)

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

    def plot_scatter_matrix(self, target, feature_columns):
        target = self.sample_like(target, frac=self.sample_frac)

        pd.plotting.scatter_matrix(pd.DataFrame(
            target, columns=feature_columns), figsize=(10, 10))

    def plot_corrcoef(self, target, feature_columns):
        target = self.sample_like(target, frac=self.sample_frac)

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(
            np.corrcoef(target, rowvar=False),
            xticklabels=feature_columns,
            yticklabels=feature_columns,
            fmt="1.2f", annot=True, lw=0.7, cmap='YlGnBu')
        ax.set_ylim(len(feature_columns), 0)
        self._show_plt(ax, plt)

    def plot_ndarray_histograms(self, targets, labels, title=None):
        ax = plt.subplot()
        if title:
            ax.set_title(title)
        cmap = plt.get_cmap('tab10')
        for i, (target, label) in enumerate(zip(targets, labels)):
            target = self.sample_like(target, frac=self.sample_frac)
            ax.hist(
                target, **self.hist_params,
                color=cmap(i), label=f'{label}')
        self._show_plt(ax, plt)

    def plot_df_histograms(self, df, label):
        df = self.sample_like(df, frac=self.sample_frac)
        # regかつtrain/testでないときは、label統一
        uniq_labels = np.sort(np.unique(label))
        if self.configs['pre']['train_mode'] == 'reg':
            if len(uniq_labels) >= 3:
                label = np.array(['reg'] * len(label))
        label = self.sample_like(label, frac=self.sample_frac)
        uniq_labels = np.sort(np.unique(label))

        for key in df.keys():
            if key == self.id_col:
                continue

            _targets = []
            for i, l in enumerate(uniq_labels):
                _targets.append(
                    df[key][self.ravel_like(label) == l].dropna().to_numpy())
            self.plot_ndarray_histograms(_targets, uniq_labels, title=key)

    def plot_with_2_dimensions(self, target, label, target_ids):
        target = self.sample_like(target, frac=self.sample_frac)
        # regかつtrain/testでないときは、label統一
        uniq_labels = np.sort(np.unique(label))
        if self.configs['pre']['train_mode'] == 'reg':
            if len(uniq_labels) >= 3:
                label = np.array(['reg'] * len(label))
        label = self.sample_like(label, frac=self.sample_frac)
        uniq_labels = np.sort(np.unique(label))
        target_ids = self.sample_like(target_ids, frac=self.sample_frac)

        model_obj = TSNE(n_components=2, random_state=42)
        target = model_obj.fit_transform(target)

        cmap = plt.get_cmap('tab10')
        ax = plt.subplot()
        ax.set_title('target')
        ax.set_xlabel('tsne_0')
        ax.set_ylabel('tsne_1')

        for i, l in enumerate(uniq_labels):
            _target = target[np.where(self.ravel_like(label) == l)]
            ax.scatter(
                _target[:, 0], _target[:, 1],
                alpha=0.5, color=cmap(i), label=f'label: {l}')
        self._show_plt(ax, plt)

        return pd.DataFrame(
            np.concatenate((target_ids, target, label), axis=1),
            columns=[self.id_col, 'tsne_0', 'tsne_1', 'label'])

    def visualize_decision_tree(
        self, X_train, Y_train, feature_names, max_depth=3
    ):
        X_train = self.sample_like(X_train, frac=self.sample_frac)
        Y_train = self.sample_like(Y_train, frac=self.sample_frac)

        Y_train = self.ravel_like(Y_train)
        clf = DecisionTreeClassifier(max_depth=max_depth)
        clf.fit(X_train, Y_train)
        for pred_col in self.pred_cols:
            viz = dtreeviz(
                clf, X_train, Y_train, target_name=pred_col,
                feature_names=feature_names,
                class_names=list(set([str(i) for i in Y_train])))
            display(viz)

    def draw_images(self, target, label):
        target = self.sample_like(target, frac=self.sample_frac)
        label = self.sample_like(label, frac=self.sample_frac)

        fig = plt.figure(figsize=(len(target), 1), dpi=100)
        plt.subplots_adjust(wspace=1.0)
        for i, (_t, _l) in enumerate(zip(target, label)):
            ax = fig.add_subplot(1, len(target), i + 1)
            ax.set_title(f'label: {_l}')
            ax.imshow(_t)
        self._show_plt(ax, plt)

    def display_feature_importances(self, estimator, feature_columns):
        feature_importances = pd.DataFrame(
            data=[estimator.feature_importances_], columns=feature_columns)
        feature_importances = feature_importances.iloc[
            :, np.argsort(feature_importances.to_numpy()[0])[::-1]]
        display(feature_importances)
        display(feature_importances / np.sum(feature_importances.to_numpy()))

    def display_permutation_importances(
        self, estimator, X_train, Y_train, feature_columns
    ):
        perm = PermutationImportance(estimator, random_state=42).fit(
            self.toarray_like(X_train), Y_train)
        display(eli5.explain_weights_df(perm, feature_names=feature_columns))

    def plot_learning_curve(
        self, title, estimator, X_train, Y_train, scorer, cv
    ):
        train_sizes = np.linspace(.1, 1.0, 5)
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X_train, Y_train, scoring=scorer,
            cv=cv, train_sizes=train_sizes, n_jobs=-1)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        ylim = (0.7, 1.01)
        ax = plt.subplot()
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Score")
        ax.fill_between(
            train_sizes, train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std, alpha=0.1,
            color="r")
        ax.fill_between(
            train_sizes, test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std, alpha=0.1, color="g")
        ax.plot(
            train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
        ax.plot(
            train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")
        self._show_plt(ax, plt)

    def plot_roc(self, Y_train, Y_train_pred_proba):
        fpr, tpr, thresholds = metrics.roc_curve(
            Y_train, Y_train_pred_proba)
        auc = metrics.auc(fpr, tpr)

        ax = plt.subplot()
        ax.set_title('ROC curve')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.plot(fpr, tpr, label=f'AUC: {auc}')
        self._show_plt(ax, plt)
