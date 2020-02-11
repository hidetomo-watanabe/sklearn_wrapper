import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from dtreeviz.trees import dtreeviz
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import learning_curve
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from logging import getLogger


logger = getLogger('predict').getChild('Visualizer')
if 'ConfigReader' not in globals():
    from .ConfigReader import ConfigReader
if 'CommonMethodWrapper' not in globals():
    from .CommonMethodWrapper import CommonMethodWrapper


class Visualizer(ConfigReader, CommonMethodWrapper):
    def __init__(self):
        self.configs = {}
        self.hist_params = {
            'alpha': 0.5,
            'density': True,
            'stacked': True,
        }

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

    def plot_x_train_histogram_with_pred_group(self, train_df, pred_df):
        for key in train_df.keys():
            if key == self.id_col:
                continue
            ax = plt.subplot()
            ax.set_title(key)
            if self.configs['fit']['train_mode'] == 'clf':
                for pred_col in self.pred_cols:
                    cmap = plt.get_cmap('tab10')
                    for i, pred_val in enumerate(
                        np.unique(pred_df[pred_col].to_numpy())
                    ):
                        ax.hist(
                            train_df[
                                pred_df[pred_col] == pred_val
                            ][key].dropna(), **self.hist_params,
                            color=cmap(i), label=f'{pred_col}: {pred_val}')
            elif self.configs['fit']['train_mode'] == 'reg':
                ax.hist(
                    train_df[key].dropna(), **self.hist_params)
            ax.legend()
            plt.show()

    def plot_x_train_test_histogram(self, train_df, test_df):
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
                    df[key].dropna(), **self.hist_params,
                    color=cmap(i), label=f'{label}')
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
        Y_train = self.ravel_like(Y_train)
        clf = DecisionTreeClassifier(max_depth=max_depth)
        clf.fit(X_train, Y_train)
        for pred_col in self.pred_cols:
            viz = dtreeviz(
                clf, X_train, Y_train, target_name=pred_col,
                feature_names=feature_names,
                class_names=list(set([str(i) for i in Y_train])))
            display(viz)

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
        ylim = (0.7, 1.01)
        train_sizes = np.linspace(.1, 1.0, 5)

        plt.figure()
        plt.title(title)
        plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X_train, Y_train, scoring=scorer,
            cv=cv, train_sizes=train_sizes, n_jobs=-1)
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

    def plot_roc(self, Y_train, Y_train_pred_proba):
        fpr, tpr, thresholds = metrics.roc_curve(
            Y_train, Y_train_pred_proba)
        auc = metrics.auc(fpr, tpr)
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot(fpr, tpr, label=f'AUC: {auc}')
        plt.legend(loc="best")

    def plot_y_targets_histogram(self, Y_targets, labels):
        ax = plt.subplot()
        ax.set_title('Y_targets')
        cmap = plt.get_cmap('tab10')
        for i, (Y_target, label) in enumerate(zip(Y_targets, labels)):
            ax.hist(
                Y_target, **self.hist_params,
                color=cmap(i), label=f'{label}')
        ax.legend()
        plt.show()
