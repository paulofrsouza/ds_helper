#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix, \
 roc_auc_score, silhouette_score, roc_curve
from sklearn.model_selection import ShuffleSplit, GridSearchCV

import matplotlib.pyplot as plt

import time


def classification_results(y_test, pred_clf, export=False, auc=False):
    """
    Classification Model Results Report

    Returns the common classification performance reports Classification Report
    and Confusion Matrix. If it is a binary classification, it can be set to
    also return the AUC score. The user can also choose to return the results
    as pandas DataFrames.

    Parameters
    ----------
    y_test : pandas Series or numpy 1-D array
        Ground truth values to be compared to.
    pred_clf : pandas Series or numpy 1-D array
        Values predicted by the classification algorithm.
    export : bool, optional
        Wheter to export the Classification Report and Confusion Matrix as
        pandas DataFrames. Default value is False.
    auc : bool, optional
        Wheter to calculate the ROC-AUC score for the classification results.
        USE ONLY FOR BINARY CLASSIFICATIONS. Default is False.

    Returns
    -------
    comf_m_clf : pandas DataFrame
        Exported Confusion Matrix
    class_r_clf : pandas DataFrame
        Exported Classification Report
    """
    print('\nConfusion Matrix')
    print(confusion_matrix(y_test, pred_clf))
    print('\nClassification Report')
    print(classification_report(y_test, pred_clf))
    if auc:
        print('\nAUC score : %0.3f' % roc_auc_score(y_test, pred_clf))

    if export:
        conf_m_clf = pd.DataFrame(confusion_matrix(y_test, pred_clf))
        conf_m_clf.index = [1, 0]
        conf_m_clf.columns = [1, 0]
        class_r_clf = pd.DataFrame(
            classification_report(y_test, pred_clf, output_dict=True)).T
        return conf_m_clf, class_r_clf


def plot_vote_roc_curve(y_test, y_pred):
    # plot the ROC-AUC curve for voting classifiers
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.figure(figsize=(12, 8))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test,
                                                                     y_pred))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def grid_search_cv_supervised(x_train, y_train, model, param_grid, n_splits=5,
                              test_size=.1, train_size=.8, scoring='accuracy',
                              random_state=42):
    """
    Supervised Grid Search-CV

    Performs a Cross-Validated Grid Search for supervised sklearn models.
    Returns a given model tuned with its best parameters. Also generates some
    reporting information, such as tuning time, best parameters and best score.

    Parameters
    ----------
    x_train: pandas.DataFrame or numpy ndarray
        Dataset containing independent variables for training. All values must
        be numeric and already cleaned.
    y_train : pandas.Series or numpy 1-D array
        Dataset containing the dependent variable for training. All values must
        be numeric and already cleaned.
    model : sklearn model object
        Sklearn model chosen to be tuned.
    param_grid : sklearn.model_selection._search.ParameterGrid
        ParameterGrid object, representing the possibile combinations over a
        dictionary of lists containing candidate values for the model's
        parameters.
    n_splits : int
        Number of splits to be performed by the Cross-Validation. Default value
        is 5.
    test_size : float
        Proportion of the training data to be used for validation. Default
        value is 0.1.
    train_size : float
        Proportion of the traiing data to be used for training. Default value
        is 0.8. The default values of train_size and test_size leave 10% of the
        training data out, aiming to reduce overfitting.
    scoring : str
        Name of the metric chosen to evaluate the performance of each set of
        parameters. Must be one of the accepeted metrics by sklearn to evaluate
        the given model. The default value is 'accuracy'.
    random_state : int
        Random state seed, used for replicability of results.

    Returns
    -------
    sklearn model object
        Given sklearn supervised model, set with the best parameters found.
    """
    cv_split = ShuffleSplit(n_splits=n_splits, test_size=test_size,
                            train_size=train_size, random_state=random_state)

    # GridSearchCV declaration and starting processing timer
    start = time.perf_counter()
    best_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=cv_split, scoring=scoring, n_jobs=-1)
    best_search.fit(x_train, y_train)
    run = time.perf_counter() - start

    best_param = best_search.best_params_
    print('The best parameters for {} are {} with runtime of \
          {:.2f} seconds and score {}.'
          .format(model.__class__.__name__, best_param, run,
                  best_search.best_score_))
    model.set_params(**best_param)

    return model


def grid_search_cv_unsupervised(model, df, param_grid, n_splits=5,
                                random_state=42):
    """
    Unsupervised Grid Search-CV

    Performs a Cross-Validated Grid Search for unsupervised sklearn models.
    Returns a given model tuned with its best parameters. Also generates some
    reporting information, such as tuning time, best parameters and best score.
    The default evaluation metric is sklearn.metrics.silhouette_score, given
    its standartized output values.

    Parameters
    ----------
    model : sklearn model object
        Sklearn unsupervised model chosen to be tuned.
    df : pandas.DataFrame or numpy.ndarray
        Dataset containing the data to be analyzed. The data must be all
        numeric and already treated.
    param_grid : sklearn.model_selection._search.ParameterGrid
        ParameterGrid object, representing the possibile combinations over a
        dictionary of lists containing candidate values for the model's
        parameters.
    n_splits : int
        Number of splits to be performed by the Cross-Validation. Default value
        is 5.
    random_state : int
        Random state seed, used for replicability of results.

    Returns
    -------
    sklearn model object
        Given sklearn unsupervised model, set with the best parameters found.
    """
    # Start processing timer, declaration of n-fold cross-validation and
    # performing Grid Search
    start = time.perf_counter()
    res = {}
    frac = round(1/n_splits, 2)
    for _ in range(n_splits):
        df_param = df.sample(frac=frac,
                             random_state=np.random.randint(low=0, high=100))
        for i, param in enumerate(list(param_grid)):
            mdl = model.set_params(**param).fit_predict(df_param)
            avg_score = silhouette_score(df_param, mdl,
                                         sample_size=int(df_param.shape[0]/2),
                                         random_state=random_state)
            if i in res:
                res[i].append(avg_score)
            else:
                res[i] = []

    run = time.perf_counter() - start
    print('{} minutes processing.'.format(run/60))

    # Applying Central Limit Theorem to obtain the population's Silhouette
    # Score. Choice and settting of best set of parameters.
    for key in res:
        res[key] = np.mean(res[key])
    rank = pd.Series(res)
    best_param = param_grid[rank.idxmax()]
    print('The best parameters for {} are {} with runtime of \
          {:.2f} seconds and score {}.'
          .format(model.__class__.__name__, best_param, run,
                  rank.max()))
    return model.set_params(**best_param)
