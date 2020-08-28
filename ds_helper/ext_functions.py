#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from category_encoders import BinaryEncoder

from sklearn.metrics import classification_report, confusion_matrix, \
 roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

sns.set()
warnings.filterwarnings('ignore')


def desc_analysis(df, fig_h=6, fig_w=25):
    """
    Descriptive Analysis Report generator.

    Returns general descriptive statistics such as feature mean, std, row-wise
    Coefficient of Variation as well as scatterplots and histograms of each
    feature.

    Parameters
    ----------
    df : pandas DataFrame
        `df` must contain only numerical variables.
    fig_h : int, optional
        The height of the generated images. The default value is 6.
    fig_w : int, optional
        The width of the generated images. The default value is 25.
    """
    print('-' * 30)
    print("DATASET'S DESCRIPTIVE STATISTICS: \n\n{}\n\n".format(
        df.describe(include='all')))

    cv_row = round(100 * (df.sum(axis=1).std() / df.sum(axis=1).mean()), 3)
    cv_col = round(100 * (df.sum(axis=0).std() / df.sum(axis=0).mean()), 3)
    print('-' * 30)
    print('COEFFICIENT OF VARIATION (CV%)\n')
    print("ROW-WISE: {}%".format(cv_row))
    print("COLUMN-WISE: {}%\n\n".format(cv_col))

    print('-' * 30)
    print("DATASET'S PLOTS:\n")
    for el in df.columns.values:
        print('Feature: {} \nMinimum Value: {} | Maximum Value: {}'.format(
            el, df[el].min(), df[el].max()))
        print('Mean: {} | Standard Deviation: {}'.format(
            round(df[el].mean(), 3), round(df[el].std(), 3)))
        print('Skeewness: {} | Median: {}'.format(round(df[el].skew(), 3),
                                                  round(df[el].median(), 3)))

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

        fig.set_figheight(fig_h)
        fig.set_figwidth(fig_w)

        ax1.plot(df[el].reset_index(drop=True), '.')
        ax1.set_xlabel('index')
        ax1.set_ylabel('sample')
        ax1.set_title('{} Countplot'.format(el))

        ax2.hist(df[el], bins=50)
        ax2.set_xlabel('value')
        ax2.set_ylabel('count')
        ax2.set_title('{} Histogram'.format(el))

        plt.show()
        return


def correlation_heatmap(df):
    """
    Correlation Heatmap generator

    Returns a correlogram of the present features. Not recommended to use if
    there are too many variables to be analysed, given image space.

    Parameters
    ----------
    df : pandas DataFrame
        `df` must contain only numerical variables.
    """
    _, ax = plt.subplots(figsize=(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)

    _ = sns.heatmap(df.corr().round(3),
                    cmap=colormap,
                    square=True,
                    cbar_kws={'shrink': .9},
                    ax=ax,
                    annot=True,
                    linewidths=0.1,
                    vmax=1.0,
                    linecolor='white',
                    annot_kws={'fontsize': 12})

    plt.title("Feature's Pearson Correlation", y=1.05, size=15)
    return


def drop_quasi_zero(df, thresh=0.05):
    """
    Drop Quasi Zero Features

    Returns a passed pandas DataFrame without columns containing too few
    non-zero values.

    Parameters
    ----------
    df : pandas DataFrame
        Dataset whose columns will be dropped.
    thresh : float, optional
        Minimum percentage of non-zero values in any given column for it to be
        kept in the dataset. Default value is 0.05.

    Returns
    -------
    pandas DataFrame
    """
    drop_list = []
    for el in df.columns.values:
        non_zero = df[el][df[el] != 0].shape[0] / df.shape[0]
        if non_zero < thresh:
            drop_list.append(el)
            print('Dropping column: {} | Non-zero values ratio: {}%'.format(
                el, round(100 * non_zero, 3)))
    return df.drop(drop_list, axis=1)


def log_trans(vec):
    """
    Regularized Logarithmic Transformation

    Returns a logarithmic transformation that deals elegantly with very small
    positive values.

    Parameters
    ----------
    vec : pandas Series or numpy 1-D array
        List containing the series o values to be transformed. The input must
        be all positive.

    Returns
    -------
    pandas Series or numpy 1-D array
    """
    m = vec[vec != 0]
    c = int(np.log(min(m)))
    d = 10 ^ c
    return np.log(vec + d) - c


def drop_outliers_z_score(df, z=3):
    """
    Drop Outliers by Z-Score

    Deletes observations classified as outliers. The classification method
    analyzes if the observation is out of the established z-score bounds.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame to be cleaned. `df` must contain only numerical values.
    z : int, optional
        The z-score used as threshold for deletion. Default value is 3.

    Returns
    -------
    pandas DataFrame
    """
    n_initial_rows = df.shape[0]
    drop_list = set()

    print('-' * 25)
    print('OUTLIERS DELETION: Z-SCORE METHOD\n')

    for el in df.columns.values:
        drop_list = drop_list | \
            set(df[el][np.abs(df[el]-df[el].mean()) >=
                       (z*df[el].std())].index.values)

    drop_list = list(set(drop_list))
    count = len(drop_list)
    df.drop(drop_list, inplace=True)

    print('N of deleted rows: {} | % of deleted rows: {}% | '
          'z-score: {}'.format(count, round(100 * (count / n_initial_rows), 3),
                               z))
    return df


def drop_outliers_quantile(df, upper=0.99, lower=0):
    """
    Drop Outliers by Quantiles

    Deletes observations classified as outliers. The classification method
    analyzes if the observation is out of the established quantile bounds.

    Parameters
    ----------

    df : pandas DataFrame
        DataFrame to be cleaned. `df` must contain only nummerical values.
    upper : float
        Upper quantile boundary. Float value between 0 and 1. Must be bigger
        than `lower`. Default value is 0.99.
    lower : float
        Lower quantile boundary. Float value between 0 and 1. Must be smaller
        than `upper`. Default value is 0.

    Returns
    -------
    pandas DataFrame
    """
    n_initial_rows = df.shape[0]
    drop_list = set()
    quant_upper = df.quantile(upper)
    quant_lower = df.quantile(lower)

    print('-' * 25)
    print('OUTLIERS DELETION: QUANTILE METHOD\n')

    for el in df.columns.values:
        drop_list = drop_list | \
                    set(df[el][df[el] > quant_upper[el]].index.values) | \
                    set(df[el][df[el] < quant_lower[el]].index.values)

    drop_list = list(set(drop_list))
    count = len(drop_list)
    df.drop(drop_list, inplace=True)

    print('Lower quantile: {} | Upper quantile: {}'.format(lower, upper))
    print('N of deleted rows: {} | % of deleted rows: {}%'.format(
        count, round(100 * (count / n_initial_rows), 3)))
    return df


def check_nan(df, show_plot=False):
    """
    Count Nan's Per Feature

    Returns a pandas DataFrame containing the absolute and percentual number
    of missing values per column. It can also show this information in a plot
    if show_plot=True.

    Parameters
    ----------
    df : pandas DataFrame
        Dataset to be analysed.
    show_plot : bool, optional
        Flag indicating if the function should also return a barplot with the
        number os missing values in each column. Default value is False.

    Returns
    -------
    pandas DataFrame
    """
    from numpy import sum
    void = pd.DataFrame(sum(df.isna()), columns=['absolute'])
    void['percentage'] = round((void.absolute / df.shape[0]) * 100, 2)

    if show_plot:
        print('\n\n')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        plt.plot(void.index.values, void.percentage.values, 'ro')
        plt.xlabel('Columns indexes')
        plt.ylabel('Number of missing values')
        plt.title('Percentage of missing values per feature')
    return void


def drop_nan(df, perc=20):
    """
    Drop Quasi Null Features

    Returns a passed pandas DataFrame without columns containing too many
    Null values.

    Parameters
    ----------
    df : pandas DataFrame
        Dataset whose columns will be dropped.
    perc : int, optional
        Maximum percentage of Null values in any given column for it to be
        dropped from the dataset. Default value is 20.

    Returns
    -------
    pandas DataFrame
    """
    from ext_functions import check_nan
    check = check_nan(df, show_plot=False)
    return df.drop(check[check.percentage > perc].index.values, axis=1)


def pickle_load(path):
    """
    Load Pickle

    Wrapper for loading pickle files.

    Parameters
    ----------
    path : str
        String containing the path to the Pickle file to be loaded.
    """
    from pickle import load
    with open(path, 'rb') as f:
        return load(f)


def pickle_dump(obj, path):
    """
    Dump Pickle

    Wrapper for serializing Python objects as Pickle files.

    Parameters
    ----------
    obj : Python object
        Python object to be serialized.
    path : str
        String containing the path to the Pickle file to be created.
    """
    from pickle import dump
    with open(path, 'wb') as f:
        dump(obj, f)


def downsampling(x_train, y_train):
    """
    Perform Automated Downsampling

    Returns a set of balanced x_train and y_train datasets for classification.
    As of now, it only works for binary classifications.

    Parameters
    ----------
    x_train : pandas DataFrame or numpy ndarray
        Dataset containing the independent variables, already splitted from the
        test dataset.
    y_train : pandas Series or numpy 1-D array
        Dataset containing the dependent variable, already splitted from the
        test dataset.

    Returns
    -------
    pandas DataFrame or numpy ndarray
        Balanced independent variables for training.
    pandas Series or numpy 1-D array
        Balanced dependent variable for training.
    """
    from sklearn.utils import resample

    sampling = pd.concat([x_train, y_train], axis=1)
    big = sampling[y_train[0] == y_train[0].value_counts().index[0]]
    small = sampling[y_train[0] == y_train[0].value_counts().index[1]]

    downsampled = resample(big,
                           replace=False,
                           n_samples=len(small),
                           random_state=42)
    downsampled = pd.concat([downsampled, small])
    x_train_bal = downsampled[downsampled.columns.values[:-1]]
    y_train_bal = downsampled[downsampled.columns.values[-1]]

    del sampling, big, small, downsampled
    return x_train_bal, y_train_bal


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


def setup_pandas(max_rows=10, max_columns=10, width=1000):
    """
    Pandas Setup Wrapper

    Wrapper for quick loading and configuring of Python package Pandas.

    Parameters
    ----------
    max_rows : int, optional
        Maximum number of rows to be displayed when calling a pandas object.
    max_columns : int, optional
        Maximum number of columns to be displayed when calling a pandas object.
    width : int, optional
        Maximum width when displaying called pandas objects.
    """
    import pandas as pd
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_columns', max_columns)
    pd.set_option('display.width', width)


def corr_list(df, target, thresh=0.1, sort=True, fill=True):
    """
    List Most Correlated Features

    Returns a pandas Series with the most correlated features to a certain
    target variable. The function will return features with a correlation value
    bigger than some threshold, which can be adjusted.

    Parameters
    ----------
    df : pandas DataFrame
        `df` must contain only numerical values.
    target : str or int
        String or integer indicating the target variable.
    thresh : float, optional
        Float indicating the minimum correlation between a feature and the
        target above wich the feature will be present in the returned list.
        Default value is 0.1.
    sort : bool, optional
        Wheter to sort the returned pandas Series. If True, it will be sorted
        descending. Default value is False.
    fill : bool, optional
        Wheter to fill null values. If True, Null values will be replaced
        with 0's. Default value is False.

    Returns
    -------
    pandas Series
    """
    from numpy import abs as npabs
    if fill:
        interest = df.corr().fillna(0)[target]
    else:
        interest = df.corr()[target]
    interest = interest[npabs(interest) > thresh]
    if len(interest) > 0:
        if sort:
            return interest.sort_values(ascending=False)
        else:
            return interest
    else:
        return []


def reduce_memory(df):
    """
    Memory Consumption Reduction

    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.

    Parameters
    ----------
    df : pandas DataFrame

    Returns
    -------
    pandas DataFrame
        Memory-optimized pandas object
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min \
                        and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min \
                        and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min \
                        and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min \
                        and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min \
                        and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min \
                        and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'
          .format(100 * (start_mem - end_mem) / start_mem))
    return df


def get_prj_path(proj_name):
    """
    Project Absolute Path

    Returns a string describing the absolute path of the project root folder.

    Parameters
    ----------
    proj_name : str
        String describing the project root folder name.
    """
    import os
    curr = os.getcwd().split('/')
    while curr[-1] != proj_name:
        curr.pop()
    path = '/'.join(curr)
    return path


def rm_dot(item):
    """
    Limpeza de Pontos Duplos

    Deletes extra dots (thousands dot) in float values being represented as
    unprocessed strings.

    Parameters
    ----------
    item : str
        String containing the unprocessed float value to be cleaned before
        type-casting.

    Returns
    -------
    str
        String containing the unprocessed float value to be cleaned before
        type-casting, sans any extra dots.
    """
    if len(str(item).split('.')) > 2:
        return item.replace('.', '', 1)
    else:
        return item


def check_cardinality(df, cat_cols, threshold=8):
    """
    Check categorical cardinality

    Checks the cardinality of categorical features of a given dataset. Returns
    two dictionaries, one for features with low cardinality and another for
    features with high cardinality. The low/high cardinality criteria can be
    tunned with the `threshold` parameter.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset whose categorical features will be analyzed.
    cat_cols : list of str
        List of column names. The columns must be all categorical.
    threshold : int, optional
        Numeric criteria to separate low cardinality features from high
        cardinality ones. Default value is 8.

    Returns
    -------
    low_card : dict
        Dictionary containing the name of the low cardinality features as keys
        and their cardinality as values.
    high_card : dict
        Dictionary containing the name of the high cardinality features as keys
        and their cardinality as values.
    """
    from collections import Counter
    high_card = {}
    low_card = {}
    for col in cat_cols:
        rank = len(Counter(df[col]))
        if rank <= threshold:
            low_card[col] = rank
        else:
            high_card[col] = rank
    return low_card, high_card


def one_hot_enc(df, ohe_cols):
    """
    Perform One-Hot Encoding

    Wrapper for perfoming One-Hot Encoding on a set of categorical columns from
    a given DataFrame. The original categorical column is dropped. Good for
    encoding categorical features with low cardinality.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to be encoded.
    ohe_cols : list of str
        List containing the names of the categorical columns to be encoded.

    Returns
    -------
    pandas.DataFrame
        Original dataset with categorical columns replaced with groups of
        encoded numerical columns.
    """
    for col in ohe_cols:
        df_ohe = pd.get_dummies(df[col], preffix=col)
        df = pd.concat([df, df_ohe], axis=1)
        df.drop(col, axis=1, inplace=True)
    return df


def binary_encoding(df, bin_cols):
    """
    Perform Bniary Encoding

    Wrapper for perfoming Binary Encoding on a set of categorical columns from
    a given DataFrame. The original categorical columns are dropped. Good for
    encoding categorical features with high cardinality.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to be encoded.
    ohe_cols : list of str
        List containing the names of the categorical columns to be encoded.

    Returns
    -------
    pandas.DataFrame
        Original dataset with categorical columns replaced with groups of
        encoded numerical columns.
    """
    for col in bin_cols:
        enc = BinaryEncoder(cols=col)
        bin_enc = enc.fit_transform(df[col])
        df = pd.concat([df, bin_enc], axis=1)
        df.drop(col, axis=1, inplace=True)
    return df


def get_corr_sparse_df(df, client_id, product_id, tgt_feature,
                       tgt_filter=True, tgt_val=1):
    """
    Get sparse matrix of scores

    Returns a sparse score matrix of implicit Client -> Product interactions.
    The scores are wightened by the correlation of each feature with the
    target feature. It is also possible to filter the values from the target
    feature we want to be considered in the building of the sparse matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing Product_ID, Client_ID, the target feature (usually
        indicating some kind of product conversion) and other independent
        features available for analysis. The dataset must be in the
        normalized form of transactional tables, where each row is ONE
        interaction of a given Client with some Product. Moreover, all the
        columns in the dataset must be numerical and already cleaned.
    client_id : str
        Name of the column containing the Client_IDs.
    product_id : str
        Name of the column containing the Product_IDs.
    tgt_feature : str
        Name of the columns containing the Target Feature.
    tgt_filter : bool, optional
        Flag indicating wheter or not to filter the Target Feature values to be
        conseidered in the building of the sparse matrix. Default value is
        True.
    tgt_val : int or float, optional
        Value of the Target Feature to be used for filtering. Only used if
        `tgt_feature=True`. Default value is 1 (usually indicating positive
        product conversion).

    Returns
    -------
    df_spr : pandas.DataFrame
        Sparse dataset containing the Client -> Product implicit interaction
        scores.
    """
    # Auxiliary tables declaration
    df_keys = df[[client_id, product_id, tgt_feature]]
    df_score = df.drop([client_id, product_id, tgt_feature], axis=1)

    # Calculating the score weight matrix, using the correlation of each
    # feature with the target variable.
    weights = df.drop([client_id, product_id], axis=1) \
                .corr() \
                .fillna(0)[tgt_feature]
    weights.drop(tgt_feature, inplace=True)

    # Building scores dataset, dot-multiplying the interaction values with
    # their corresponding weight.
    df_score = pd.DataFrame(df_score.values * weights.values,
                            index=df_score.index,
                            columns=df_score.columns)
    score = df_score.sum(axis=1)
    df_keys = pd.concat([df_keys, score], axis=1)
    df_keys.rename(columns={0: 'Product_Score'}, inplace=True)

    if tgt_filter:
        # Filtering by the given 'target value' of the given 'target variable',
        # useful to focus on positive conversion, e.g.
        df_keys = df_keys[df_keys.tgt_feature == tgt_val]
        df_keys.drop(tgt_feature, axis=1, inplace=True)
    else:
        df_keys.drop(tgt_feature, axis=1, inplace=True)

    # Building the weightened score sparse matrix
    df_spr = df_keys.groupby([client_id, product_id]).mean().unstack()
    df_spr.fillna(0, inplace=True)
    return df_spr


def evaluate_for_scaling(df, transform=False, lower_quant=25, upper_quant=75):
    """
    Evaluate and perform dataset sclaing

    Evaluates if a given dataset needs scaling by checking the differences in
    the orders of magnitudes of its features. Also, can perform Robust Scaling,
    returning the transformed dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to be analysed. All of its features must be numeric.
    transform : bool, optional
        Flag indicating wheter to scale the transform the dataset with Robust
        Scaling. Default value is False.
    lower_quant : int, optional
        Lower boundary of the Quantile Range to be considered in the Robust
        Scaler. Default value is 25.
    upper_quant : int, optional
        Upper boundary of the Quantile Range to be considered in the Robust
        Scaler. Default value is 75.

    Returns
    -------
    bool
        Flag indicating if the dataset needs scaling (True) or not (False).
    pandas.DataFrame
        If transform=False, returns the original DataFrame. Otherwise, returns
        the scaled DataFrame.
    """
    df_std_mean = df.mean().std()
    df_median_mean = df.mean().median()

    if df_std_mean > df_median_mean:
        if transform:
            # Scaling the features
            from sklearn.preprocessing import RobustScaler
            rsc = RobustScaler(quantile_range=(lower_quant,
                                               upper_quant)).fit(df)
            df_scl = pd.DataFrame(rsc.transform(df), index=df.index,
                                  columns=df.columns)
            return True, df_scl
        else:
            return True, df
    else:
        return False, df


def explained_variability(df, threshold=0.95):
    """
    Determine Number of Principal Components

    Returns a PCA-transformed dataset with enough Principal Components to
    explain a certain amout of variability. Used for sparse datasets,
    leveraging the properties of the TruncatedSVD transformer from sklearn.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to be transformed.
    threshold : float
        Minimum percentagem of variability to be explained by the Principal
        Components.

    Returns
    -------
    transf : sklearn.decomposition.truncated_svd.TruncatedSVD
        The fitted TruncatedSVD transformer, utilizing the determined number
        of Principal Components.
    df_pca : pandas.DataFrame
        Transformed dataset.
    """
    from sklearn.decomposition import TruncatedSVD
    # Using all components initially
    transf = TruncatedSVD(n_components=df.shape[1]-1,
                          random_state=42).fit(df)
    s = pd.DataFrame(transf.explained_variance_ratio_,
                     index=range(1, df.shape[1]),
                     columns=['explained_var_ratio'])
    s['cumulative_explained_ratio'] = s.cumsum()

    # Determine the number of components to explain threshold% of variance
    n_pc = s[s['cumulative_explained_ratio'] <= threshold].shape[0]+1
    transf = TruncatedSVD(n_components=n_pc, random_state=42).fit(df)
    df_pca = pd.DataFrame(transf.transform(df), index=df.index)

    print('Explained Variance Ratio: {} || Number of components: {}'.
          format(sum(transf.explained_variance_ratio_),
                 transf.components_.shape[0]))
    return transf, df_pca


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
    from sklearn.model_selection import ShuffleSplit, GridSearchCV
    import time

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
    import time
    from sklearn.metrics import silhouette_score

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
