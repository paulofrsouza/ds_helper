#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from category_encoders import BinaryEncoder
from collections import Counter

from sklearn.utils import resample
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt


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
    void = pd.DataFrame(np.sum(df.isna()), columns=['absolute'])
    void['percentage'] = round((void.absolute / df.shape[0]) * 100, 2)

    if show_plot:
        print('\n\n')
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
    check = check_nan(df, show_plot=False)
    return df.drop(check[check.percentage > perc].index.values, axis=1)


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
    if fill:
        interest = df.corr().fillna(0)[target]
    else:
        interest = df.corr()[target]
    interest = interest[np.abs(interest) > thresh]
    if len(interest) > 0:
        if sort:
            return interest.sort_values(ascending=False)
        else:
            return interest
    else:
        return []


def rm_dot(item):
    """
    Cleaning of extra dots

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
