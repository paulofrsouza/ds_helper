#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from warnings import filterwarnings

sns.set()
filterwarnings('ignore')


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
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_columns', max_columns)
    pd.set_option('display.width', width)
