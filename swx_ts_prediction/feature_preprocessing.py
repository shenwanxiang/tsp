# coding=utf-8
from __future__ import division
from scipy.stats import skew
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
'''
@Feature processing in time-series data_frame
@Author:shenwanxiang
@Date:2017-06
'''
def remove_low_variance_features(data_frame,var = 0.8):
    #get the 0.8 variance features by fitting VarianceThreshold
    #remove some feature who's variance that is below 0.8
    n_features_originally = data_frame.shape[1]
    selector = VarianceThreshold(var)
    selector.fit(data_frame)
    # Get the indices of zero variance feats
    feat_ix_keep = selector.get_support(indices=True)
    orig_feat_ix = np.arange(data_frame.columns.size)
    feat_ix_delete = np.delete(orig_feat_ix, feat_ix_keep)
    # Delete zero variance feats from the original pandas data frame
    data_frame = data_frame.drop(labels=data_frame.columns[feat_ix_delete],
                                 axis=1)
    # Print info
    n_features_deleted = feat_ix_delete.size
    print(" Deleted %s / %s features (= %.1f %%)" % (n_features_deleted, n_features_originally,
                                                      100.0 * (np.float(n_features_deleted) / n_features_originally)))
    return data_frame

def check_skew_log(df,alpha = 0.75):
    # Transform the skewed numeric features by taking log(feature + 1).
    # This will make the features more normal.
    skewed = df.apply(lambda x: skew(x.dropna().astype(float)))
    skewed = skewed[skewed > alpha]
    skewed = skewed.index
    df2 = df.copy()
    df2[skewed] = np.log1p(df2[skewed])
    return df2,skewed

def diff_shift_lag(df,diff_lag = 10,shift_lag = 2):
    #diff and shift
    dflist = []
    for i in range(shift_lag):
        for j in range(diff_lag):
            dff = df.diff(j+1).shift(i+1)
            dff.columns = ['shift'+str(i+1) + '_diff'+str(j+1)]
            dflist.append(dff)
    return pd.concat(dflist,axis =1)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    from pandas import DataFrame
    from pandas import concat
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
       dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
    Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg