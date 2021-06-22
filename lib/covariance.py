#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 23:01:26 2020

@author: owlthekasra
"""

import numpy as np
import pandas as pd
import math
from sklearn.feature_selection import mutual_info_regression

# get covariance and correlation scalar given two lists
def get_covariance(arr1, arr2):
    n = len(arr1)
    cov = 1/(n-1)*sum((arr1-np.mean(arr1))*(arr2-np.mean(arr2)))
    return cov

def get_correlation(arr1, arr2):
    num = sum((arr1-np.mean(arr1))*(arr2-np.mean(arr2)))
    denom1 = math.sqrt(sum((arr1-np.mean(arr1))**2))
    denom2 = math.sqrt(sum((arr2-np.mean(arr2))**2))
    corr = num/(denom1*denom2)
    return corr

# average of all trials
def get_ERP(df, trials, M):
    l1 = []
    erp = pd.DataFrame()
    for i in range(M):
        l1.append(pd.DataFrame(np.array_split(np.array(df.iloc[:, i]), trials)))
        erp = pd.concat([erp, np.mean(l1[i])], axis = 1)
    return erp

def get_covariance_matrix(df, M):
    covmat1 = pd.DataFrame(index = range(M), columns=range(M))
    for i in range(M):
        for j in range(M):
            subi = df.iloc[:, i] - np.mean(df.iloc[:, i])
            subj = df.iloc[:, j]-np.mean(df.iloc[:, j])
            covmat1.iloc[i, j] = sum(subi*subj)/(len(df)-1)
    return covmat1

def get_correlation_matrix(df, M):
    corrmat1 = pd.DataFrame(index = range(M), columns=range(M))
    for i in range(M):
        for j in range(M):
            subi = df.iloc[:, i] - np.mean(df.iloc[:, i])
            subj = df.iloc[:, j] - np.mean(df.iloc[:, j])
            corrmat1.iloc[i, j] = sum(subi*subj)/(math.sqrt(sum(subi**2))*math.sqrt(sum(subj**2)))
    return corrmat1

# insert list of dataframes to calculate average of matrices for individual trials
    # use carefully, heavy computations
def get_cov_mat_inds(listofdfs, trials, M):
    covmat1 = pd.DataFrame(0, index = range(M), columns=range(M))
    for i in range(0, trials):
        covmat1 = covmat1 + get_covariance_matrix(listofdfs[i], M)
    covmat1 = covmat1/trials
    return covmat1

def get_corr_mat_inds(listofdfs, trials, M):
    corrmat = pd.DataFrame(0, index = range(M), columns=range(M))
    for i in range(0, trials):
        corrmat = corrmat + get_correlation_matrix(listofdfs[i], M)
    corrmat = corrmat/trials
    return corrmat

# first calculate the ERP of all trials and then calculate the covariance matrix
    # less computationally expensive
def get_cov_mat_ERP(df, trials, M):
    erp = get_ERP(df, trials, M)
    cov = get_covariance_matrix(erp, M)
    return cov

def get_corr_mat_ERP(df, trials, M):
    erp = get_ERP(df, trials, M)
    corr = get_correlation_matrix(erp, M)
    return corr

def get_mean_center(df):
    dfM = pd.DataFrame()
    for col in range(0,len(df.columns)):
        dfM = dfM.append(df.iloc[:,col] - np.mean(df.iloc[:,col]))
    return dfM

def make_mi_scores1(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores, discrete_features


# # can replace path str with your own path
# path = '/Users/owlthekasra/Documents/Code/Python/AudioStimulus'
# blinkDf = pd.read_csv(path + '/nonBlinkTraining.csv')

# # easier way to get covariance matrix
# blinkCovMat1 = get_cov_mat_ERP(blinkDf, 68, 4)
# blinkErp = get_ERP(blinkDf, 68, 4)
# blinkErpM = get_mean_center(blinkErp)
# blinkCovMat2 = blinkErpM.dot(blinkErpM.T) / (len(blinkErp) - 1)

# # split data before putting calculating matrices for individual trials
# mainsplit = np.array_split(blinkDf, 68)
# covmatind = get_cov_mat_inds(mainsplit, 68, 4)
# corrmatind = get_corr_mat_inds(mainsplit, 68, 4)
# corrmat = get_correlation_matrix()
