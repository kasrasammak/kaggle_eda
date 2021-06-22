#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:55:35 2020

@author: owlthekasra
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from math import sqrt
from sklearn.metrics import mean_squared_error 
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


#PREPROCESSING/BASE FUNCTIONALITY

def add(a, b):
    return a + b

def dropNoVarianceColumns(df):
    for col in df.columns:
        val = df[col][0]
        if (df[col] == val).sum()/len(df) == 1:
            df = df.drop(col,axis=1)
    return df

def variance_threshold_selector(data, threshold=0.5):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]

def dropNthObs(df, n):
    return df.iloc[::n, :]

def createNewTarget(df, df2, name):
    y = df2[name]
    return df, y  
    
def appender(strt, end):
    appenders = range(strt, end)
    mylists = [[], [], [], []]
    for x in appenders:
        for lst in mylists:
            lst.append(x)
    return mylists


#ACCURACY
    
# get accuracy metric by hand
    
def getR2(test, pred):
    sse = sum((abs(test - pred))**2)
    tss = sum((abs(test - pred.mean()))**2)
    r2 = 1 - sse/tss
    return r2

def getKappa(test, pred):
    cm = confusion_matrix(test, pred)
    num = 0
    denom = 0
    obs = 0
    for i in range(0,len(cm)):
        num = num + (sum(cm[i])*sum(cm[:,i]))
        denom = denom+sum(cm[i])
        obs = obs + cm[i,i]
    expected = num/denom
    kappa = (obs - expected)/(denom - expected)
    return kappa

def kasra_accuracy_score(y_test, y_pred):
    num = 0
    denom = len(y_test)
    for i in range(0, y_test.shape[0]) :
        if y_test[i] == y_pred[i]:
                num = num +1
    return num/denom

# assess accuracies

# return classification model with best accuracy
def getBest(model, acc, index, name, finalModel, acc_type=0):
    if (index == 0):
        finalModel = (model, acc, name)
    elif (acc[acc_type] > finalModel[1][acc_type]):
        finalModel = (model, acc, name)
    print(acc[acc_type])
    print(finalModel[1])
    return finalModel

# return regression model with best accuracy
def getBestReg(model, acc, index, name, finalModel):
    if (index == 0):
        finalModel = (model, acc[1], acc[0], name)
    elif (acc[1] < finalModel[1]):
        finalModel = (model, acc[1], acc[0], name)
    return finalModel

#cross validation
def checkKCV(X, y, cv, model):
    train_test = train_test_split(X, y, random_state=1)
    accuracies = cross_val_score(estimator = model, X = train_test[0], y = train_test[2], cv = cv, n_jobs = -1)
    return accuracies, train_test
            

#MODELS
    
#initialize models given a model name
    
def initializeModel(name, param_1 = 5, neighbors=5, radius = 1.0, n_estimators=100, max_depth=2, weights='uniform', kernel='linear', n_components=1):
    if (name == 'knn'):
        model =  KNeighborsClassifier(n_neighbors = param_1)
    elif (name == 'tree'):
        model = tree.DecisionTreeClassifier()
    elif (name == 'forest'):
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    elif (name == 'knnr'):
        model = KNeighborsRegressor(n_neighbors=neighbors)
    elif (name == 'rnr'):
        model = RadiusNeighborsRegressor(radius=radius, weights=weights, n_jobs=-1)
    elif (name == 'svm'):
        model = SVC(kernel=kernel)
    elif (name == 'lda'):
        model = LDA(n_components=n_components)

    return model

#splitting, fitting, predicting
    
def splitNFit(X, y, model):
    train_test = train_test_split(X, y, random_state=1)
    model = model
    model.fit(train_test[0], train_test[2])
    return train_test, model

def predictTest(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    if (name != 'rnr') & (name != 'knnr'):
        acc = accuracy_score(y_test, y_pred)
        kappa = getKappa(y_test, y_pred)
        acc = (acc, kappa)
    else:
        if (np.any(np.isnan(y_pred)) != True):
            r2 = getR2(y_test, y_pred)
            error = sqrt(mean_squared_error(y_test,y_pred))
            acc = (r2, error)
    return acc, y_pred, y_test

def fitPredict(X, y, model, name='knn'):
    train_test, model = splitNFit(X, y, model)
    # acc_train, _ = checkKCV(X, y, cv, model)
    acc, y_pred, y_test = predictTest(model, train_test[1], train_test[3], name)
    return  model, y_pred, acc, train_test

def fitPredictValSet(X, y, X_val, y_val, name, param=5, neighbors=5, radius = 1.0, n_estimators=100, max_depth=2, n_components=1):
    model = initializeModel(name, param_1 = param, neighbors= neighbors, radius = radius, n_estimators=n_estimators, max_depth=max_depth, n_components=n_components)
    model, _, _, _ = fitPredict(X, y, model, name)
    acc, pred, y_test = predictTest(model, X_val, y_val, name)
    return model, acc, pred, y_test

# evaluate models, return list of models and their accuracies as well as best model
    
# evaluate same model with different hyperparameters
def evalHyperParams(X, y, X_val, y_val, name, pStrt, pSize, acctype=0):
    mylists = [[],[],[],[]]
    if (name != 'rnr') & (name != 'knnr'):
        finalModel = (0,(0,0),0)
    else: 
        finalModel = (0,10000000000,0,0)
    for i in range(pStrt,pStrt+pSize):
        model = fitPredictValSet(X, y, X_val, y_val, name, i, neighbors=i, radius=i/2)
        for j in range(0, len(mylists)):   
            mylists[j].append(model[j])
        if (name != 'rnr') & (name != 'knnr'):
            finalModel = getBest(mylists[0][i-pStrt], mylists[1][i-pStrt], i-pStrt, i, finalModel, acc_type=acctype)
        else:
            finalModel = getBestReg(mylists[0][i-pStrt], mylists[1][i-pStrt], i-pStrt, i, finalModel)
    return finalModel, mylists

# evaluate different models
def evaluateModels(X, y, X_val, y_val, modelnames, acctype=0):
    models = [[],[],[],[]]
    finalModel = (0,0,0)
    for i in range(0, len(modelnames)):
        model = fitPredictValSet(X, y, X_val, y_val, modelnames[i][0], param=modelnames[i][1])
        for j in range(0, len(models)):   
            models[j].append(model[j])
        finalModel = getBest(models[0][i], models[1][i], i, modelnames[i], finalModel, acc_type = acctype)
    return finalModel, models
