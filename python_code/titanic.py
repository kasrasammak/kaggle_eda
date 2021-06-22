#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 23:45:48 2021

@author: owlthekasra
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('titanicChallenge/train.csv')

#%%
train['Fare'].describe()
train['Fare_bins'] = pd.cut(x=train['Fare'],bins = [0,8,15, 35,np.inf], labels=[0,1,2,3])
train.groupby('Fare_bins').size() / len(train)
#%% 
def impute_nan_most_frequent_category(DataFrame,ColName):
    # .mode()[0] - gives first category name
     most_frequent_category=DataFrame[ColName].mode()[0]
    
    # replace nan values with most occured category
     DataFrame[ColName + "_Imputed"] = DataFrame[ColName]
     DataFrame[ColName + "_Imputed"].fillna(most_frequent_category,inplace=True)


impute_nan_most_frequent_category(train,'Embarked')
train =  train.drop("Embarked", axis=1).rename(columns={"Embarked_Imputed": "Embarked"})
#%%
X_train = train.drop("Survived", axis=1)
y_train = train["Survived"].copy()
#%%

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])


cat_attribs = ["Sex", "Embarked"]
drop_train = X_train.drop(["Cabin", "Name", "Ticket", "PassengerId"], axis=1)
num_attribs = drop_train.drop(cat_attribs, axis=1).columns.tolist()

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])

#%%
train_prepared = full_pipeline.fit_transform(drop_train)
#%%
import methods as md

knn = md.initializeModel('knn')
forest = md.initializeModel('forest')
lda = md.initializeModel('lda')
tree = md.initializeModel('tree')
#%%
knn.fit(train_prepared, y_train)
#%%
from sklearn.metrics import accuracy_score

y_pred = knn.predict(train_prepared)
score = accuracy_score(y_train, y_pred)
kappa = md.getKappa(y_train, y_pred)
#%% VALIDATION
test = pd.read_csv('titanicChallenge/test.csv')
impute_nan_most_frequent_category(test,'Embarked')
test = test.drop("Embarked", axis=1).rename(columns={"Embarked_Imputed": "Embarked"})

#%% Final Prediction
drop_test = test.drop(["Cabin", "Name", "Ticket", "PassengerId"], axis=1)
test_prepared = full_pipeline.fit_transform(drop_test)

final_prediction = knn.predict(test_prepared)






