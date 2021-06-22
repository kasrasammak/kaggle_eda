#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 23:36:44 2021

@author: owlthekasra
"""

#%% get data DO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fraud_df = pd.read_csv('isFraud.csv')
df = fraud_df.copy()

#%% analysis
fraud = fraud_df.iloc[1::100,:]
fraud['type'].value_counts();

corrmat = df.corr()

df['isFraud'].value_counts()
df['isFraud'].value_counts(normalize=True)
df['isFlaggedFraud'].value_counts()
df['isFlaggedFraud'].value_counts(normalize=True)

#%% histograms

fraud.hist(bins=50, figsize=(20,15))
plt.show()

fraud.groupby('type').hist()
plt.style.use('seaborn')
plt.show()
df.boxplot()
hey = np.log(df['amount'])[1:100,:]
yo = df['amount'][:100]
yo.hist(bins=20)
hey.hist(bins=20)

#%% more analysis
df['amount'].describe()
df['amount_bins'] = pd.cut(x=df['amount'], bins=[0, 10000, 30000, 100000, 300000, np.inf], labels=[1, 2, 3, 4, 5])

df.groupby(['type', 'isFraud']).size()

df[df['isFraud'] ==1].nameDest.nunique()
fraud_df.head()


#%% get rid of unwanted values DO
fraud_df = fraud_df.drop("oldbalanceDest", axis=1)
fraud_df = fraud_df.drop("oldbalanceOrg", axis=1)
fraud_df = fraud_df.drop("nameOrig", axis=1)
fraud_df = fraud_df.drop("nameDest", axis=1)
fraud_df = fraud_df[fraud_df.type.isin(['TRANSFER','CASH_OUT'])]
fraud_df = fraud_df.reset_index().iloc[:,1:]
#%% post analysis
fraud_df['isFraud'].value_counts()
fraud_df.groupby('type').size()
fraud_df['isFraud'].value_counts(normalize=True)
#%% Make set smaller, but stratify (for easier working purposes) DO
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)


for train_index, test_index in split.split(X=fraud_df, y=fraud_df['isFraud']):
    strat_train_set = fraud_df.loc[train_index]
    strat_test_set = fraud_df.loc[test_index]

strat_test_set['isFraud'].value_counts() / len(strat_test_set)
# use stratified set as new main set
stratified_set = strat_test_set.reset_index().iloc[:,1:].copy()
#%% Stratify set DO
split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)


for train_index, test_index in split2.split(X=stratified_set, y=stratified_set['isFraud']):
    train_set = stratified_set.loc[train_index]
    test_set = stratified_set.loc[test_index]

#%% initialize models DO
import methods as md

knn = md.initializeModel('knn')
forest = md.initializeModel('forest')
lda = md.initializeModel('lda')
tree = md.initializeModel('tree')

#%% Separate x and Y for train set DO
X_train = train_set.drop("isFraud", axis=1)
y_train = train_set["isFraud"].copy()

#%% One Hot Encoder (alone)
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder()
fraud_cat = df[['type']]
fraud_cat_1hot = one_hot_encoder.fit_transform(fraud_cat.values)
fraud_cat_1hot_array = fraud_cat_1hot.toarray()

#%% make pipelines DO
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


num_pipeline = Pipeline([
    ('std_scaler', StandardScaler())
])

drop_train = X_train.drop(["type"], axis=1)
num_attribs = drop_train.columns.tolist()
cat_attribs = ["type"]


full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])
#%% fit transform pipeline DO
X_train_prepared = full_pipeline.fit_transform(X_train)

#%% train test split  DO
from sklearn.model_selection import train_test_split

X_tr, X_t, y_tr, y_t = train_test_split(X_train_prepared, y_train, random_state=42)
#%%  fit model DO

tree.fit(X=X_tr, y=y_tr)
#%% get metrics DO
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

y_pred = tree.predict(X_t)
score = accuracy_score(y_t, y_pred)
kappa = md.getKappa(y_t, y_pred)


#%% define score function DO
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())
#%% k=fold cross validation DO
from sklearn.model_selection import cross_val_score
#  scoring methods: accuracy, balanced_accuracy, average_precision
scores = cross_val_score(estimator=tree, X=X_tr, 
                         y=y_tr, scoring='balanced_accuracy', cv=10)
display_scores(scores)
print("Score:", score)
#%% Separate x and Y for test set DO
X_test = test_set.drop("isFraud", axis=1)
y_test = test_set["isFraud"].copy()

#%% fit transform and metrics of validation set DO

X_test_prepared = full_pipeline.transform(X=X_test)
final_pred = tree.predict(X=X_test_prepared)
score = accuracy_score(y_test, final_pred)
kappa = md.getKappa(y_test, final_pred)

#%% predictions DO
test_pred = pd.DataFrame([final_pred, y_test.values]).T
test_pred.columns = ['predictions', 'test_values']

test_pred.groupby(['predictions', 'test_values']).size()
test_pred.groupby('test_values').size()
#%% confusion matrix DO
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(final_pred,y_test)


