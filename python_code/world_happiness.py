#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 23:31:08 2021

@author: owlthekasra
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("data/world-happiness-report-2021.csv", index_col="Country name")

#%% feature engineering
df["RegionalLadder"] = df.groupby("Regional indicator")["Ladder score"].transform("mean")
df["RegionalGDP"] = df.groupby("Regional indicator")["Logged GDP per capita"].transform("mean")
df["WhiskerDifference"] = df["upperwhisker"] - df["lowerwhisker"]
#%% separate categorical and numerical attributes for encoding
cat=df.select_dtypes(exclude=np.number)
num=df.select_dtypes(include=np.number)

#correlation matrix for the numerical attributes
corrmat = num.corr()

#%% Normalize numerical data, one hot encode categorical data
from sklearn.preprocessing import StandardScaler, OneHotEncoder
scal = StandardScaler().fit_transform(num)
scal = pd.DataFrame(scal, columns = num.columns)
encode = OneHotEncoder(sparse=False).fit(cat)
cat_encoded = pd.DataFrame(encode.transform(cat))
cat_encoded.columns = encode.get_feature_names(cat.columns)
df_encoded = pd.concat([cat_encoded, scal],axis=1)

#%% Run linear regression on the data
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import methods as md

X_train, X_test, y_train, y_test= train_test_split(df_encoded.drop("Ladder score",axis=1), df_encoded["Ladder score"], test_size=.3, random_state=42)
model = LinearRegression().fit(X_train,y_train)
y_pred = model.predict(X_test)
lin_mse = mean_squared_error(y_test, y_pred)
lin_rmse = np.sqrt(lin_mse)
cv_score=cross_val_score(LinearRegression(),X_train, y_train, scoring="neg_root_mean_squared_error", cv=10)

#%% Create a heatmap based on minimum and maximum values matrix
minmaxmat = pd.DataFrame().reindex_like(corrmat)
for i in range(0,len(minmaxmat)):
    maxi = max(corrmat.iloc[:,i])
    mini = min(corrmat.iloc[:,i])
    for j in range(0,len(minmaxmat)):
        minmaxmat.iloc[j,i] = (corrmat.iloc[j,i]-mini)/(maxi-mini)
        
#plot and compare it to the correlation matrix
sns.heatmap(minmaxmat)
sns.heatmap(corrmat)
#%%
#plot correlation to ladder score
ladder_corr =df_encoded.corrwith(df["Ladder score"]).sort_values(ascending=False)
#plot correlation of GDP with the Happiness Index
sns.regplot("Logged GDP per capita", "Ladder score",df_encoded)

#%%
#show a boxplot distribution of the happiness index
sns.boxplot(df_encoded["Ladder score"])

#%%
#plot regional ladder scores by region
plt.figure(figsize=(30,6))
ax=sns.barplot(x=df.index, y=df["RegionalLadder"].sort_values(ascending=False), orient="v")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
ax.set_ylim([3.5, 7.5])

# plot each countries ladder score
plt.figure(figsize=(30,6))
ax2=sns.barplot(x=df.index, y=df["Ladder score"], orient="v")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, ha="right")
ax2.set_ylim([2.5, 8])
