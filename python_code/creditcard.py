#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 22:37:06 2021

@author: owlthekasra
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("creditcard.csv")
corrmat = df.corr()
correlations = corrmat["Class"].sort_values(ascending=False)
df_short = df.iloc[1::20, :]

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score
X_train, X_test, y_train, y_test = train_test_split(df.drop("Class", axis=1), df["Class"], test_size=0.5, random_state=42)


import methods as md
model = md.initializeModel("lda")

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
kappa = md.getKappa(y_test,y_pred)
kappa = cohen_kappa_score(y_test,y_pred)
cv_score = cross_val_score(md.initializeModel("lda"), X_train, y_train)
#%%
sns.histplot(data=df_short, x="Amount", hue="Class")

fraud = df[df["Class"] == 1]

sns.distplot()
df_short["Amount"].describe()
sns.boxplot(df_short["LogAmount"])

sns.histplot(df_short[df_short["Amount"] < 500],x="Amount", hue="Class")
df_short["LogAmount"] = np.log(df_short["Amount"])

df_short.groupby("Class")["Class"].size()
