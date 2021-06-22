#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 00:22:46 2021

@author: owlthekasra
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
df = pd.read_csv("Churn_Modelling.csv")

#%%
df = df.replace(' ', '')
df = df.drop(["CustomerId", "Surname"], axis=1)
#%%
df["BalanceSettled"] = pd.cut(df["Balance"], bins=[-np.inf, 0, np.inf], labels=[np.int(0),np.int(1)])
df[df["EstimatedSalary"]< 10000]
df["SalaryBins"] = pd.cut(df["EstimatedSalary"], bins=[0, 10000, 50000, 150000, np.inf], labels=[1, 2, 3, 4])
df["BalanceToSalary"] = df["Balance"] / df["EstimatedSalary"]
corrmat = df.corr()
num = df.select_dtypes(include=np.number).columns
sns.heatmap(df[num])
#%%
sns.histplot(data=df, x="NumOfProducts", hue="Exited")
sns.histplot(data=df, x="Age", hue="Exited")
fig.update_layout(
    height=500, width=800, 
    title_text='NumOfProducts Feature in Detail',
    yaxis_title='Percentage of Churn',
    yaxis={'ticksuffix':'%'}
)
# fig = px.histogram(df, x='EstimatedSalary', color='Exited', marginal='box', color_discrete_map={0: '#636EFA', 1: '#EF553B'}, barmode='overlay', nbins=20)
# fig.update_layout(height=500, width=800, 
#                   title_text='EstimatedSalary Feature in Detail')
# fig.show()
#%%


sns.boxplot(x=df["Exited"], y=df["Balance"], vertical=True)
df["Age"].corr(df["Exited"])

#%%


#%%

sns.swarmplot(df["SalaryBins"], df["Balance"])
sns.lmplot("EstimatedSalary", "Balance", data=df)
#%%
from sklearn.preprocessing import OneHotEncoder
cat = df.select_dtypes(exclude=np.number)
num = df.select_dtypes(include=np.number)

encode=OneHotEncoder(sparse=False).fit(cat)
# for modelling
# cat_cols = list(cat.columns)
# cat[cat_cols[1]].nunique()

encoded = pd.DataFrame(encode.transform(cat))
encoded.columns = encode.get_feature_names(cat.columns)
df = pd.concat([encoded,num ],axis=1)
df = df.set_index("RowNumber")

#%%
X = df.drop("Exited", axis=1)
y=df["Exited"]

#%%
df.groupby("Exited")["Age"].count()

#%%
from sklearn.feature_selection import mutual_info_regression

corrmat = df.corr()
correlations = corrmat["Exited"]
correlations = correlations.sort_values(ascending=False)
corr_cols = list(correlations.index)[1:6]
#%%
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores, discrete_features

mi_scores, f1  = make_mi_scores(X, y)

#%%
columns = np.full((corrmat.shape[0],), True, dtype=bool)
for i in range(corrmat.shape[0]):
    for j in range(i+1, corrmat.shape[0]):
        if corrmat.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = df.columns[columns]
data = df[selected_columns]
#%%
selected_columns = selected_columns[1:].values
#%%
import statsmodels.formula.api as sm
def backwardElimination(x, Y, sl, columns):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.ols(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)
                    
    regressor_OLS.summary()
    return x, columns
SL = 0.05
data_modeled, selected_columns = backwardElimination(X.values,y .values, SL, selected_columns)
#%%
import methods as md
from sklearn.model_selection import train_test_split, cross_val_score
knn = md.initializeModel("tree")

X_train, X_test, y_train, y_test =train_test_split(X[corr_cols],y, test_size=.2, random_state=42)

mod, acc, _, _ = md.fitPredictValSet(X_train,y_train,X_test,y_test, "forest")

knn.fit(X_train, y_train)

pred = knn.predict(X_test)

cv_scores = cross_val_score(md.initializeModel("forest"), X_train, y_train, cv=10)