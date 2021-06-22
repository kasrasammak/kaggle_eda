# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("data/marketing_data.csv")
#%% string replacements for Income

df = df.rename(columns={' Income ': 'Income'})
df.columns = df.columns.str.replace(' ', '')
df["Income"] = df["Income"].replace(regex=['\$',','], value='')
df["Income"] =pd.to_numeric(df["Income"])

#%%
df['Income'] = df['Income'].fillna(df['Income'].median())
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])

#%% feature engineering (from Kaggle solutions)
df['Dependents'] = df['Kidhome'] + df['Teenhome']
df['TotalMnt'] = df[[col for col in df.columns if 'Mnt' in col]].sum(axis=1)
df['TotalPurchases'] = df[[col for col in df.columns if 'Purchases' in col]].sum(axis=1)
df['TotalCampaignsAcc'] = df[[col for col in df.columns if 'Cmp' in col]+['Response']].sum(axis=1)
df['Year_Customer'] = pd.DatetimeIndex(df['Dt_Customer']).year
df[['ID', 'Dependents', 'Year_Customer', 'TotalMnt', 'TotalPurchases', 'TotalCampaignsAcc']].head()

#%%
df.isnull().sum().sort_values(ascending=False)

#%%
df.drop(columns=['ID', 'Dt_Customer'], inplace=True)
# one-hot encoding of categorical features
from sklearn.preprocessing import OneHotEncoder

# get categorical features and review number of unique values
cat = df.select_dtypes(exclude=np.number)
print("Number of unique values per categorical feature:\n", cat.nunique())

# use one hot encoder
enc = OneHotEncoder(sparse=False).fit(cat)
cat_encoded = pd.DataFrame(enc.transform(cat))
cat_encoded.columns = enc.get_feature_names(cat.columns)

# merge with numeric data
num = df.drop(columns=cat.columns)
df2 = pd.concat([cat_encoded, num], axis=1)
df2.head()
#%% impute median values for missing values
from sklearn.impute import SimpleImputer
impute = SimpleImputer(strategy="median")
X= df.drop(["Education","Marital_Status","Dt_Customer", "Country"], axis=1)
col_names = list(X.columns)

df[col_names] = impute.fit_transform(X)


#%%plotting 

edu_mean_income = df.groupby("Education")["Income"].mean()
country_mean_income = df.groupby("Country")["Income"].mean()
sns.barplot(x = country_mean_income.index, y=country_mean_income)
sns.swarmplot(x=df["Education"],y=df["Year_Birth"]).set_ylim(min(df["Year_Birth"]), max(df["Year_Birth"]+10))

Basic_edu = df[df["Education"] == "Basic"]["Income"]


#%%
sns.lmplot(x="TotalMnt", y="TotalPurchases", data=df[df["Income"] < 200000])
#%%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
X = df2.drop("NumStorePurchases", axis=1)
y= df2["NumStorePurchases"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_pred,y_test))
#%%
plt.figure(figsize=(8,3))
sns.distplot(df['NumStorePurchases'], kde=False, hist=True, bins=12)

#%%
from IPython.display import display
import eli5
from eli5.sklearn import PermutationImportance
import webbrowser
perm = PermutationImportance(lr, random_state=1).fit(X_test, y_test)
perm_df = eli5.format_as_dataframe(eli5.explain_weights(perm, feature_names = X_test.columns.tolist()))

#%%
# Write html object to a file (adjust file path; Windows path is used here)
with open('C:\\Tmp\\Desktop\purchases-importance.htm','wb') as f:
    f.write(html_obj.data.encode("UTF-8"))

# Open the stored HTML file on the default browser
url = r'C:\\Tmp\\Desktop\purchases-importance.htm'
webbrowser.open(url, new=2)

#%%
from sklearn.feature_selection import mutual_info_regression

X = df.drop("NumStorePurchases", axis=1)
y= df["NumStorePurchases"]
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
mi_scores, feats_ = make_mi_scores(X,y)
corrmat= df.corr()
#%%
Purchases_corr = corrmat["NumStorePurchases"].sort_values(ascending=False)

df.groupby("Kidhome")["Income"].mean()

df.groupby("Kidhome")["Kidhome"].count()/df["Kidhome"].count()

163 +167 + 144 + 163 + 30

#%% which campaign did the best (4)
df[df["AcceptedCmp1"]==1]["AcceptedCmp3"].count()
df[df["AcceptedCmp2"]==1]["AcceptedCmp3"].count()
df[df["AcceptedCmp3"]==1]["AcceptedCmp3"].count()
df[df["AcceptedCmp4"]==1]["AcceptedCmp3"].count()
df[df["AcceptedCmp5"]==1]["AcceptedCmp3"].count()

#%% US Spends least
you = df.groupby("Country")["NumStorePurchases"].count().sort_values(ascending=False)

#%% above average on gold products do spend more in store on average
mean_gold = df["MntGoldProds"].mean()
median_gold = df["MntGoldProds"].median()

above_ppl = df[df["MntGoldProds"].gt(mean_gold)]
less_ppl =df[df["MntGoldProds"].lt(mean_gold)]

df["GoldDiggers"] = pd.cut(df["MntGoldProds"], bins=[0,mean_gold,np.inf], labels=["NonDiggers","GoldDiggers"])
sns.barplot(df.groupby("GoldDiggers")["NumStorePurchases"].mean().index, df.groupby("GoldDiggers")["NumStorePurchases"].mean())

#%%
df.groupby("Education")["MntFishProducts"].count()
df.groupby("Education")["MntFishProducts"].mean()
sum_by_count = df.groupby("Education")["MntFishProducts"].sum() / df.groupby("Education")["MntFishProducts"].count()
df.groupby("Education")["MntFishProducts"].std()

money_on_fish = (df.groupby("Education")["MntFishProducts"].sum() -df.groupby("Education")["MntFishProducts"].mean() ) / df.groupby("Education")["MntFishProducts"].std()
ppl_on_fish = df.groupby("Education")["MntFishProducts"].count()
sns.barplot(sum_by_count.index, sum_by_count)
plt.title("Total Ppl who bought Fish By Education Group")

mi_scores_fish, feat = make_mi_scores(df.drop("MntFishProducts",axis=1),df["MntFishProducts"])

#%%
df['isMarried'] = df["Marital_Status"]=="Married"

# df['isMarried']=df['isMarried'].astype(int)
df["Married_edu"] = df["Education"] + "_" + df["isMarried"]
# df['isMarried'] = df.replace({True: 'Married', False: 'Unmarried'})
Fish_married = df.groupby(["Education", "isMarried"])["MntFishProducts"].sum()
Fish_married["eduMarried"] = Fish_married.index[0][0] + "_" + Fish_married.index[0][1].astype(str)
#%% Dont do this
mean_fish = df.groupby("Education")["MntFishProducts"].transform("mean")

df["mean_fish"] = (df["MntFishProducts"] -mean_fish)
yo = df.groupby("Education")["mean_fish"].sum() / df.groupby("Education")["MntFishProducts"].std()
sns.barplot(yo.index, yo)

#%%
camp_succ_country = df.groupby("Country")["AcceptedCmp1"].mean()*100
camp_succ_country = pd.concat([camp_succ_country, df.groupby("Country")["AcceptedCmp2"].mean()*100], axis=1)
camp_succ_country = pd.concat([camp_succ_country, df.groupby("Country")["AcceptedCmp3"].mean()*100], axis=1)
camp_succ_country = pd.concat([camp_succ_country, df.groupby("Country")["AcceptedCmp4"].mean()*100], axis=1)
camp_succ_country = pd.concat([camp_succ_country, df.groupby("Country")["AcceptedCmp5"].mean()*100], axis=1)

sns.lineplot(data=camp_succ_country[camp_succ_country.index != "ME"])
camp_succ_country, df.groupby("Country")["AcceptedCmp3"].count()
plt.ylabel("% of Population")

success = df.groupby("AcceptedCmp1")["AcceptedCmp1"].count()
success =pd.concat([success, df.groupby("AcceptedCmp2")["AcceptedCmp2"].count()], axis=1)
success =pd.concat([success, df.groupby("AcceptedCmp3")["AcceptedCmp3"].count()], axis=1)

success =pd.concat([success, df.groupby("AcceptedCmp4")["AcceptedCmp4"].count()], axis=1)

success =pd.concat([success, df.groupby("AcceptedCmp5")["AcceptedCmp5"].count()], axis=1)

#%%
filter_col = [col for col in df if col.startswith('Mnt')]
heyo=df[filter_col].mean()
heyoo = df[filter_col].sum() 
ax = sns.barplot(heyo.index, heyo)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
