#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:56:02 2021

@author: owlthekasra
"""


import pandas as pd
import numpy as np
df = pd.read_csv("Cricket.csv")

#%% pivot Margin column into a runs margin and a wickets margin
yo =  df[df["Margin"].str.contains('wickets', na=False)]
yo = yo[["Margin","Scorecard"]].rename(columns={"Margin":"MarginByWickets"})
df = pd.merge(df, yo,  on="Scorecard", how="left")
df["MarginByWickets"] = pd.to_numeric(df["MarginByWickets"].str.replace(" wickets", "")).fillna(0)

yo2 = df[df["Margin"].str.contains('runs', na=False)]
yo2 = yo2[["Margin","Scorecard"]].rename(columns={"Margin":"MarginByRuns"})
df = pd.merge(df, yo2,  on="Scorecard", how="left")
df["MarginByRuns"] = pd.to_numeric(df["MarginByRuns"].str.replace(" runs", "")).fillna(0)
#%%
import matplotlib.pyplot as plt
import seaborn as sns
df['Year'] = pd.DatetimeIndex(df['Match Date']).year

# create respective columns for each team and number of games played
teamcard= pd.merge(
    df[df["Year"]==2018].groupby('Team 1')["Scorecard"].count().sort_values(ascending=False).reset_index().rename(columns={"Team 1":"Teamname","Scorecard": "Scores1"}),
    df[df["Year"]==2018].groupby("Team 2")["Winner"].count().sort_values(ascending=False).reset_index().rename(columns={"Team 2":"Teamname", "Winner":"Scores2"}),
    on="Teamname", how="outer"
    )
#%%
# create a column to count number of inconclusive matches (for countries with such matches)
inc = df[df.Winner.isin(["no result","tied"])]
inc_country=pd.DataFrame(np.concatenate((inc["Team 1"].values,inc["Team 2"].values), axis=0)).rename(columns={0:"Teamname"})
inky = inc_country.reset_index()
inky["Inconclusive"] = inky.groupby("Teamname")["index"].transform("count")
inky = inky.drop("index", axis=1).drop_duplicates()
#%%
# add up total games for each team
teamcard=teamcard.fillna(0)
teamcard["TotalGames"] = teamcard["Scores1"] + teamcard["Scores2"]
#%%
# create column with number of wins for each team
hi = df[df["Year"]==2018][~df.Winner.isin(["no result","tied"])].groupby('Winner')["Scorecard"].count().reset_index().rename(columns={"Winner": "Teamname", "Scorecard":"Wins"})
# merge the wins and inconclusive matches, then calculate losses
teamcard = pd.merge(teamcard, hi, on="Teamname", how="outer")
teamcard = pd.merge(teamcard, inky, on="Teamname", how="outer" ).fillna(0)
teamcard["Loses"] = teamcard["TotalGames"] - (teamcard["Wins"] + teamcard["Inconclusive"])
#%% calculate conversion ratios and plot
teamcard["WinsToLosses"] = teamcard["Wins"] / teamcard["Loses"]
teamcard = teamcard.sort_values(by="WinsToLosses",ascending=False)

# plot conversion ratios
ax = sns.barplot(x=teamcard.Teamname, y=teamcard.WinsToLosses)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.title("Conversion Ratio")
#%% plot conversion ratios with teams who had more losses than wins
ax2 = sns.barplot(x=teamcard[teamcard.WinsToLosses.lt(1)].Teamname, y=teamcard[teamcard.WinsToLosses.lt(1)].WinsToLosses)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=40, ha="right")
plt.title("Conversion Ratio of Teams with more Losses")


#%% plot total overall games played
dff = teamcard.sort_values(by="TotalGames",ascending=False)
ax = sns.barplot(x=dff["Teamname"],y=dff["TotalGames"])
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.title("Total Overall Games Played")


#%% plot overall wins

winners = df.groupby("Winner")["Team 1"].count().sort_values(ascending=False)
ax2 = sns.barplot(x=winners.index, y = winners)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=40, ha="right")
plt.title("Overall Wins")
#%% plot the top three countries with most wins
winners = df.groupby("Winner")["Team 1"].count().sort_values(ascending=False)
ax2 = sns.barplot(x=winners[0:3].index, y = winners[0:3])
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=40, ha="right")
plt.title("Top 3 countries with most wins")

#%% plot top three countries with margin by runs
top_3 = df[df["MarginByRuns"]>0].groupby("Winner")["Winner"].count().sort_values(ascending=False)
top_wins_by_runs = df[df["MarginByRuns"]>0].sort_values(by="MarginByRuns",ascending=False)["Winner"][0:3]
sns.barplot(x=df[df["MarginByRuns"]>0].sort_values(by="MarginByRuns",ascending=False)["Winner"][0:3], y=df[df["MarginByRuns"]>0].sort_values(by="MarginByRuns",ascending=False)["MarginByRuns"][0:3])
plt.title("Top 3 countries with margin by runs")
