"""
Air Quality in Indian Cities (2015-2020)
BDS3106 Programming for Data Science - Mid-Term Project (Phase I)

How to run:   python air_quality_analysis.py
Before that:  put city_day.csv (from Kaggle) inside data/raw/
Every step prints what it is doing, so you can explain it in the viva.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# 0. SETTINGS
# ------------------------------------------------------------------
DATA_FILE = "data/raw/city_day.csv"
os.makedirs("figures", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
sns.set_style("whitegrid")

POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2",
              "O3", "Benzene", "Toluene", "Xylene"]

# ------------------------------------------------------------------
# 1. LOAD THE DATA
# ------------------------------------------------------------------
print("\n===== STEP 1: LOADING DATA =====")
df = pd.read_csv(DATA_FILE)
print("Loaded:", DATA_FILE)

# ------------------------------------------------------------------
# 2. DATA UNDERSTANDING
# ------------------------------------------------------------------
print("\n===== STEP 2: DATA UNDERSTANDING =====")
print("Rows, columns:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names and data types:")
print(df.dtypes)
print("\nSummary statistics:")
print(df.describe().round(2))

missing_pct = (df.isnull().mean() * 100).round(1).sort_values(ascending=False)
print("\nMissing values (% of rows) per column:")
print(missing_pct)
print("\nDuplicate rows:", df.duplicated().sum())
print("Cities:", df["City"].nunique(), "->", sorted(df["City"].unique()))

# keep the raw copy for comparison and the missing-value chart
df_raw = df.copy()

plt.figure(figsize=(8, 5))
sns.barplot(x=missing_pct.values, y=missing_pct.index, color="indianred")
plt.title("Missing values per column (raw data)")
plt.xlabel("% of rows missing")
plt.tight_layout()
plt.savefig("figures/fig0_missing_values.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 3. DATA VALIDATION AND PREPROCESSING
# ------------------------------------------------------------------
print("\n===== STEP 3: PREPROCESSING =====")

# 3.1 check that the columns we need are present
needed = ["City", "Date", "AQI"] + POLLUTANTS
absent = [c for c in needed if c not in df.columns]
if absent:
    print("WARNING: these expected columns are not in your file:", absent)
    POLLUTANTS = [c for c in POLLUTANTS if c in df.columns]

# 3.2 tidy text
df.columns = df.columns.str.strip()
df["City"] = df["City"].astype(str).str.strip().str.title()

# 3.3 dates: text -> real dates; unreadable ones become empty
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
print("Rows with unreadable/missing Date:", df["Date"].isnull().sum())
df = df.dropna(subset=["Date"])

# 3.4 numbers: make sure every pollutant column is numeric
for col in POLLUTANTS + ["AQI"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 3.5 duplicates: same city on the same day should appear once
before = len(df)
df = df.drop_duplicates(subset=["City", "Date"])
print("Duplicate city-day rows removed:", before - len(df))

# 3.6 impossible values: a pollution reading cannot be negative
neg = (df[POLLUTANTS + ["AQI"]] < 0).sum().sum()
print("Negative readings turned into missing:", neg)
for col in POLLUTANTS + ["AQI"]:
    df.loc[df[col] < 0, col] = np.nan

# 3.7 missing values: decision
#     - columns with more than 50% missing are too empty to trust -> not used
#     - other missing values are left empty (NOT invented). pandas ignores them
#       when it calculates averages, so no fake numbers enter the analysis.
too_empty = [c for c in POLLUTANTS if df[c].isnull().mean() > 0.5]
print("Columns with more than 50% missing (not used in correlations):", too_empty)
usable = [c for c in POLLUTANTS if c not in too_empty]
print("Rows with no AQI value:", df["AQI"].isnull().sum(), "(kept, ignored in AQI results)")

# 3.8 outliers: flag them, but keep them (real pollution episodes, e.g. Delhi winter)
q1, q3 = df["PM2.5"].quantile(0.25), df["PM2.5"].quantile(0.75)
upper = q3 + 1.5 * (q3 - q1)
print("PM2.5 above %.0f (IQR rule): %d rows -> kept. Maximum PM2.5 = %.0f" %
      (upper, (df["PM2.5"] > upper).sum(), df["PM2.5"].max()))

# 3.9 AQI category using the official Indian (CPCB) AQI bands
bands = [-1, 50, 100, 200, 300, 400, 10000]
labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
df["AQI_Category"] = pd.cut(df["AQI"], bins=bands, labels=labels)
if "AQI_Bucket" in df.columns:
    same = (df["AQI_Bucket"].astype(str).str.strip() == df["AQI_Category"].astype(str))
    print("Rows where the file's AQI_Bucket matches the CPCB band:", same.sum(), "of", df["AQI"].notnull().sum())

# 3.10 new time columns
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
season_map = {12: "Winter", 1: "Winter", 2: "Winter",
              3: "Summer", 4: "Summer", 5: "Summer",
              6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
              10: "Post-monsoon", 11: "Post-monsoon"}
df["Season"] = df["Month"].map(season_map)

print("\nRows before cleaning:", len(df_raw), "| after cleaning:", len(df))
print("Date range:", df["Date"].min().date(), "to", df["Date"].max().date())
df.to_csv("data/processed/city_day_clean.csv", index=False)

# ------------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------------
print("\n===== STEP 4: EDA =====")

# Coverage: cities were not monitored for the same number of days
cover = df.groupby("City")["Date"].agg(["count", "min", "max"])
print("\nDays of data per city (coverage):\n", cover.sort_values("count"))

# Q1. Which cities have the worst air?
city = df.groupby("City").agg(Mean_AQI=("AQI", "mean"),
                              Mean_PM25=("PM2.5", "mean"),
                              Days=("AQI", "count")).sort_values("Mean_AQI", ascending=False)
poor_share = (df[df["AQI"].notnull()].assign(poor=lambda x: x["AQI"] > 200)
              .groupby("City")["poor"].mean() * 100)
city["Poor_or_worse_days_%"] = poor_share
print("\nQ1 Cities ranked by average AQI:\n", city.round(1))

plt.figure(figsize=(8, 6))
sns.barplot(x=city["Mean_AQI"], y=city.index, color="steelblue")
plt.title("Average AQI by city (2015-2020)")
plt.xlabel("Average AQI"); plt.ylabel("")
plt.tight_layout()
plt.savefig("figures/fig1_city_aqi.png", dpi=150); plt.close()

# Q2. Which months and seasons are the most polluted?
by_month = df.groupby("Month")["PM2.5"].mean()
by_season = df.groupby("Season")["PM2.5"].mean().reindex(["Winter", "Summer", "Monsoon", "Post-monsoon"])
print("\nQ2 Average PM2.5 by month:\n", by_month.round(1))
print("\nQ2 Average PM2.5 by season:\n", by_season.round(1))

plt.figure(figsize=(8, 4.5))
plt.plot(by_month.index, by_month.values, marker="o", color="darkred")
plt.xticks(range(1, 13))
plt.title("Average PM2.5 by month (all cities)")
plt.xlabel("Month (1 = Jan)"); plt.ylabel("PM2.5 (ug/m3)")
plt.tight_layout()
plt.savefig("figures/fig2_monthly_pm25.png", dpi=150); plt.close()

plt.figure(figsize=(6, 4.5))
sns.barplot(x=by_season.index, y=by_season.values, color="darkorange")
plt.title("Average PM2.5 by season")
plt.xlabel(""); plt.ylabel("PM2.5 (ug/m3)")
plt.tight_layout()
plt.savefig("figures/fig2b_season_pm25.png", dpi=150); plt.close()

# Q3. Did air quality change from 2015 to 2020?
by_year = df.groupby("Year")["AQI"].mean()
print("\nQ3 Average AQI by year (all cities):\n", by_year.round(1))
plt.figure(figsize=(8, 4.5))
plt.plot(by_year.index, by_year.values, marker="o", label="All cities")
if "Delhi" in df["City"].values:
    delhi = df[df["City"] == "Delhi"].groupby("Year")["AQI"].mean()
    print("\nQ3 Delhi only:\n", delhi.round(1))
    plt.plot(delhi.index, delhi.values, marker="s", linestyle="--", label="Delhi only")
plt.title("Average AQI by year")
plt.xlabel("Year"); plt.ylabel("Average AQI"); plt.legend()
plt.tight_layout()
plt.savefig("figures/fig3_yearly_aqi.png", dpi=150); plt.close()
print("NOTE: the set of cities changes over the years, so use the Delhi line to check the trend.")

# Q4. Which pollutants move together with AQI?
corr = df[usable + ["AQI"]].corr(method="spearman")["AQI"].drop("AQI").sort_values(ascending=False)
print("\nQ4 Spearman correlation with AQI:\n", corr.round(2))
plt.figure(figsize=(7, 4.5))
sns.barplot(x=corr.values, y=corr.index, color="seagreen")
plt.title("Correlation of each pollutant with AQI (Spearman)")
plt.xlabel("Correlation"); plt.ylabel("")
plt.tight_layout()
plt.savefig("figures/fig4_pollutant_corr.png", dpi=150); plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(df[usable + ["AQI"]].corr(method="spearman"), annot=True, fmt=".1f", cmap="coolwarm")
plt.title("Correlation heatmap")
plt.tight_layout()
plt.savefig("figures/fig4b_heatmap.png", dpi=150); plt.close()

# Q5. Did PM2.5 fall during the 2020 COVID-19 lockdown?
# India's national lockdown began on 25 March 2020. Compare 25 Mar - 30 Apr
# of 2020 with the same dates of 2019, city by city.
def period(year):
    mask = (df["Date"] >= "%d-03-25" % year) & (df["Date"] <= "%d-04-30" % year)
    return df[mask].groupby("City")["PM2.5"].mean()

p19, p20 = period(2019), period(2020)
covid = pd.DataFrame({"PM25_2019": p19, "PM25_2020": p20}).dropna()
covid["Change_%"] = (covid["PM25_2020"] - covid["PM25_2019"]) / covid["PM25_2019"] * 100
covid = covid.sort_values("Change_%")
print("\nQ5 PM2.5, 25 Mar-30 Apr: 2019 vs 2020 (lockdown)\n", covid.round(1))
if len(covid) > 0:
    print("Average change across cities: %.1f%%" % covid["Change_%"].mean())
    plt.figure(figsize=(8, 6))
    sns.barplot(x=covid["Change_%"], y=covid.index, color="slateblue")
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Change in PM2.5, 25 Mar-30 Apr 2020 vs 2019")
    plt.xlabel("% change"); plt.ylabel("")
    plt.tight_layout()
    plt.savefig("figures/fig5_covid_change.png", dpi=150); plt.close()

# Extra: share of days in each AQI category
cat = df["AQI_Category"].value_counts(normalize=True).reindex(labels) * 100
print("\nShare of city-days by AQI category (%):\n", cat.round(1))
plt.figure(figsize=(7, 4.5))
sns.barplot(x=cat.index, y=cat.values, color="goldenrod")
plt.title("Share of days in each AQI category")
plt.xlabel(""); plt.ylabel("% of city-days")
plt.xticks(rotation=20); plt.tight_layout()
plt.savefig("figures/fig6_aqi_categories.png", dpi=150); plt.close()

print("\nDone. Charts are in /figures, clean data is in /data/processed.")
