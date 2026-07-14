import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nUnique Regions:")
print(df["Region"].unique())
import os

# Create screenshots folder automatically
os.makedirs("screenshots", exist_ok=True)

# -----------------------------
# Data Cleaning
# -----------------------------

print("\nRemoving Extra Spaces from Column Names...")
df.columns = df.columns.str.strip()

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

# Remove missing values
df = df.dropna()

print("\nDataset Shape After Cleaning:")
print(df.shape)

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

print("\nData Types After Conversion:")
print(df.dtypes)

# -----------------------------
# Exploratory Data Analysis
# -----------------------------

print("\nAverage Unemployment Rate by Region:")
region_unemployment = df.groupby("Region")["Estimated Unemployment Rate (%)"].mean()
print(region_unemployment.sort_values(ascending=False))

# -----------------------------
# Bar Plot
# -----------------------------

plt.figure(figsize=(12,6))

sns.barplot(
    x=region_unemployment.index,
    y=region_unemployment.values,
    palette="viridis"
)

plt.xticks(rotation=90)
plt.title("Average Unemployment Rate by Region")
plt.xlabel("Region")
plt.ylabel("Average Unemployment Rate (%)")

plt.tight_layout()

plt.savefig("screenshots/region_unemployment.png")

plt.show()

# -----------------------------
# Line Plot
# -----------------------------

plt.figure(figsize=(12,6))

sns.lineplot(
    data=df,
    x="Date",
    y="Estimated Unemployment Rate (%)"
)

plt.title("Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")

plt.tight_layout()

plt.savefig("screenshots/unemployment_trend.png")

plt.show()

# -----------------------------
# Correlation Heatmap
# -----------------------------

numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(8,6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("screenshots/heatmap.png")

plt.show()

# -----------------------------
# Labour Participation Rate
# -----------------------------

plt.figure(figsize=(12,6))

sns.barplot(
    data=df,
    x="Region",
    y="Estimated Labour Participation Rate (%)",
    palette="Set2"
)

plt.xticks(rotation=90)

plt.title("Labour Participation Rate by Region")

plt.tight_layout()

plt.savefig("screenshots/labour_participation.png")

plt.show()

# -----------------------------
# Scatter Plot
# -----------------------------

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df,
    x="Estimated Employed",
    y="Estimated Unemployment Rate (%)",
    hue="Region"
)

plt.title("Employment vs Unemployment")

plt.tight_layout()

plt.savefig("screenshots/scatter_plot.png")

plt.show()

# -----------------------------
# Conclusion
# -----------------------------

print("""
================ PROJECT CONCLUSION ================

1. Loaded the unemployment dataset.
2. Cleaned the dataset.
3. Performed Exploratory Data Analysis (EDA).
4. Visualized unemployment trends.
5. Compared unemployment across regions.
6. Analyzed labour participation rate.
7. Generated and saved all graphs automatically.
8. Identified unemployment patterns in India.

Project Completed Successfully.
""")
