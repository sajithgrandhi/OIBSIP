import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# Create screenshots folder automatically
# ==========================================

os.makedirs("screenshots", exist_ok=True)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("CarPrice_Assignment.csv")

print("\n========== FIRST 5 ROWS ==========\n")
print(df.head())

print("\n========== DATASET SHAPE ==========\n")
print(df.shape)

print("\n========== COLUMN NAMES ==========\n")
print(df.columns)

print("\n========== DATA TYPES ==========\n")
print(df.dtypes)

print("\n========== DATASET INFORMATION ==========\n")
df.info()

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========\n")
print(df.duplicated().sum())

print("\n========== STATISTICAL SUMMARY ==========\n")
print(df.describe())

# ==========================================
# Remove Car_ID
# ==========================================

if "car_ID" in df.columns:
    df.drop("car_ID", axis=1, inplace=True)

# ==========================================
# Encode Categorical Columns
# ==========================================

label_encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == "object":
        df[column] = label_encoder.fit_transform(df[column])

print("\n========== ENCODED DATA ==========\n")
print(df.head())

# ==========================================
# Correlation Heatmap
# ==========================================

plt.figure(figsize=(14,10))

sns.heatmap(
    df.corr(),
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("screenshots/heatmap.png")

plt.show()

# ==========================================
# Price Distribution
# ==========================================

plt.figure(figsize=(8,5))

sns.histplot(
    df["price"],
    bins=30,
    kde=True
)

plt.title("Car Price Distribution")

plt.tight_layout()

plt.savefig("screenshots/price_distribution.png")

plt.show()

# ==========================================
# Top 10 Expensive Cars
# ==========================================

top10 = df.sort_values(
    by="price",
    ascending=False
).head(10)

plt.figure(figsize=(10,6))

sns.barplot(
    x=top10.index.astype(str),
    y=top10["price"]
)

plt.title("Top 10 Expensive Cars")
plt.xlabel("Car Index")
plt.ylabel("Price")

plt.tight_layout()

plt.savefig("screenshots/top10_expensive_cars.png")

plt.show()



# ==========================================
# Prepare Features and Target
# ==========================================

X = df.drop("price", axis=1)
y = df["price"]

# ==========================================
# Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

# ==========================================
# Train Linear Regression Model
# ==========================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ==========================================
# Prediction
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Evaluation
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")

print("Mean Absolute Error :", mae)
print("Mean Squared Error  :", mse)
print("Root Mean Squared Error :", rmse)
print("R2 Score :", r2)

# ==========================================
# Actual vs Predicted
# ==========================================

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nFirst 10 Predictions")
print(comparison.head(10))

# ==========================================
# Scatter Plot
# ==========================================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted Car Prices")

plt.tight_layout()

plt.savefig("screenshots/actual_vs_predicted.png")

plt.show()

# ==========================================
# Prediction Error Distribution
# ==========================================

errors = y_test - y_pred

plt.figure(figsize=(8,5))

sns.histplot(
    errors,
    bins=30,
    kde=True
)

plt.title("Prediction Error Distribution")

plt.tight_layout()

plt.savefig("screenshots/error_distribution.png")

plt.show()

# ==========================================
# Feature Importance
# ==========================================

importance = pd.Series(
    model.coef_,
    index=X.columns
)

importance = importance.sort_values()

plt.figure(figsize=(10,8))

importance.plot(kind="barh")

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("screenshots/feature_importance.png")

plt.show()

# ==========================================
# Sample Prediction
# ==========================================

sample = X.iloc[[0]]

prediction = model.predict(sample)

print("\nSample Car Predicted Price:")
print(prediction[0])

# ==========================================
# Project Conclusion
# ==========================================

print("""
================ PROJECT CONCLUSION ================

1. Loaded the Car Price dataset.
2. Cleaned and preprocessed the data.
3. Encoded categorical variables.
4. Performed Exploratory Data Analysis (EDA).
5. Generated and saved multiple visualizations.
6. Split the dataset into training and testing sets.
7. Trained a Linear Regression model.
8. Evaluated the model using:
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - R² Score
9. Predicted car prices successfully.

Project Completed Successfully.
""")
