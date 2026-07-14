import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["Species"] = iris.target

df["Species"] = df["Species"].map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

print(df.head())
print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nSpecies Count:")
print(df["Species"].value_counts())
print("\n========== DATASET INFORMATION ==========\n")

print("Shape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nSpecies Distribution:")
print(df["Species"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x="Species", data=df)
plt.title("Species Distribution")
plt.show()
sns.pairplot(df, hue="Species")
plt.show()
features = df.columns[:-1]

for feature in features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x="Species", y=feature, data=df)
    plt.title(feature)
    plt.show()
plt.figure(figsize=(8,6))
sns.heatmap(df.iloc[:, :-1].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
df.hist(figsize=(10,8))
plt.tight_layout()
plt.show()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
lr = LogisticRegression()

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, lr_pred))

print("\nClassification Report")
print(classification_report(y_test, lr_pred, target_names=iris.target_names))
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

print("KNN Accuracy:", accuracy_score(y_test, knn_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, knn_pred))

print("\nClassification Report")
print(classification_report(y_test, knn_pred, target_names=iris.target_names))
tree = DecisionTreeClassifier(random_state=42)

tree.fit(X_train, y_train)

tree_pred = tree.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, tree_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, tree_pred))

print("\nClassification Report")
print(classification_report(y_test, tree_pred, target_names=iris.target_names))
result = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "KNN",
        "Decision Tree"
    ],
    "Accuracy": [
        accuracy_score(y_test, lr_pred),
        accuracy_score(y_test, knn_pred),
        accuracy_score(y_test, tree_pred)
    ]
})

print(result)
plt.figure(figsize=(8,5))

sns.barplot(
    data=result,
    x="Model",
    y="Accuracy",
    palette="viridis"
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy")

plt.show()
best_model = result.sort_values(
    by="Accuracy",
    ascending=False
)

print("Best Performing Model")
print(best_model)
sample = [[5.1, 3.5, 1.4, 0.2]]

sample = scaler.transform(sample)

prediction = knn.predict(sample)

print("Predicted Flower Species:")
print("""
================ PROJECT CONCLUSION ================

1. Successfully loaded the Iris dataset.
2. Performed Exploratory Data Analysis (EDA).
3. Visualized the data using different plots.
4. Split the dataset into training and testing sets.
5. Applied Feature Scaling.
6. Trained three Machine Learning models:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Decision Tree
7. Evaluated each model using:
   - Accuracy
   - Confusion Matrix
   - Classification Report
8. Compared all models and selected the best-performing model.
9. Successfully predicted the species of a new Iris flower.

This project demonstrates the complete Machine Learning workflow using Python and Scikit-learn.
""")
print(iris.target_names[prediction][0])
