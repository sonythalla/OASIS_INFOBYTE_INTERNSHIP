"""
Internship Portfolio: Machine Learning Projects
Author: [Your Name]
Date: 2026-03-01
Description:
This script includes three machine learning projects:
1. Iris Flower Classification
2. Car Price Prediction
3. Sales Prediction
Each project includes data loading, exploration, model training, evaluation, 
and visualization for internship submission.
"""

# ------------------------------
# 1. IRIS FLOWER CLASSIFICATION
# ------------------------------
print("\n================ IRIS FLOWER CLASSIFICATION ================\n")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
feature_names_iris = iris.feature_names
target_names_iris = iris.target_names

# Split data
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42
)

# Train Random Forest Classifier
model_iris = RandomForestClassifier(n_estimators=100, random_state=42)
model_iris.fit(X_train_iris, y_train_iris)

# Evaluate
y_pred_iris = model_iris.predict(X_test_iris)
print(f"Accuracy: {accuracy_score(y_test_iris, y_pred_iris):.2f}")
print("\nClassification Report:")
print(classification_report(y_test_iris, y_pred_iris, target_names=target_names_iris))

# Feature importance
importances_iris = model_iris.feature_importances_
indices_iris = np.argsort(importances_iris)[::-1]
plt.figure(figsize=(8,5))
plt.bar(range(X_iris.shape[1]), importances_iris[indices_iris], color='skyblue', align='center')
plt.xticks(range(X_iris.shape[1]), [feature_names_iris[i] for i in indices_iris], rotation=45)
plt.ylabel('Importance')
plt.title('Iris Feature Importance')
plt.show()

# ------------------------------
# 2. CAR PRICE PREDICTION
# ------------------------------
print("\n================ CAR PRICE PREDICTION ================\n")

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
car_data = pd.read_csv('car_prices.csv')  # Replace with your dataset path
print(car_data.head())

# Preprocess (one-hot encode categorical columns)
categorical_cols = car_data.select_dtypes(include=['object']).columns
car_data = pd.get_dummies(car_data, columns=categorical_cols, drop_first=True)

# Split features and target
X_car = car_data.drop('Price', axis=1)
y_car = car_data['Price']
X_train_car, X_test_car, y_train_car, y_test_car = train_test_split(
    X_car, y_car, test_size=0.2, random_state=42
)

# Train Random Forest Regressor
model_car = RandomForestRegressor(n_estimators=100, random_state=42)
model_car.fit(X_train_car, y_train_car)

# Evaluate
y_pred_car = model_car.predict(X_test_car)
print(f"Mean Squared Error: {mean_squared_error(y_test_car, y_pred_car):.2f}")
print(f"R2 Score: {r2_score(y_test_car, y_pred_car):.2f}")

# Feature importance
importances_car = model_car.feature_importances_
indices_car = np.argsort(importances_car)[::-1]
plt.figure(figsize=(10,6))
plt.bar(range(X_car.shape[1]), importances_car[indices_car], color='skyblue', align='center')
plt.xticks(range(X_car.shape[1]), [X_car.columns[i] for i in indices_car], rotation=45)
plt.ylabel('Importance')
plt.title('Car Price Feature Importance')
plt.tight_layout()
plt.show()

# Predicted vs Actual
plt.figure(figsize=(8,6))
plt.scatter(y_test_car, y_pred_car, alpha=0.7, color='purple')
plt.plot([y_test_car.min(), y_test_car.max()], [y_test_car.min(), y_test_car.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predicted vs Actual Car Prices')
plt.grid(True)
plt.show()

# ------------------------------
# 3. SALES PREDICTION
# ------------------------------
print("\n================ SALES PREDICTION ================\n")

# Load dataset
sales_data = pd.read_csv('sales_data.csv')  # Replace with your dataset path
print(sales_data.head())

# Split features and target
X_sales = sales_data.drop('Sales', axis=1)
y_sales = sales_data['Sales']
X_train_sales, X_test_sales, y_train_sales, y_test_sales = train_test_split(
    X_sales, y_sales, test_size=0.2, random_state=42
)

# Train Random Forest Regressor
model_sales = RandomForestRegressor(n_estimators=100, random_state=42)
model_sales.fit(X_train_sales, y_train_sales)

# Evaluate
y_pred_sales = model_sales.predict(X_test_sales)
print(f"Mean Squared Error: {mean_squared_error(y_test_sales, y_pred_sales):.2f}")
print(f"R2 Score: {r2_score(y_test_sales, y_pred_sales):.2f}")

# Feature importance
importances_sales = model_sales.feature_importances_
indices_sales = np.argsort(importances_sales)[::-1]
plt.figure(figsize=(8,5))
plt.bar(range(X_sales.shape[1]), importances_sales[indices_sales], color='skyblue', align='center')
plt.xticks(range(X_sales.shape[1]), [X_sales.columns[i] for i in indices_sales], rotation=45)
plt.ylabel('Importance')
plt.title('Sales Feature Importance')
plt.tight_layout()
plt.show()

# Predicted vs Actual
plt.figure(figsize=(8,6))
plt.scatter(y_test_sales, y_pred_sales, alpha=0.7, color='green')
plt.plot([y_test_sales.min(), y_test_sales.max()], [y_test_sales.min(), y_test_sales.max()], 'r--')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Predicted vs Actual Sales')
plt.grid(True)
plt.show()
