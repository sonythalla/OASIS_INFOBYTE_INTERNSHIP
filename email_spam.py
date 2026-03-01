"""
Car Price Prediction - Internship Submission
Author: [Your Name]
Date: 2026-03-01
Description:
This script trains a Random Forest Regressor to predict car prices based on features
like brand, mileage, horsepower, and more. It includes feature importance visualization
and predicted vs actual price comparison for evaluation.
"""

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Step 1: Load dataset
data = pd.read_csv('car_prices.csv')  # Replace with your dataset path
print("First 5 rows of dataset:")
print(data.head())

# Step 2: Explore dataset
print("\nDataset info:")
print(data.info())
print("\nMissing values per column:")
print(data.isnull().sum())
print("\nSummary statistics:")
print(data.describe())

# Optional: Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

# Step 3: Preprocess dataset (one-hot encoding for categorical columns)
categorical_cols = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Step 4: Split data into features and target
X = data.drop('Price', axis=1)
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance on Test Set:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")

# Step 7: Feature Importance Visualization
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.bar(range(X.shape[1]), importances[indices], color='skyblue', align='center')
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=45)
plt.ylabel('Importance')
plt.title('Feature Importance - Random Forest Regressor')
plt.tight_layout()
plt.show()

# Step 8: Predicted vs Actual Price Comparison
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predicted vs Actual Car Prices')
plt.grid(True)
plt.show()

# Step 9: Predict prices for new cars (example)
# Replace values with actual feature vectors after preprocessing
new_cars = [
    [150, 1200, 15, 1, 0, 0],  # Example car 1
    [200, 1600, 12, 0, 1, 0]   # Example car 2
]
predicted_prices = model.predict(new_cars)
print("\nPredicted prices for new cars:", predicted_prices)
