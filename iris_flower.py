"""
Iris Flower Classification with Feature Importance
Author: [Your Name]
Date: 2026-03-01
Description:
This script trains a Random Forest Classifier to classify Iris flowers
into three species (setosa, versicolor, virginica) using Scikit-learn's built-in dataset.
It also visualizes feature importance to show which measurements are most informative.
"""

# Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Step 2: Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Step 5: Predict new samples
new_samples = [
    [5.1, 3.5, 1.4, 0.2],  # Likely setosa
    [6.0, 2.9, 4.5, 1.5],  # Likely versicolor
    [6.7, 3.0, 5.2, 2.3]   # Likely virginica
]
predicted_species = [target_names[i] for i in model.predict(new_samples)]
print("Predicted species for new samples:", predicted_species)

# Step 6: Visualize feature importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8,5))
plt.title("Feature Importance - Random Forest")
plt.bar(range(X.shape[1]), importances[indices], color='skyblue', align='center')
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
plt.ylabel("Importance")
plt.tight_layout()
plt.show()
