
# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score
from matplotlib.colors import Normalize, ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

# For Suppressing warnings
import warnings
warnings.filterwarnings('ignore')

# Loading dataset
data = pd.read_csv('C:/Users/Faraz Yusuf Khan/Desktop/Musa/Bank_Churn/train.csv')

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Creating a copy of the data
data_copy = data.copy()

# Check if the copy was successful
print("Original Data Shape:", data.shape)
print("Copied Data Shape:", data_copy.shape)

# Drop the third column
data_copy = data_copy.drop(data_copy.columns[[2]], axis=1)

# Display the updated DataFrame
print("Updated Data Shape:", data_copy.shape)
print("Updated Data with Dropped Columns:\n", data_copy.head())

# Assuming 'Geography' and 'Gender' are categorical columns
categorical_columns = ['Geography', 'Gender']

# One-hot encode categorical columns
data_copy = pd.get_dummies(data_copy, columns=categorical_columns, drop_first=True)

# Display the updated DataFrame
print("Updated Data Shape:", data_copy.shape)
print("Updated Data with Dropped Columns and One-Hot Encoding:\n", data_copy.head())

# Assuming the target variable is in the 'Exited' column
target_column = 'Exited'

# Separate features (X) and target variable (y)
X = data_copy.drop(target_column, axis=1)
y = data_copy[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the resulting sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Display the names of columns in X_train
print("Column names in X_train:", X_train.columns.tolist())

# Initialize a RandomForestClassifier for hyperparameter tuning
classifier = RandomForestClassifier(random_state=42)

# Define the parameter distribution for hyperparameter tuning
param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 8, 10],
    'min_samples_leaf': [1, 2, 3, 4]
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(classifier, param_dist, n_iter=20, cv=5, scoring='roc_auc', n_jobs=-1, random_state=42)

# Fit RandomizedSearchCV to the data
random_search.fit(X_train, y_train)

# Print the best parameters found by RandomizedSearchCV
print("Best Hyperparameters:", random_search.best_params_)

# Get the best model from the random search
best_classifier = random_search.best_estimator_

# Train the best classifier
best_classifier.fit(X_train, y_train)

# Predict probabilities on the test set using the best model
y_pred_prob = best_classifier.predict_proba(X_test)[:, 1]

# Evaluate the model using AUC-ROC score
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"AUC-ROC Score: {roc_auc}")

# Load the test dataset
test_data = pd.read_csv('C:/Users/Faraz Yusuf Khan/Desktop/Musa/Bank_Churn/test.csv')

# Check for missing values in the test dataset
print("Missing values in test dataset:\n", test_data.isnull().sum())

# Create a copy of the test data
test_data_copy = test_data.copy()

# Drop the same columns as you did in the training data
test_data_copy = test_data_copy.drop(test_data_copy.columns[[2]], axis=1)

# One-hot encode categorical columns
test_data_copy = pd.get_dummies(test_data_copy, columns=categorical_columns, drop_first=True)

# Display the updated test DataFrame
print("Updated Test Data Shape:", test_data_copy.shape)
print("Updated Test Data with Dropped Columns and One-Hot Encoding:\n", test_data_copy.head())

# Separate features for the test set
X_test_submission = test_data_copy

# Predict probabilities on the test set
y_test_submission_prob = best_classifier.predict_proba(X_test_submission)[:, 1]

# Create a DataFrame for submission
submission_df = pd.DataFrame({
    'id': test_data['id'],  # Assuming 'id' is the identifier column in the test set
    'Exited': y_test_submission_prob
})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("We are here")
