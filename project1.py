# HUSSEIN MOHAMMAD - 501098569
# AER850 Section 01 Project 1


'''STEP 1 - DATA PROCESSING'''
import pandas as pd

# Reading the dataset
df = pd.read_csv("Project_1_Data.csv")

# Checking the first few rows of the dataset
print(df.head())

# Splitting data into features (X) and target variable (y)
X = df[['X', 'Y', 'Z']]  # Features: X, Y, Z coordinates
y = df['Step']           # Target: Step (maintenance step)

# Check for class imbalance
print(f"\nClass Distribution: \n{y.value_counts(normalize=True)}")


'''STEP 2 - SPLITTING DATA INTO TRAIN AND TEST SETS'''
from sklearn.model_selection import StratifiedShuffleSplit

# Splitting Data into Train and Test Sets
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
for train_index, test_index in sss.split(X, y):
    strat_df_train = df.loc[train_index].reset_index(drop=True)
    strat_df_test = df.loc[test_index].reset_index(drop=True)
    
X_train = strat_df_train.drop("Step", axis = 1)
y_train = strat_df_train["Step"]
X_test = strat_df_test.drop("Step", axis = 1)
y_test = strat_df_test["Step"]


'''STEP 3 - DATA VISUALIZATION'''
import matplotlib.pyplot as plt

# Creating a new figure with a specified size, and adding a 3D subplot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(projection='3d')

# Creating a 3D scatter plot using the 'X', 'Y', and 'Z' columns from the DataFrame (df)
# The points are colored based on the values in the 'Step' column.
scatter = ax.scatter(df['X'], df['Y'], df['Z'], c=df['Step'], cmap='plasma', vmin=1, vmax=12)

# Setting the labels for the axis to X, Y and Z
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Adding a colorbar to the plot
plt.colorbar(scatter, label='Step')
plt.title('3D Scatter plot of X, Y, Z colored by Step')
plt.show()


'''STEP 4 - CORRELATION ANALYSIS'''
import seaborn as sns
# Compute the correlation matrix
# The correlation matrix helps us understand how features like X, Y, Z are related to the target (Step)
correlation_matrix = df[['X', 'Y', 'Z', 'Step']].corr()

# Display the correlation matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, linewidths=0.5)
plt.title("Correlation Matrix of Features and Target")
plt.show()


'''STEP 5 - DATA CLEANING & PREPROCESSING'''
from sklearn.preprocessing import StandardScaler
# Checking for missing values
print(df.isnull().sum())  # Ensure no missing values

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


'''STEP 6 - MODEL TRAINING'''
# We will train 4 models: Logistic Regression, Support Vector Classifier (SVC), Random Forest, and Decision Tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

# Model 1: Logistic Regression with Grid Search CV
log_reg = LogisticRegression(max_iter=1000)
param_grid_log_reg = {'C': [0.1, 1, 10]}
grid_log_reg = GridSearchCV(log_reg, param_grid_log_reg, cv=5)  
grid_log_reg.fit(X_train_scaled, y_train)
print(f"Best Logistic Regression Parameters: {grid_log_reg.best_params_}")

# Model 2: Support Vector Classifier (SVC) with Grid Search CV
svc = SVC()
param_grid_svc = {'C': [0.1, 1, 10]}
grid_svc = GridSearchCV(svc, param_grid_svc, cv=5)  
grid_svc.fit(X_train_scaled, y_train)
print(f"Best SVC Parameters: {grid_svc.best_params_}")

# Model 3: Random Forest Classifier with Grid Search CV
rf = RandomForestClassifier()
param_dist_rf = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf':[1, 2, 4]}
grid_rf = GridSearchCV(rf, param_dist_rf, cv=5)
grid_rf.fit(X_train_scaled, y_train)
print(f"Best Random Forest Parameters: {grid_rf.best_params_}")

# Model 4: Decision Tree Classifier with Randomized Search CV
dt = DecisionTreeClassifier()
param_dist_dt = {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
random_search_dt = RandomizedSearchCV(dt, param_distributions=param_dist_dt, n_iter=10, cv=5)
random_search_dt.fit(X_train_scaled, y_train)
print(f"Best Decision Tree Parameters: {random_search_dt.best_params_}")


'''STEP 7 - MODEL PERFORMANCE ANALYSIS WITH RECALL'''
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Compare performance (Accuracy, Precision, Recall, F1) for all models
models = {
    'Logistic Regression': grid_log_reg,
    'SVC': grid_svc,
    'Random Forest': grid_rf,
    'Decision Tree': random_search_dt
}

best_model = None
best_f1_score = 0

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Check if this is the best model based on F1 score
    if f1 > best_f1_score:
        best_f1_score = f1
        best_model = model

# Display confusion matrix for the best model
print(f"\nBest Model (based on F1 score): {best_model.estimator.__class__.__name__}")
y_pred_best = best_model.predict(X_test_scaled)
conf_matrix_best = confusion_matrix(y_test, y_pred_best)
ConfusionMatrixDisplay(conf_matrix_best).plot()
plt.title(f"Confusion Matrix for Best Model: {best_model.estimator.__class__.__name__}")
plt.show()


'''STEP 8 - STACKED MODEL PERFORMANCE ANALYSIS'''
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Define base estimators for stacking (using RandomForest and SVC as base models)
base_estimators = [
    ('random_forest', grid_rf.best_estimator_),
    ('svc', grid_svc.best_estimator_)
]

# Stacking with Logistic Regression as final estimator
stacking_clf = StackingClassifier(estimators=base_estimators, final_estimator=LogisticRegression(max_iter=1000), cv=5)

# Train the stacked model
stacking_clf.fit(X_train_scaled, y_train)

# Predict using stacked model
y_pred_stacked = stacking_clf.predict(X_test_scaled)

# Evaluate stacked model performance
stacked_accuracy = accuracy_score(y_test, y_pred_stacked)
stacked_precision = precision_score(y_test, y_pred_stacked, average='weighted')
stacked_recall = recall_score(y_test, y_pred_stacked, average='weighted')
stacked_f1 = f1_score(y_test, y_pred_stacked, average='weighted')

print(f"\nStacked Model Performance:")
print(f"Accuracy: {stacked_accuracy:.4f}")
print(f"Precision: {stacked_precision:.4f}")
print(f"Recall: {stacked_recall:.4f}")
print(f"F1 Score: {stacked_f1:.4f}")

# Display confusion matrix for stacked model
conf_matrix_stacked = confusion_matrix(y_test, y_pred_stacked)
ConfusionMatrixDisplay(conf_matrix_stacked).plot()
plt.title("Confusion Matrix for Stacked Classifier")
plt.show()


'''STEP 9 - MODEL SAVING AND PREDICTION'''
import joblib
import numpy as np

# Save the best model
joblib.dump(best_model, 'best_model.joblib')
print("Best model saved successfully.")

# Save the stacked classifier
joblib.dump(stacking_clf, 'stacked_classifier_model.joblib')
print("Stacked model saved successfully.")

# Load model (demonstration) and make predictions
loaded_model = joblib.load('stacked_classifier_model.joblib')

# Predict maintenance steps for new coordinates
new_coordinates = np.array([[9.375, 3.0625, 1.51],
                            [6.995, 5.125, 0.3875],
                            [0, 3.0625, 1.93],
                            [9.4, 3, 1.8],
                            [9.4, 3, 1.3]])

# Scale the new data
new_data_scaled = scaler.transform(new_coordinates)

# Predict and display results
predicted_steps = loaded_model.predict(new_data_scaled)
print(f"\nPredicted maintenance steps for new data: {predicted_steps}")

