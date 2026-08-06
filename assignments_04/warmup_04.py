import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
    f1_score
)
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ROC and AUC

# Q1
# Train Logistic Regression model using raw, unscaled data
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

logistic_model.fit(X_train, y_train)

# Compute predicted probabilities on the test sets for Logistic Regression model
logistic_probs = logistic_model.predict_proba(X_test)[:, 1]

# Compute and print the AUC score using roc_auc_score for Logistic Regression model
logistic_auc = roc_auc_score(y_test, logistic_probs)
print(f"{'Logistic Regression AUC:':<24}{logistic_auc:>8.4f}")

# Train KNN model using scaled data
knn_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ]
)

knn_model.fit(X_train, y_train)

# Compute predicted probabilities on the test sets for KNN model
knn_probs = knn_model.predict_proba(X_test)[:, 1]

# Compute and print the AUC score using roc_auc_score for Logistic Regression model
knn_auc = roc_auc_score(y_test, knn_probs)
print(f"{'KNN AUC:':<24}{knn_auc:>8.4f}")

# KNN has the higher AUC (0.9394 vs. 0.7060).
# This indicates that KNN separates the two classes much better than
# Logistic Regression.
# A higher AUC means the model is better at telling positive and
# negative examples apart, regardless of the decision threshold.


# Q2
# Compute ROC curve values
logistic_fpr, logistic_tpr, tre = roc_curve(y_test, logistic_probs)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)

# Plot both ROC curves
fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(
    logistic_fpr,
    logistic_tpr,
    label=f"Logistic Regression (AUC = {logistic_auc:.4f})",
)

ax.plot(
    knn_fpr,
    knn_tpr,
    label=f"KNN (AUC = {knn_auc:.4f})",
)

# Random-classifier diagonal
ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier",
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison")
ax.legend()

plt.tight_layout()
plt.savefig("outputs/roc_comparison.png")

# At a TPR of about 0.80, KNN has a much lower FPR than Logistic Regression.
# This means that if we wanted to detect about 80% of the positive cases,
# KNN would produce far fewer false positives.


# Q3
# Compute ROC values
logistic_fpr, logistic_tpr, thresholds = roc_curve(y_test, logistic_probs)

optimum_threshold = None
tpr_at_optimum = None
fpr_at_optimum = None
f1_at_optimum = 0

for threshold, fpr, tpr in zip(thresholds, logistic_fpr, logistic_tpr):

    y_pred = (logistic_probs >= threshold).astype(int)

    f1 = f1_score(y_test, y_pred)

    if f1 > f1_at_optimum:
        optimum_threshold = threshold
        tpr_at_optimum = tpr
        fpr_at_optimum = fpr
        f1_at_optimum = f1

print(f"Optimum Threshold:    {optimum_threshold:.4f}")
print(f"True Positive Rate:   {tpr_at_optimum:.4f}")
print(f"False Positive Rate:  {fpr_at_optimum:.4f}")
print(f"F1 Score:             {f1_at_optimum:.4f}")

# The optimum threshold (0.2757) is lower than the default threshold of 0.5.
# Lowering the threshold increases the number of positive predictions,
# improving recall but also increasing the false positive rate.
# In a real application, a threshold lower than 0.5 is appropriate when
# missing positive cases is more costly than generating extra false positives,
# such as in medical screening or fraud detection.


# GridSearchCV

#Q1
# Build a Pipeline with a StandardScaler and a LogisticRegression
logistic_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=1000)),
    ]
)

# Set C values
logistic_param_grid = {
    "logistic__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

# Grid search using ROC AUC
logistic_grid_search = GridSearchCV(
    estimator=logistic_pipeline,
    param_grid=logistic_param_grid,
    cv=5,
    scoring="roc_auc",
)

logistic_grid_search.fit(X_train, y_train)

# Best estimator's probabilities on the test set
best_logistic_probs = logistic_grid_search.best_estimator_.predict_proba(X_test)[:, 1]

# Test AUC
best_logistic_test_auc = roc_auc_score(
    y_test,
    best_logistic_probs,
)

print(f"{'Best Logistic  C Value:':<30}{logistic_grid_search.best_params_['logistic__C']}")
print(f"{'Best Logistic CV AUC:':<30}{logistic_grid_search.best_score_:.4f}")
print(f"{'Best Logistic Test AUC:':<30}{best_logistic_test_auc:.4f}")

# GridSearchCV selected C = 100.0 rather than the default C = 1.0.
# I would likely have guessed the default value of C = 1.0, 
# so the grid search did not select the same value. 
# The test AUC changed from 0.7060 to 0.7057, a decrease of 0.0003. 
# This difference is extremely small,
# so tuning C did not meaningfully improve performance on the test set.


# Q2
# Build a Pipeline with a DecisionTreeClassifier
decision_tree_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "decision_tree",
            DecisionTreeClassifier(random_state=42),
        ),
    ]
)

# Set C values
decision_tree_param_grid = {
    "decision_tree__max_depth": [2, 3, 5, 8, None]
}

# Grid search using ROC AUC
decision_tree_grid_search = GridSearchCV(
    estimator=decision_tree_pipeline,
    param_grid=decision_tree_param_grid,
    cv=5,
    scoring="roc_auc",
)

decision_tree_grid_search.fit(X_train, y_train)

# Best estimator's probabilities on the test set
best_decision_tree_probs = (
    decision_tree_grid_search.best_estimator_
    .predict_proba(X_test)[:, 1]
)

# Test AUC
best_decision_tree_test_auc = roc_auc_score(
    y_test,
    best_decision_tree_probs,
)

print(
    f"{'Best Decision Tree Max Depth:':<30}{decision_tree_grid_search.best_params_['decision_tree__max_depth']}"
)
print(
    f"{'Best Decision Tree CV AUC:':<30}{decision_tree_grid_search.best_score_:.4f}"
)
print(f"{'Best Test Decision Tree AUC:':<30}{best_decision_tree_test_auc:.4f}")




# Q3
# Create a DataFrame from the grid search results
logistic_cv_results = (
    pd.DataFrame(logistic_grid_search.cv_results_)
    .loc[:, ["param_logistic__C", "mean_test_score", "std_test_score"]]
    .rename(
        columns={
            "param_logistic__C": "C",
            "mean_test_score": "Mean CV AUC",
            "std_test_score": "Std Dev",
        }
    )
    .sort_values(by="Mean CV AUC", ascending=False)
)

# Print sorted values
print(logistic_cv_results)

# C = 100.0 and C = 10.0 produced nearly identical mean CV AUC scores.
# However, C = 100.0 had a slightly lower standard deviation, 
# indicating slightly more consistent performance 
# across the cross-validation folds.
# I would choose C = 100.0 because it achieved the highest mean CV AUC
# while also having lowest standard deviation.


# joblib

# Q1
# Save the best Pipeline from GridSearch Question 1
best_lr_pipe = logistic_grid_search.best_estimator_

joblib.dump(
    best_lr_pipe,
    "models/warmup_model.pkl",
)

# Load the saved pipline
loaded_clf = joblib.load("models/warmup_model.pkl")

# Compare predictions
original_preds = best_lr_pipe.predict(X_test)
loaded_preds   = loaded_clf.predict(X_test)

assert (original_preds == loaded_preds).all(), "Predictions do not match!"
print("Predictions match. Model saved and loaded successfully.")

# The Logistic Regression model was trained on scaled data.
# If only the Logistic Regression model were saved, predictions
# would be made on unscaled data and could be inaccurate.


# Q2
# --- Simulated prediction script ---

# Load the saved pipeline fresh from disk
loaded_clf = joblib.load("models/warmup_model.pkl")

# Three hand-crafted test cases — raw, unscaled data
new_samples = np.array([
    [2.5,  1.2, -0.3,  0.8,  1.0, -0.5,  0.2,  0.9, -1.1,  0.4],
    [-1.0, 0.5,  0.9, -0.7, -0.2,  1.3, -0.8,  0.1,  0.5, -0.3],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])

# Predict classes and positive-class probabilities
new_predictions = loaded_clf.predict(new_samples)
new_probabilities = loaded_clf.predict_proba(new_samples)

for index, (prediction, probabilities) in enumerate(
    zip(new_predictions, new_probabilities),
    start=1,
):
    predicted_probability = probabilities[prediction]

    print(
        f"Sample {index}: "
        f"Predicted Class = {prediction}, "
        f"Positive-Class Probability = {predicted_probability:.4f}"
    )

# I expected the all-zeros row to be predicted as class 1 
# because the model first scales the data before making a prediction.

