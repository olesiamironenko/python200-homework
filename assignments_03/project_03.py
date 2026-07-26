import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# Create the output directory if it doesn't exist
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Task 1
# Load the dataset
COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES

# Explore the data
print("\nDataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print(f"{'Emails Total:':<15}{len(df):>6}")

class_counts = (
    df["spam_label"]
    .value_counts()
    .rename(index={0: "Ham", 1: "Spam"})
)

class_percentages = (
    class_counts
    .mul(100)
    .div(class_counts.sum())
    .round(1)
)

for label in class_counts.index:
    print(f"{label + ':':<15}{class_counts[label]:>6} ({class_percentages[label]:4.1f}%)")

# The classes are reasonably balanced (60.6% ham and 39.4% spam), 
# although ham emails are slightly more common. 
# This class distribution means that accuracy is a useful metric, 
# but it should not be interpreted on its own 
# because a classifier could still achieve 60.6% accuracy 
# by always predicting the majority class (ham). 
# Therefore, additional metrics such as precision, recall, 
# and F1-score are needed to properly evaluate model performance.

# Create boxplots

feature_labels = {
    "word_freq_free": "Word 'Free' Frequency",
    "char_freq_!": "Exclamation Marks (!) Frequency",
    "capital_run_length_total": "Capital-Letter Sequence Length"
}

features = [
    "word_freq_free",
    "char_freq_!",
    "capital_run_length_total"
]

for feature in features:
    plt.figure(figsize=(8, 4.5))

    df.boxplot(column=feature, by="spam_label")

    label = feature_labels[feature]

    plt.title(f"Distribution of the {label} by Email Type")
    plt.suptitle("")
    plt.xlabel("Email Type")
    plt.ylabel(label)

    # Replace x-axis labels
    plt.xticks([1, 2], ["Ham", "Spam"])

    # Safe filename
    filename = feature.replace("!", "exclamation")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_DIR / f"boxplot_{filename}.png")
    plt.close()

# The boxplots show that spam emails generally 
# contain higher frequencies of the word "free," 
# more exclamation marks, 
# and longer sequences of capital letters than ham emails. 
# Among these features, capital-letter sequence length 
# shows the largest difference between the two classes. 
# Although the differences are clearly visible, 
# there is still considerable overlap between spam and ham emails, 
# indicating that no single feature can perfectly separate the classes. 
# Overall, the differences are noticeable but not dramatic.

# The large number of zero values indicates that 
# many of the tracked words and characters are relatively rare. 
# Most emails do not contain specific words such as "free" or 
# large numbers of exclamation marks. 
# As a result, many features are sparse, 
# meaning that most of their values are zero.

# The numeric scale vary so dramatically across features 
# because different features measure 
# different characteristics of an email. 
# Word- and character-frequency features are percentages, 
# so their values are generally small. 
# In contrast, features such as capital-letter sequence length 
# measure counts and can reach much larger values, 
# sometimes into the thousands. 
# Since the features represent different types of measurements, 
# they naturally have different numeric scales.

# Large differences in feature scales can influence models 
# that rely on distances or numerical optimization. 
# Features with much larger values may dominate smaller-scale features, 
# even if they are not more important. 
# Therefore, models such as Logistic Regression and K-Nearest Neighbors 
# often benefit from feature scaling, 
# while Decision Trees and Random Forests are generally 
# much less affected by differences in feature scales.

# Task 2
# Separate features (X) and target variable (y)
X = df.drop(columns="spam_label")
y = df["spam_label"]

# Split the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Standardize the features.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit PCA using only the scaled training data.
pca = PCA()
pca.fit(X_train_scaled)

# Calculate cumulative explained variance.
cumulative_variance = np.cumsum(
    pca.explained_variance_ratio_
)

# Find the first component reaching 90%
n = np.argmax(cumulative_variance >= 0.90) + 1
print(f"Number of components explaining at least 90% variance: {n}")

# Create a plot
plt.figure(figsize=(8,5))

plt.plot(
    range(1, len(cumulative_variance)+1),
    cumulative_variance,
    marker="o"
)

plt.axhline(
    y=0.90,
    color="red",
    linestyle="--",
    label=f"90% Explained Variance"
)

plt.axvline(
    x=n,
    color="green",
    linestyle="--",
    label=f"{n} Components"
)

plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")

plt.legend()

plt.tight_layout()

plt.savefig("outputs/pca_cumulative_explained_variance.png")

plt.close()

# Transform both datasets using the PCA model.
# Keep only the first n principal components.
X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca = pca.transform(X_test_scaled)[:, :n]

# Task 3
# Evaluation helper
def evaluate_classifier(model_name, model, X_test_data, y_test):
    # Print evaluation results and return predictions and accuracy
    y_pred = model.predict(X_test_data)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel name: {model_name}")
    print(f"{'-' * 40}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Ham", "Spam"],
        )
    )

    return y_pred, accuracy

# Collect model results
model_results = {}

# KNN on unscaled data
# This experiment demonstrate 
# how features with large numeric ranges can dominate distances.

knn_unscaled = KNeighborsClassifier(n_neighbors=5)

knn_unscaled.fit(X_train, y_train)

knn_unscaled_pred, knn_unscaled_accuracy = evaluate_classifier(
    "KNN — Unscaled Data",
    knn_unscaled,
    X_test,
    y_test,
)

model_results["KNN — Unscaled"] = {
    "model": knn_unscaled,
    "X_test": X_test,
    "predictions": knn_unscaled_pred,
    "accuracy": knn_unscaled_accuracy,
}

# KNN on scaled data
# Scaling puts the features on comparable numeric scales,
# preventing large-valued features from dominating KNN distances.
knn_scaled = KNeighborsClassifier(n_neighbors=5)

knn_scaled.fit(X_train_scaled, y_train)

knn_scaled_pred, knn_scaled_accuracy = evaluate_classifier(
    "KNN — Scaled Data",
    knn_scaled,
    X_test_scaled,
    y_test,
)

model_results["KNN — Scaled"] = {
    "model": knn_scaled,
    "X_test": X_test_scaled,
    "predictions": knn_scaled_pred,
    "accuracy": knn_scaled_accuracy,
}

# KNN on PCA-reduced data
# KNN may benefit because distances can become
# more meaningful in a lower-dimensional feature space.

knn_pca = KNeighborsClassifier(n_neighbors=5)

knn_pca.fit(X_train_pca, y_train)

knn_pca_pred, knn_pca_accuracy = evaluate_classifier(
    "KNN — PCA-Reduced Data",
    knn_pca,
    X_test_pca,
    y_test,
)

model_results["KNN — PCA"] = {
    "model": knn_pca,
    "X_test": X_test_pca,
    "predictions": knn_pca_pred,
    "accuracy": knn_pca_accuracy,
}

# Print models comparison
print("\nKNN Comparison")
print("-" * 40)
print(f"Unscaled accuracy: {knn_unscaled_accuracy:.4f}")
print(f"Scaled accuracy:   {knn_scaled_accuracy:.4f}")
print(f"PCA accuracy:      {knn_pca_accuracy:.4f}")

# Decision Tree depth comparison
tree_depths = [3, 5, 10, None]
tree_depth_results = []

print("\nDecision Tree Depth Comparison")
print("-" * 60)

for depth in tree_depths:
    tree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42,
    )

    tree.fit(X_train, y_train)

    train_accuracy = tree.score(X_train, y_train)
    test_accuracy = tree.score(X_test, y_test)

    tree_depth_results.append(
        {
            "max_depth": depth,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        }
    )

    depth_label = "None" if depth is None else str(depth)

tree_depth_df = pd.DataFrame(tree_depth_results)
print(tree_depth_df.to_string(index=False))

# Decision Tree Depth Selection
#
# At max_depth=3, the model had the lowest training and test accuracy,
# suggesting that the tree was too simple and underfit the data.
#
# Increasing the depth to 5 improved both training and test accuracy
# while keeping the difference between them relatively small.
#
# At max_depth=10, test accuracy improved again, but training accuracy
# increased much more sharply. This larger train-test gap suggests that
# the model was beginning to overfit.
#
# With max_depth=None, the tree achieved almost perfect training
# accuracy (0.9997), but test accuracy improved only slightly compared
# with depth 10. This indicates stronger overfitting because the tree
# learned the training data in much greater detail without a comparable
# improvement on unseen data.
#
# I selected max_depth=10 for the final Decision Tree. It achieved
# nearly the highest test accuracy while remaining simpler and less
# overfit than the unrestricted tree. Although max_depth=None had a
# slightly higher test accuracy, the improvement was only about 0.2
# percentage points and did not justify the much greater model
# complexity.
chosen_depth = 10

# Train and evaluate the final tree
decision_tree = DecisionTreeClassifier(
    max_depth=chosen_depth,
    random_state=42,
)

decision_tree.fit(X_train, y_train)

tree_pred, tree_accuracy = evaluate_classifier(
    f"Decision Tree — max_depth={chosen_depth}",
    decision_tree,
    X_test,
    y_test,
)

model_results["Decision Tree"] = {
    "model": decision_tree,
    "X_test": X_test,
    "predictions": tree_pred,
    "accuracy": tree_accuracy,
}

# Random Forest
random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

random_forest.fit(X_train, y_train)

rf_pred, rf_accuracy = evaluate_classifier(
    "Random Forest",
    random_forest,
    X_test,
    y_test,
)

model_results["Random Forest"] = {
    "model": random_forest,
    "X_test": X_test,
    "predictions": rf_pred,
    "accuracy": rf_accuracy,
}

# Logistic Regression on scaled data
# Logistic Regression is sensitive to differences in feature scales,
# so it is trained on standardized data.

logistic_scaled = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="liblinear",
)

logistic_scaled.fit(X_train_scaled, y_train)

logistic_scaled_pred, logistic_scaled_accuracy = evaluate_classifier(
    "Logistic Regression — Scaled Data",
    logistic_scaled,
    X_test_scaled,
    y_test,
)

model_results["Logistic Regression — Scaled"] = {
    "model": logistic_scaled,
    "X_test": X_test_scaled,
    "predictions": logistic_scaled_pred,
    "accuracy": logistic_scaled_accuracy,
}

# Logistic Regression on PCA data
# This experiment tests whether dimensionality reduction improves
# Logistic Regression compared with using all standardized features.

logistic_pca = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="liblinear",
)

logistic_pca.fit(X_train_pca, y_train)

logistic_pca_pred, logistic_pca_accuracy = evaluate_classifier(
    "Logistic Regression — PCA-Reduced Data",
    logistic_pca,
    X_test_pca,
    y_test,
)

model_results["Logistic Regression — PCA"] = {
    "model": logistic_pca,
    "X_test": X_test_pca,
    "predictions": logistic_pca_pred,
    "accuracy": logistic_pca_accuracy,
}

# Print the comparison
print("\nLogistic Regression Comparison")
print("-" * 40)
print(f"Scaled accuracy: {logistic_scaled_accuracy:.4f}")
print(f"PCA accuracy:    {logistic_pca_accuracy:.4f}")

# Compare all model accuracies
accuracy_summary = pd.DataFrame(
    {
        "Model": model_results.keys(),
        "Accuracy": [
            result["accuracy"]
            for result in model_results.values()
        ],
    }
).sort_values(
    by="Accuracy",
    ascending=False,
)

print("\nOverall Model Comparison")
print("-" * 60)
print(
    accuracy_summary.to_string(
        index=False,
        formatters={"Accuracy": "{:.4f}".format},
    )
)

# Determine the best model dynamically
best_model_name = accuracy_summary.iloc[0]["Model"]
best_result = model_results[best_model_name]

print(f"\nBest model by accuracy: {best_model_name}")
print(f"Best accuracy: {best_result['accuracy']:.4f}")

# Model Comparison Summary
#
# The Random Forest achieved the highest test accuracy (94.57%) and
# was the best-performing classifier overall. 
# Logistic Regression on the scaled data ranked second (92.94%), 
# followed by the Decision Tree (90.88%), 
# KNN on scaled data (90.77%), KNN with PCA (90.66%),
# and KNN on the original unscaled data (79.91%).
#
# Scaling dramatically improved KNN performance, 
# increasing its accuracy from 79.91% to 90.77%. 
# This confirms the expectation from Task 2 
# that KNN is highly sensitive to differences in feature scales 
# because it relies on distance calculations.
#
# PCA did not improve performance for either KNN or Logistic Regression. 
# KNN with PCA performed nearly the same 
# as KNN with the full scaled data (90.66% vs. 90.77%), 
# while Logistic Regression performed better on the scaled data 
# than on the PCA-reduced data (92.94% vs. 91.86%). 
# This suggests that reducing the dimensionality to preserve 90% 
# of the variance removed some information 
# that was still useful for classification.
#
# For a spam filter, I would not optimize accuracy alone. 
# Instead, I would prioritize minimizing false positives 
# (legitimate emails incorrectly marked as spam). 
# Missing an important email from an employer, bank, doctor, 
# or family member could have much more serious consequences 
# than receiving an occasional spam message.
# Therefore, I would place greater importance on reducing false
# positives, even if it means allowing a small number 
# of spam messages into the inbox.

# Confusion matrix for the best model
ConfusionMatrixDisplay.from_predictions(
    y_test,
    best_result["predictions"],
    display_labels=["Ham", "Spam"],
    values_format="d",
)

plt.title(f"Confusion Matrix — {best_model_name}")
plt.tight_layout()

plt.savefig(
    "outputs/best_model_confusion_matrix.png",
    bbox_inches="tight",
)

plt.close()

false_positives = (
    (y_test.to_numpy() == 0)
    & (best_result["predictions"] == 1)
).sum()

false_negatives = (
    (y_test.to_numpy() == 1)
    & (best_result["predictions"] == 0)
).sum()

print("\nBest Model Errors")
print("-" * 40)
print(f"False positives: {false_positives}")
print(f"False negatives: {false_negatives}")

if false_positives > false_negatives:
    print("The best model makes more false-positive errors.")
elif false_negatives > false_positives:
    print("The best model makes more false-negative errors.")
else:
    print("The best model makes equal numbers of both error types.")

# Decision Tree top 10 feature importances
tree_importances = pd.Series(
    decision_tree.feature_importances_,
    index=X.columns,
).sort_values(ascending=False)

print("\nDecision Tree — Top 10 Features")
print("-" * 50)
print(tree_importances.head(10))

# Random Forest top 10 feature importances
rf_importances = pd.Series(
    random_forest.feature_importances_,
    index=X.columns,
).sort_values(ascending=False)

print("\nRandom Forest — Top 10 Features")
print("-" * 50)
print(rf_importances.head(10))

# Random Forest feature-importance chart
top_10_rf_importances = rf_importances.head(10).sort_values()

plt.figure(figsize=(9, 6))

top_10_rf_importances.plot(kind="barh")

plt.title("Random Forest — Top 10 Feature Importances")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    "outputs/feature_importances.png",
    bbox_inches="tight",
)

plt.close()

# Comparing the tree and forest features
tree_top_10 = set(tree_importances.head(10).index)
rf_top_10 = set(rf_importances.head(10).index)

common_features = tree_top_10.intersection(rf_top_10)

print("\nFeatures appearing in both top-10 lists:")
for feature in sorted(common_features):
    print(f"- {feature}")

# Feature Importance Summary
#
# The Decision Tree and Random Forest showed strong agreement on the
# most important features. Six of their top ten features overlapped,
# including char_freq_$, char_freq_!, word_freq_remove,
# word_freq_free, word_freq_hp, and capital_run_length_total.
#
# The Decision Tree relied heavily on a few features, particularly
# char_freq_$ and word_freq_remove, while the Random Forest
# distributed importance across a larger number of features. This is
# expected because a Random Forest averages the predictions of many
# decision trees, resulting in more stable and balanced feature
# importance estimates.
#
# The results generally match my intuition about what makes an email
# spam. Features such as the frequency of dollar signs, exclamation
# marks, the word "free," and long sequences of capital letters are
# commonly associated with promotional or spam messages. 

# Task 4
# Cross-validation results
# Helper function
def run_cross_validation(model_name, model, X_data, y_data):
    scores = cross_val_score(
        model,
        X_data,
        y_data,
        cv=5,
    )

    print(f"\n{model_name}")
    print("-" * 50)
    print(f"Fold scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.4f}")
    print(f"Standard deviation: {scores.std():.4f}")

    return {
        "Model": model_name,
        "Mean Accuracy": scores.mean(),
        "Std Dev": scores.std(),
    }

cv_results = []

# KNN (Unscaled)
cv_results.append(
    run_cross_validation(
        model_name="KNN — Unscaled",
        model=KNeighborsClassifier(n_neighbors=5),
        X_data=X_train,
        y_data=y_train,
    )
)


# KNN (Scaled)
# Use a pipeline so the scaler is fitted separately within each fold.
knn_scaled_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])

cv_results.append(
    run_cross_validation(
        model_name="KNN — Scaled",
        model=knn_scaled_pipeline,
        X_data=X_train,
        y_data=y_train,
    )
)


# KNN (PCA)
# Use a pipeline so scaling and PCA are fitted separately within each fold.
knn_pca_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=n)),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])

cv_results.append(
    run_cross_validation(
        model_name="KNN — PCA",
        model=knn_pca_pipeline,
        X_data=X_train,
        y_data=y_train,
    )
)


# Decision Tree
decision_tree_cv = DecisionTreeClassifier(
    max_depth=chosen_depth,
    random_state=42,
)

cv_results.append(
    run_cross_validation(
        model_name="Decision Tree",
        model=decision_tree_cv,
        X_data=X_train,
        y_data=y_train,
    )
)


# Random Forest
random_forest_cv = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

cv_results.append(
    run_cross_validation(
        model_name="Random Forest",
        model=random_forest_cv,
        X_data=X_train,
        y_data=y_train,
    )
)


# Logistic Regression (Scaled)
logistic_scaled_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    (
        "logistic",
        LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="liblinear",
        ),
    ),
])

cv_results.append(
    run_cross_validation(
        model_name="Logistic Regression — Scaled",
        model=logistic_scaled_pipeline,
        X_data=X_train,
        y_data=y_train,
    )
)


# Logistic Regression (PCA)
logistic_pca_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=n)),
    (
        "logistic",
        LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="liblinear",
        ),
    ),
])

cv_results.append(
    run_cross_validation(
        model_name="Logistic Regression — PCA",
        model=logistic_pca_pipeline,
        X_data=X_train,
        y_data=y_train,
    )
)


# Summary table
cv_results_df = pd.DataFrame(cv_results)

cv_results_df = cv_results_df.sort_values(
    by="Mean Accuracy",
    ascending=False,
)

print("\nCross-Validation Summary")
print("-" * 65)

print(
    cv_results_df.to_string(
        index=False,
        formatters={
            "Mean Accuracy": "{:.4f}".format,
            "Std Dev": "{:.4f}".format,
        },
    )
)

# Cross-Validation Summary
#
# The Random Forest achieved the highest mean cross-validation
# accuracy (95.43%), confirming that it was the best-performing
# classifier overall. Logistic Regression trained on the scaled data
# ranked second (92.31%), while KNN trained on the original unscaled
# data remained the weakest model (79.43%).
#
# Logistic Regression (both the scaled and PCA versions) had the
# lowest standard deviation across the five folds (0.0077),
# indicating the most consistent performance. Although the Random
# Forest achieved the highest accuracy, its fold-to-fold variation
# was slightly larger (0.0133), suggesting a small trade-off between
# maximum accuracy and stability.
#
# Overall, the cross-validation results closely matched the ranking
# observed with the single train/test split. The Random Forest
# remained the strongest classifier, and Logistic Regression
# continued to outperform both the Decision Tree and KNN. One
# difference was that KNN with PCA achieved a slightly higher mean
# cross-validation accuracy than KNN using only scaled features,
# whereas the opposite was observed with the single train/test split.
# This demonstrates why cross-validation provides a more reliable
# estimate of model performance than relying on a single split.

# Task 5
# Random Forest pipeline
rf_pipeline = Pipeline([
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
    )
])

rf_pipeline.fit(X_train, y_train)

rf_pipeline_pred = rf_pipeline.predict(X_test)

print("\nRandom Forest Pipeline")
print("-" * 40)
print(f"Accuracy: {accuracy_score(y_test, rf_pipeline_pred):.4f}")
print(
    classification_report(
        y_test,
        rf_pipeline_pred,
        target_names=["Ham", "Spam"],
    )
)

# Logistic Regression pipeline
logistic_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="liblinear",
        ),
    ),
])

logistic_pipeline.fit(X_train, y_train)

logistic_pipeline_pred = logistic_pipeline.predict(X_test)

print("\nLogistic Regression Pipeline")
print("-" * 50)
print(
    f"Accuracy: "
    f"{accuracy_score(y_test, logistic_pipeline_pred):.4f}"
)
print(
    classification_report(
        y_test,
        logistic_pipeline_pred,
        target_names=["Ham", "Spam"],
    )
)

print("\nPipeline Verification")
print("-" * 40)

print(
    "Random Forest predictions match manual approach:",
    np.array_equal(rf_pipeline_pred, rf_pred),
)

print(
    "Logistic Regression predictions match manual approach:",
    np.array_equal(
        logistic_pipeline_pred,
        logistic_scaled_pred,
    ),
)

# Pipeline Reflection
#
# The two pipelines do not have the same structure. The Random Forest
# pipeline contains only the classifier because tree-based models are
# not sensitive to feature scaling. The Logistic Regression pipeline
# includes a StandardScaler before the classifier because Logistic
# Regression performs better when features are standardized.
#
# The pipeline predictions matched the manual implementation exactly,
# confirming that the pipelines reproduced the same models while
# simplifying the workflow.
#
# Packaging preprocessing and the classifier into a pipeline reduces
# the risk of applying transformations incorrectly or introducing data
# leakage. Pipelines also make models easier to reuse, share with other
# developers, and deploy because users can provide raw feature data
# without manually repeating the preprocessing steps.