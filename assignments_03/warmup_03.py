from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Create the output directory if it doesn't exist
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- Preprocessing ---

# Q1
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    stratify=y,
    random_state=42,
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


# Q2
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"X_train_scaled columns means: {np.round(X_train_scaled.mean(axis=0), 10)}")

# The scaler is fit on X_train only to keep the test set truly unseen and 
# obtain a clean measure of how the model performs on new data.


# --- KNN ---

# Q1
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

preds = knn.predict(X_test)

print(f"Accuracy score: {accuracy_score(y_test, preds)}")
print(f"Full classification report: \n{classification_report(y_test, preds)}")


# Q2
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)

preds_scaled = knn_scaled.predict(X_test_scaled)

print(f"Scaled data accuracy score: {accuracy_score(y_test, preds_scaled)}")

# Scaling slightly reduced the accuracy from 1.00 to 0.93.
# The Iris features are already on similar scales, so scaling was not necessary.
# Scaling changed the distances between the data points, 
# so KNN chose different nearest neighbors.


# Q3
knn = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn, X_train, y_train, cv=5)

print(f"Cross Validation Scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.3f}")
print(f"Std:  {cv_scores.std():.3f}")

# Cross-validation is more trustworthy than a single train/test split
# because the model is evaluated on multiple different subsets of the data.
# This gives a more reliable estimate of how well the model is likely to
# perform on new, unseen data.


# Q4
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k={k:2d}:  mean={scores.mean():.3f}")

# Both k=5 and k=7 achieved the highest mean cross-validation accuracy (0.975).
# I would choose k=5 because it performs just as well as k=7 while using fewer neighbors,
# making it a simpler model.


# --- Classifier Evaluation --- 
# Q1
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()
plt.title("KNN Confusion Matrix (Iris)")
plt.savefig(OUTPUT_DIR / "knn_confusion_matrix.png")

# The model did not confuse any species.
# All test samples were classified correctly, 
# so every prediction appears on the diagonal of the confusion matrix.


# --- The sklearn API: Decision Trees --- 
# Q1
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

dt_preds = dt.predict(X_test)

print(f"Decision Tree Accuracy score: {accuracy_score(y_test, dt_preds)}")
print(f"Decision Tree Full classification report: \n{classification_report(y_test, dt_preds)}")

# Comment 1:
# The Decision Tree performed well, 
# but KNN achieved a perfect accuracy on this test set 
# and therefore performed slightly better.

# Comment 2:
# Scaling should not change the results 
# because Decision Trees split the data using feature values 
# instead of distance calculations.


# --- Logistic Regression and Regularization --- 
# Q1
c_values = [0.01, 1.0, 100]
for c in c_values:
    logreg_scaled = LogisticRegression(C=c, max_iter=1000, solver="liblinear")
    logreg_scaled.fit(X_train_scaled, y_train)

    coef_size = np.abs(logreg_scaled.coef_).sum()

    print(f"C = {c:4}: Total coefficient magnitude = {coef_size:7.4f}")


# --- PCA --- 

digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting


# Q1
# Print X_digits and images shapes
print(f"X_digits Shape: {X_digits.shape}")
print(f"images Shape: {images.shape}")

# Create 1-row subplot showing one example of each digit class (0-9)
fig, axes = plt.subplots(1, 10, figsize=(12, 2))

# Display one example of each digit
for digit in range(10):
    # Find the first occurrence of this digit
    index = np.where(y_digits == digit)[0][0]

    # Display the image
    axes[digit].imshow(images[index], cmap="gray_r")

    # Add the digit label as the title
    axes[digit].set_title(str(digit))

    # Hide the axes
    axes[digit].axis("off")

plt.tight_layout()

# Save the figure
plt.savefig(OUTPUT_DIR / "sample_digits.png")
plt.close()


# Q2
# Fit the model
pca = PCA()
pca.fit(X_digits)

# Get the scores
scores = pca.transform(X_digits)

# Create a scatter plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    scores[:, 0], 
    scores[:, 1], 
    c=y_digits, # c = color array
    cmap='tab10', 
    s=10
)  
plt.colorbar(scatter, label='Digit')

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Digits projected onto first two principal components")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pca_2d_projection.png")
plt.close()

# The same-digit images do tend to cluster together 
# in the first two principal components, 
# but there is noticeable overlap between several digit classes.
# This indicates that two principal components capture much 
# of the data's structure, though they are 
# not sufficient enough to completely separate all digits.


# Q3
# Plot cumulative explained variance vs. number of components
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 5))

plt.plot(cumulative_variance)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance by PCA Components")

plt.grid(True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pca_variance_explained.png")
plt.close()

components_80 = np.argmax(cumulative_variance >= 0.80) + 1
print("Components needed for 80% variance:", components_80)

# Approximately 13 principal components are needed to explain
# about 80% of the total variance.


# Q4
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

# Reconstruct the first 5 digits in X_digits
n_values = [2, 5, 15, 40]

reconstructions = {}

for n in n_values:
    reconstructions[n] = []

    for sample_idx in range(5):
        reconstructed_digit = reconstruct_digit(
            sample_idx,
            scores,
            pca,
            n
        )

        reconstructions[n].append(reconstructed_digit)

# Build a 5 x 5 grid of subplots:
# Row 0: Original digits
# Row 1: Reconstructions with n = 2
# Row 2: Reconstructions with n = 5
# Row 3: Reconstructions with n = 15
# Row 4: Reconstructions with n = 40
# Columns 0-4: The first 5 digits in X_digits
row_labels = ["Original", "n = 2", "n = 5", "n = 15", "n = 40"]

fig, axes = plt.subplots(5, 5, figsize=(9, 9))

# Row 0: Original digits
for sample_idx in range(5):
    axes[0, sample_idx].imshow(
        images[sample_idx],
        cmap="gray_r"
    )
    axes[0, sample_idx].set_title(
        f"Digit {y_digits[sample_idx]}"
    )
    axes[0, sample_idx].axis("off")

axes[0, 0].set_ylabel(
    row_labels[0],
    rotation=0,
    labelpad=35,
    va="center"
)

# Rows 1-4: PCA reconstructions for n = 2, 5, 15, and 40
for row_idx, n in enumerate(n_values, start=1):
    for sample_idx in range(5):
        axes[row_idx, sample_idx].imshow(
            reconstructions[n][sample_idx],
            cmap="gray_r"
        )
        axes[row_idx, sample_idx].axis("off")

    axes[row_idx, 0].set_ylabel(
        row_labels[row_idx],
        rotation=0,
        labelpad=35,
        va="center"
    )

plt.suptitle("PCA Digit Reconstructions")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "pca_reconstructions.png",
    bbox_inches="tight"
)
plt.close()

# The digits become recognizable with about 5 principal components.
# Using 15 components provides only modest visual improvement, 
# while 40 components produce reconstructions that are 
# much closer to the original images. 
# This is generally consistent with the explained variance curve, 
# which begins to level off around 13 components.
# Most of the important structure is captured early, 
# while additional components mainly refine image details.