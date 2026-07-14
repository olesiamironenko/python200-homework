# Imports
from pathlib import Path

# path to run the program from PYTHON200-HOMEWORK/assignment_02 
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

# --- scikit-learn API ---
# Q1
import numpy as np
from sklearn.linear_model import LinearRegression

years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])
new_years = np.array([4, 8]).reshape(-1, 1)

model = LinearRegression()                    # 1. create model
model.fit(years, salary)                      # 2. fit model to data (learn)
salary_predicted = model.predict(new_years)   # 3. predict with new data
print(salary_predicted)  
print(model.coef_[0])
print(model.intercept_)


# Q2
x = np.array([10, 20, 30, 40, 50])
print(x.shape)
x = x.reshape(-1, 1)
print(x.shape)

# scikit-learn needs X to be 2D because it always treats data (X in this case) as a table/dataframe which has columns and rows.
# Columns represent sammples, rows represent features.


#Q3
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

# 1. Create the model
kmeans = KMeans(n_clusters=3, random_state=42)
# 2. Fit -- find cluster centers
kmeans.fit(X_clusters)          
# 3. Predict a label for each point                
labels = kmeans.predict(X_clusters) 
# 4. Print the cluster centers and how many points fell into each cluster
print(kmeans.cluster_centers_)
print(np.bincount(labels))

# 5. Create a figure for plot
plt.figure(figsize=(8, 6))

# Plot the clustered data points
plt.scatter(
    X_clusters[:, 0], 
    X_clusters[:, 1], 
    c=labels, 
    cmap='viridis', 
    s=60, alpha=0.7
)

# Plot the cluster centers as black X markers
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="black",
    marker="X",
    s=200,
    label="Cluster Centers",
)

plt.title("K-Means Clustering with Three Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()

# Save the figure
plt.savefig(OUTPUT_DIR / "kmeans_clusters.png")
plt.close()

# --- Linear Regression ---

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
num_patients = 100
age = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# Q1
# Create scatter plot
plt.figure(figsize=(8, 6))

plt.scatter(
    age,
    cost,
    c=smoker,
    cmap="coolwarm",
)

plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Medical Cost")

plt.tight_layout()

# Save output 
plt.savefig(OUTPUT_DIR / "cost_vs_age.png")
plt.close()

# ----------------------------------------
# Observation:

# The scatter plot shows two distinct groups of points.
# Smokers generally have much higher medical costs than non-smokers,
# even at similar ages. 
# This suggests that smoker status is an important predictor 
# of medical cost and should be included in the regression model.
# ----------------------------------------

# Q2
# Reshape age into a 2D array
X = age.reshape(-1, 1)

# Target variable
y = cost

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Print the shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Q3
# Create the linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Print the slope and intercept
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
print("RMSE:", rmse)

# Calculate R²
r2 = model.score(X_test, y_test)
print("R² on the X test set:", r2)

# ----------------------------------------
# Interpretation:

# The slope represents the estimated increase in medical cost
# for each additional year of age. 
# For example, if the slope is around 200, 
# the model predicts that medical costs increase by about $200 
# for every one-year increase in age.
# ----------------------------------------

# Q4
# Create a feature matrix with age and smoker status
X_full = np.column_stack([age, smoker])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_full,
    cost,
    test_size=0.2,
    random_state=42,
)

# Create and fit the model
model_full = LinearRegression()
model_full.fit(X_train, y_train)

# Test R²
r2_full = model_full.score(X_test, y_test)

print("R² on the X_full test set::", r2_full)

# Print the coefficients
print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])

# ----------------------------------------
# Interpretation:

# Compared to the model using only age, 
# the R² is much higher after adding smoker status. 
# This shows that smoking is an important predictor of medical costs 
# and greatly improves the model's accuracy.
#
# The smoker coefficient represents the additional medical cost
# associated with being a smoker, while holding age constant.
# For example, if the coefficient is around 15,000, the model predicts that 
# smokers have medical costs about $15,000 higher 
# than non-smokers of the same age.
# ----------------------------------------

# Q5
# Predict using the two-feature model
y_pred = model_full.predict(X_test)

# Create the plot
plt.figure(figsize=(8, 6))

plt.scatter(y_pred, y_test, alpha=0.8)

# Diagonal reference line (perfect predictions)
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    "r--",
    label="Perfect Prediction"
)

plt.title("Predicted vs Actual")
plt.xlabel("Predicted Medical Cost")
plt.ylabel("Actual Medical Cost")
plt.legend()

plt.tight_layout()
plt.savefig("outputs/predicted_vs_actual.png")
plt.close()

# ----------------------------------------
# Interpretation:

# Points on the diagonal represent perfect predictions.
# Points above the diagonal have actual costs higher than predicted,
# meaning the model underestimated the medical cost.
# Points below the diagonal have actual costs lower than predicted,
# meaning the model overestimated the medical cost.
# ----------------------------------------
