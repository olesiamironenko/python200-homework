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