import json
import platform
import sklearn
import joblib
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report
)

# Step 1

# fetch weather data for Wayne, NJ
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 40.9254,
    "longitude": -74.2765,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

# Print a summary of the dataset
print("\nWeather Dataset Summary")
print("-" * 50)

print("Location: Wayne, NJ")
print(f"Shape: {df.shape}")

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn information:")
df.info()

print("\nMissing values:")
print(df.isna().sum())

print("\nDescriptive statistics:")
print(df.describe())


# Step 2
# Create "good for running" label
df["good_for_running"] = (
    df["temperature_2m_max"].between(7, 26)
    & (df["temperature_2m_min"] >= 0)
    & (df["precipitation_sum"] < 3.0)
    & (df["wind_speed_10m_max"] < 30)
).astype(int)

# Print the class distribution
class_counts = df["good_for_running"].value_counts().sort_index()

print("\nClass Distribution")
print("-" * 30)
print(class_counts)

# Print good for running fraction
good_day_fraction = class_counts[1] / class_counts.sum()

print(f"Good running days: {good_day_fraction:.2%}")

# About 37% of the days were labeled good for running.
# This seems reasonable for Wayne, New Jersey 
# because many winter days are too cold and 
# some summer days are too hot, rainy, or windy 
# for comfortable running.


# Step 3
# Define features and target
X = df[
    [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]
]

y = df["good_for_running"]

# Split the data into train (80%) and test (20%) sets, stratifying on the label.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Build the pipeline
logistic_weather_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=1000)),
    ]
)

# Create the parameter grid
logistic_weather_param_grid = {
    "logistic__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
}

# Run GridSearchCV
logistic_weather_grid_search = GridSearchCV(
    estimator=logistic_weather_pipeline,
    param_grid=logistic_weather_param_grid,
    cv=5,
    scoring="roc_auc",
)

logistic_weather_grid_search.fit(X_train, y_train)

# Print the best values
print(f"{'Best C Value:':<22}{logistic_weather_grid_search.best_params_['logistic__C']}")
print(f"{'Best CV AUC:':<22}{logistic_weather_grid_search.best_score_:.4f}")

# Predict on the test set
logistic_weather_predictions = logistic_weather_grid_search.best_estimator_.predict(X_test)

logistic_weather_probabilities = (
    logistic_weather_grid_search.best_estimator_
    .predict_proba(X_test)[:, 1]
)

# Classification report
print("\nClassification Report")
print("-" * 50)

print(
    classification_report(
        y_test,
        logistic_weather_predictions,
    )
)

# Test AUC
logistic_weather_test_auc = roc_auc_score(
    y_test,
    logistic_weather_probabilities,
)

print(f"\n{'Test AUC:':<22}{logistic_weather_test_auc:.4f}")

# ROC curve
weather_fpr, weather_tpr, _ = roc_curve(
    y_test,
    logistic_weather_probabilities,
)

fig, ax = plt.subplots(figsize=(6, 5))

RocCurveDisplay(
    fpr=weather_fpr,
    tpr=weather_tpr,
    roc_auc=logistic_weather_test_auc,
).plot(
    ax=ax,
    name=f"Logistic Regression (AUC = {logistic_weather_test_auc:.2f})",
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Random Classifier",
)

ax.set_title("Weather Classifier ROC Curve")

plt.tight_layout()
plt.savefig("outputs/weather_roc.png")


# Step 4
# The model achieved a test AUC of 0.6659, which means 
# it can separate good and bad running days better than random, 
# but there is still room for improvement. 
# This is about what I expected for a Logistic Regression model 
# using onlyfour weather features.
#
# The classification report shows that false positives are
# slightly more common than false negatives. This means the app
# would sometimes recommend running on days that are not ideal.
# I would rather the app under-recommend running than over-
# recommend it because poor weather could make running less
# enjoyable or even unsafe.
# 
# I would start with the default threshold of 0.5. If I wanted
# the app to make fewer incorrect recommendations, I would
# increase the threshold slightly so the model is more confident
# before recommending a run.


# Step 5
best_logistic_weather_pipeline = logistic_weather_grid_search.best_estimator_

model_path = "models/weather_classifier.pkl"
metadata_path = "models/weather_classifier_metadata.json"

# Save the pipeline
joblib.dump(best_logistic_weather_pipeline, model_path)

# Save metadata
metadata = {
    "python_version": platform.python_version(),
    "scikit_learn_version": sklearn.__version__,
    "feature_names": list(X.columns),
    "best_hyperparameters": logistic_weather_grid_search.best_params_,
    "test_auc": float(logistic_weather_test_auc),
    "city": {
        "name": "Wayne, New Jersey",
        "latitude": params["latitude"],
        "longitude": params["longitude"],
    },
    "label_thresholds": {
        "temperature_2m_max": "between 7°C and 26°C",
        "temperature_2m_min": "at least 0°C",
        "precipitation_sum": "less than 3.0 mm",
        "wind_speed_10m_max": "less than 30 km/h",
    },
}

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

# Print a confirmation message
print(f"Model is saved to {model_path}")
print(f"Metadata is saved to {metadata_path}")