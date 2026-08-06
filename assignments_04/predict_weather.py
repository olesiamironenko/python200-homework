import json
import joblib
import pandas as pd

# Task 1
# Load the Pipeline
weather_classifier = joblib.load(
    "models/weather_classifier.pkl"
)

# Load the metadata
with open(
    "models/weather_classifier_metadata.json",
    "r",
    encoding="utf-8",
) as file:
    metadata = json.load(file)

# Print the model's key metadata (city, features, test AUC)
print("\nWeather Classifier Metadata")
print("-" * 40)
print(f"City:     {metadata['city']['name']}")
print(
    f"Location: "
    f"{metadata['city']['latitude']}, "
    f"{metadata['city']['longitude']}"
)
print(f"Features: {metadata['feature_names']}")
print(f"Test AUC: {metadata['test_auc']:.4f}")


# Task 2
# Create a DataFrame of at least five hypothetical days
new_days = pd.DataFrame(
    [
        # Clearly good
        {
            "temperature_2m_max": 20.0,
            "temperature_2m_min": 10.0,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 12.0,
        },
        # Clearly good
        {
            "temperature_2m_max": 16.0,
            "temperature_2m_min": 7.0,
            "precipitation_sum": 0.5,
            "wind_speed_10m_max": 18.0,
        },
        # Clearly bad: too cold
        {
            "temperature_2m_max": 3.0,
            "temperature_2m_min": -4.0,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 10.0,
        },
        # Clearly bad: heavy rain and wind
        {
            "temperature_2m_max": 18.0,
            "temperature_2m_min": 12.0,
            "precipitation_sum": 12.0,
            "wind_speed_10m_max": 38.0,
        },
        # Borderline
        {
            "temperature_2m_max": 26.0,
            "temperature_2m_min": 1.0,
            "precipitation_sum": 2.9,
            "wind_speed_10m_max": 29.0,
        },
    ]
)

# Match the exact feature order used during training
new_days = new_days[metadata["feature_names"]]

# Predict labels and probability of class 1: good for running
predicted_labels = weather_classifier.predict(new_days)
good_probabilities = weather_classifier.predict_proba(new_days)[:, 1]

for index, (features, prediction, probability) in enumerate(
    zip(
        new_days.to_dict(orient="records"),
        predicted_labels,
        good_probabilities,
    ),
    start=1,
):
    label = "good" if prediction == 1 else "skip"

    print(f"\nDay {index}")
    print("-" * 30)

    for feature_name, value in features.items():
        print(f"{feature_name}: {value}")

    print(f"Prediction: {label}")
    print(f"Confidence for good: {probability:.2%}")

# The borderline day received a probability of 49.95% for being
# good for running. Because this probability is almost exactly
# at the 0.5 threshold, I would describe the model as uncertain.
# If the model predicted 0.52, I would also consider that a weak
# recommendation rather than a confident one. In a real app, I
# would show a message such as "borderline conditions" instead of
# making a strong recommendation.
# 
# If predict_weather.py ran before train_weather_classifier.py,
# the saved model and metadata files would not exist, causing a
# FileNotFoundError. A more helpful error message would explain
# that the training script must be run first to create the model
# and metadata files.
# 
# In a production system, the prediction script would retrieve
# tomorrow's weather forecast from a weather API instead of using
# manually created data. It would build a DataFrame with the same
# feature names and order expected by the trained model, then use
# the saved pipeline to predict whether tomorrow is good for
# running.