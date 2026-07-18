# ----------------------------------------
# Observation:

# The CSV file uses semicolons (;) as field separators instead of commas,
# so pd.read_csv() must be called with sep=";"
# ----------------------------------------

# Import necessary modules
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Create the output directory if it doesn't exist
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Task 1: Load and Explore
# Load the dataset
df = pd.read_csv("student_performance_math.csv", sep=";")

# Explore the dataset
print("Shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

# Plot the distribution of final grades
plt.figure(figsize=(8, 6))

plt.hist(
    df["G3"],
    bins=21,
    edgecolor="black"
)

plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "g3_distribution.png")
plt.close()

# Task 2: Preprocess the Data
# Print the original shape
print(f"Shape before filtering:{df.shape}")

# Remove students who did not take the final exam
df_filtered = df[df["G3"] != 0].copy()

# Print the filtered shape
print(f"Shape after filtering: {df_filtered.shape}")
print(f"Rows removed: {len(df) - len(df_filtered)}")

# Students with G3 = 0 were absent from the final exam 
# rather than earning a true grade of zero. 
# Keeping them would treat exam non-participation 
# as extremely poor academic performance 
# and could distort the relationships the regression model learns.

# Define the yes/no columns according to the dataset description
yes_no_columns = [
    "schoolsup",
    "internet",
    "higher",
    "activities",
]

print(yes_no_columns)

# Convert the yes/no columns to 1/0 
df_filtered[yes_no_columns] = df_filtered[yes_no_columns].apply(
    lambda column: column.map({"yes": 1, "no": 0})
)

# Convert the F/M in "sex" column to 1/0
df_filtered["sex"] = df_filtered["sex"].map(
    {"F": 0, "M": 1}
)

# Verify the conversions
print(df_filtered[yes_no_columns + ["sex"]].head())
print(df_filtered[yes_no_columns + ["sex"]].dtypes)

# Compute the Pearson correlation between absences and G3 on original dataset
original_correlation = df["absences"].corr(df["G3"])

# Compute the Pearson correlation between absences and G3 on filtered dataset
filtered_correlation = df_filtered["absences"].corr(df_filtered["G3"])

print(f"Absences/G3 correlation in original data: {original_correlation}")
print(f"Absences/G3 correlation in filtered data: {filtered_correlation}")

# Exploratory scatter plots for Absences vs Final Grade (G3) comparison 
# of original and filtered datasets
plt.figure(figsize=(12, 5))

plt.suptitle("Absences vs Final Grade (G3)", fontsize=16)

# Left subplot
plt.subplot(1, 2, 1)
plt.scatter(df["absences"], df["G3"], s=20, alpha=0.6)
plt.title("Original Dataset")
plt.xlabel("Absences")
plt.ylabel("Final Grade (G3)")

# Right subplot
plt.subplot(1, 2, 2)
plt.scatter(df_filtered["absences"], df_filtered["G3"], s=20, alpha=0.6)
plt.title("Filtered Dataset")
plt.xlabel("Absences")
plt.ylabel("Final Grade (G3)")

plt.tight_layout()
plt.savefig("outputs/absences_vs_g3_comparison.png")
plt.close()

# ----------------------------------------
# In the original dataset, students with G3 = 0 also have 
# zero recorded absences. 
# These points sit at (0 absences, 0 grade), which works against
# the usual negative relationship between absences and grades 
# and makes absences appear to be a weak predictor.
#
# After those exam-absence rows are removed, 
# the expected pattern becomes clearer: 
# students with more absences tend to have lower final grades.
# ----------------------------------------

# Task 3: Exploratory Data Analysis
# Compute the Pearson correlation between each numeric feature and G3
correlations = df_filtered.corr(numeric_only=True)["G3"].drop("G3")

# Print them sorted from most negative to most positive
print(correlations.sort_values())

# ----------------------------------------
# As expected, the G2 and G1 have the strongest relationship with G3,
# since they are earlier grades from the same course.
# 
# Study time has only a weak positive correlation with G3.
# Although students who report studying more tend to earn
# slightly higher grades, many other factors also influence
# final performance.
# ----------------------------------------

# 1. Scatter Plot: G2 vs G3
plt.figure(figsize=(8, 6))

plt.scatter(
    df_filtered["G2"],
    df_filtered["G3"],
    alpha=0.6,
    s=25,
)

plt.title("Second Period Grade vs Final Grade")
plt.xlabel("Second Period Grade (G2)")
plt.ylabel("Final Grade (G3)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "g2_vs_g3_scatter.png")
plt.close()

# ----------------------------------------
# Students with higher second-period grades generally earn
# higher final grades. The strong linear trend supports the
# high Pearson correlation between G2 and G3.
# ----------------------------------------

# 2. Box Plot: Study Time vs G3
plt.figure(figsize=(8, 6))

df_filtered.boxplot(
    column="G3",
    by="studytime",
    grid=False,
)

plt.title("Final Grade by Weekly Study Time")
plt.suptitle("")      # Removes pandas' automatic title

plt.xlabel("Study Time Category")
plt.ylabel("Final Grade (G3)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "studytime_vs_g3_boxplot.png")
plt.close()

# ----------------------------------------
# Students reporting more study time generally have slightly
# higher median final grades, although there is considerable
# overlap between the study time groups.
# ----------------------------------------

# Task 4: Baseline Model
# Use "failures" as the feature
X_failure = df_filtered["failures"].to_numpy().reshape(-1,1)
y_failure = df_filtered["G3"]

# Split data into training and test sets
X_failure_train, X_failure_test, y_failure_train, y_failure_test = train_test_split(
  X_failure,
  y_failure,
  test_size = 0.2,
  random_state = 42,
)

# Create and fit the model
failure_model = LinearRegression()
failure_model.fit(X_failure_train, y_failure_train)

# Print the slope
print(f"Slope: {failure_model.coef_[0]}")

# Predict on the test set
y_failure_test_pred = failure_model.predict(X_failure_test)

# Compute and print RMSE
failure_rmse = np.sqrt(np.mean((y_failure_test_pred - y_failure_test) ** 2))
print(f"RMSE: {failure_rmse:+.3f}")

# Compute and print R2
failure_test_r2 = failure_model.score(X_failure_test, y_failure_test)
print(f"R2: {failure_test_r2:+.3f}")

# ----------------------------------------
# Each additional past class failure is associated with 
# a decrease of about 1.43 pointslower predicted final grade. 
# 
# The RMSE of about 2.96 means that predictions are 
# typically off by roughly 3 grade points on a 0–20 scale.
# This is a substantial error, so failures alone 
# is not enough to predict final grades well.
#
# The test R² of about 0.09 means the model explains only about 9%
# of the variation in final grades. This is consistent with the EDA:
# failures has a noticeable negative relationship with G3, but it is
# not strong enough to be an accurate predictor by itself.
# ----------------------------------------

# Task 5: Build the Full Model
# Use all of the numeric and binary features from the Feature Guide:
feature_cols = [
    "age", 
    "Medu", 
    "Fedu", 
    "traveltime", 
    "studytime", 
    "failures",
    "absences", 
    "freetime", 
    "goout", 
    "Walc", 
    "schoolsup",
    "internet", 
    "higher", 
    "activities", 
    "sex"
]
X_full = df_filtered[feature_cols].to_numpy(dtype=float)
y_full = df_filtered["G3"].to_numpy(dtype=float)

# Split into training and test sets
X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(
  X_full,
  y_full,
  test_size = 0.2,
  random_state = 42,
)

# Fit a LinearRegression model
full_model = LinearRegression()
full_model.fit(X_full_train, y_full_train)

# Compute and print train R2 and test R2
full_train_r2 = full_model.score(X_full_train, y_full_train)
full_test_r2 = full_model.score(X_full_test, y_full_test)

print(f"Full Model Train R2: {full_train_r2:+.3f}")
print(f"Full Model Test R2: {full_test_r2:+.3f}")

# Predict on the test set
y_full_test_pred = full_model.predict(X_full_test)

# Compute and print RMSE
full_test_rmse = np.sqrt(np.mean((y_full_test_pred - y_full_test) ** 2))
print(f"RMSE: {full_test_rmse:+.3f}")

# Print each feature name and its coefficient
for name, coef in zip(feature_cols, full_model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# ----------------------------------------
# The negative schoolsup coefficient may appear surprising because
# educational support is intended to help students. 
# However, students receiving school support are probably those 
# who were already having academic difficulty. 
# Therefore, this coefficient does not prove that support lowers grades; 
# it likely reflects which students are selected to receive support.
# ----------------------------------------
# Train R² and test R² are close, with the test score slightly higher.
# This suggests that the model is not strongly overfitting. 
# The small difference is likely due to normal variation 
# in the random train/test split. 
# However, both scores are relatively low, so the larger problem
# is underfitting or limited predictive information rather than overfitting.
# ----------------------------------------
# For a production model, I would initially keep failures, schoolsup,
# internet, sex, goout, studytime, Walc, parental education, age, and
# absences because they show the larger coefficients or meaningful
# relationships in the exploratory analysis.
#
# I would consider dropping freetime, activities, higher, and traveltime
# because their coefficients are close to zero in this model. 
# 
# However, coefficient magnitude alone is not sufficient 
# for final feature selection because 
# the features use different numeric scales. 
# I would validate feature removal with cross-validation 
# and compare model performance before deploying it.
# ----------------------------------------

# Task 6: Evaluate and Summarize
# Create predicted vs actual scatter plot
plt.figure(figsize=(8, 6))

plt.scatter(
    y_full_test_pred,
    y_full_test,
    alpha=0.6,
    s=25,
)

# Create the diagonal reference line
min_value = min(y_full_test_pred.min(), y_full_test.min())
max_value = max(y_full_test_pred.max(), y_full_test.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    "r--",
    label="Perfect Prediction",
)

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Final Grade")
plt.ylabel("Actual Final Grade")
plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "predicted_vs_actual_g3.png")

# ----------------------------------------
# Points above the diagonal represent students whose actual grades
# were higher than predicted, meaning the model underestimated them.
# Points below the diagonal represent students whose actual grades
# were lower than predicted, meaning the model overestimated them.
#
# After examining the plot, the prediction errors appear to be 
# fairly uniform across the grade range.
# The model does not seem to struggle noticeably more with either
# low or high grades.
# 
# The model tends to predict grades closer to the middle of the
# distribution, slightly overestimating some lower grades and
# underestimating some higher grades.
# ----------------------------------------

# ----------------------------------------
# Summary:
# 
# After removing 38 students who did not take the final exam,
# the filtered dataset contained 357 students. 
# The 80/20 split produced 
# 285 training observations and 72 test observations.
#
# The full model achieved a test R2 of about 0.263, meaning it
# explained approximately 26% of the variation in final grades.
# Its RMSE was about 2.66, so a typical prediction was off by
# approximately 2.7 grade points on the 0–20 grading scale.
#
# The largest positive coefficient was internet, at approximately
# +1.04. 
# Holding the other features constant, students with internet
# access were predicted to score about one point higher.
#
# The largest negative coefficient was schoolsup, at approximately
# -2.26.
# The schoolsup result likely reflects that additional support is
# provided to students who are already struggling rather than showing
# that support causes lower grades.
# 
# The schoolsup coefficient was the most surprising result because
# educational support is intended to improve performance, but the
# negative coefficient likely reflects differences in which students
# receive that support.
# ----------------------------------------

# Neglected Feature: The Power of G1
# Add G1 to features
feature_cols_g1 = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "absences",
    "freetime",
    "goout",
    "Walc",
    "schoolsup",
    "internet",
    "higher",
    "activities",
    "sex",
    "G1",
]

# Create the new dataset
X_g1 = df_filtered[feature_cols_g1].to_numpy(dtype=float)
y_g1 = df_filtered["G3"].to_numpy(dtype=float)

# Split data into test and training sets
X_g1_train, X_g1_test, y_g1_train, y_g1_test = train_test_split(
    X_g1,
    y_g1,
    test_size=0.2,
    random_state=42,
)

# Create abd fit the model
g1_model = LinearRegression()
g1_model.fit(
    X_g1_train,
    y_g1_train,
)

# Compute and print R2
g1_test_r2 = g1_model.score(
    X_g1_test,
    y_g1_test,
)

print(f"Test R² with G1: {g1_test_r2:.3f}")

# ----------------------------------------
# A much higher R² does not mean that G1 causes G3.
# G1 and G3 measure the same student's academic performance
# at different times during the school year, so they naturally
# have a very strong relationship. 
# G1 is useful for prediction,
# but it is not the cause of the final grade.
# 
# This model would be useful for identifying students who may
# struggle later in the course. Students with low first-period
# grades could be flagged for additional support before the
# final exam.
# 
# If educators wanted to intervene before G1 was available,
# they would need to rely on information available earlier,
# such as previous failures, study habits, family background,
# early attendance patterns, and other characteristics collected
# at the beginning of the school year.
# ----------------------------------------
