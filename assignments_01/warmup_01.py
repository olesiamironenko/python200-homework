#  --- Pandas Review ---

print("--- Pandas Review ---")
print("\n")

# Import pandas
import pandas as pd

# --- Pandas Question 1 --- 
print("Pandas Question 1:")

# Task:
# Create the following DataFrame and print the first three rows, the shape, and the data types of each column. Print each result with a label (e.g. print(f"Num Rows: {len(df)}")).

# Given data:
data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

# Solution:
# Display the first 3 rows of the DataFrame
print("Display the first 3 rows of the DataFrame:")
print(df.head(3))
print("\n")

# Display dataframe shape
print("Display dataframe shape:")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print("\n")

# Display the data types for each column of the DataFrame
print("Data type each column of the DataFrame:")
for column in df.columns:
    print(f"Data type of the column {column}: {df[column].dtype}")
print("\n")


# --- Pandas Question 2 ---
print("Pandas Question 2:")

# Task:
# Using the DataFrame from Q1, filter the rows to show only students who passed and have a grade above 80. Print the result.

# Solution:
# Filter the DataFrame to include only students who passed and have a grade above 80
filtered_df = df[(df['passed'] == True) & (df['grade'] > 80)]

# Print the result
print("Students who passed and have a grade above 80:")
print(filtered_df)
print("\n")


# --- Pandas Question 3 ---
print("Pandas Question 3:")

# Task:
# Add a new column called "grade_curved" that adds 5 points to each student's grade. Print the updated DataFrame (all columns, all rows)

# Solution:
# Add a new column called "grade_curved" that adds 5 points to each student's grade
df['grade_curved'] = df['grade'] + 5

# Print the updated DataFrame (all columns, all rows)
print("Updated DataFrame with the new column 'grade_curved':")
print(df)
print("\n")


# --- Pandas Question 4 ---
print("Pandas Question 4:")

# Task:
# Add a new column called "name_upper" that contains each student's name in uppercase, using the .str accessor. Print the "name" and "name_upper" columns together.

# Solution:
# Add a new column called "name_upper" that contains each student's name in uppercase, using the .str accessor
df['name_upper'] = df['name'].str.upper()

# Print the "name" and "name_upper" columns together
print("DataFrame with 'name' and 'name_upper' columns:")
print(df[['name', 'name_upper']])
print("\n")


# --- Pandas Question 5 ---
print("Pandas Question 5:")

# Task:
# Group the DataFrame by "city" and compute the mean grade for each city. Print the result.

# Solution:
# Group the DataFrame by "city" and compute the mean grade for each city.
city_mean_grades = df.groupby('city')['grade'].mean()

# Print the result
print("Mean grade for each city:")
print(city_mean_grades)
print("\n")


# --- Pandas Question 6 ---
print("Pandas Question 6:")

# Task:
# Replace the value "Austin" in the "city" column with "Houston". Print the "name" and "city" columns to confirm the change.

# Solution:
# Replace the value "Austin" in the "city" column with "Houston"
df['city'] = df['city'].replace('Austin', 'Houston')

# Print the "name" and "city" columns to confirm the change
print("DataFrame with updated 'city' column:")
print(df[['name', 'city']])
print("\n")


# --- Pandas Question 7 ---
print("Pandas Question 7:")

# Task:
# Sort the DataFrame by "grade" in descending order and print the top 3 rows.

# Solution:
# Sort the DataFrame by "grade" in descending order
df_sorted = df.sort_values(by='grade', ascending=False)

# Print the top 3 rows of the sorted DataFrame
print("Top 3 students by grade:")
print(df_sorted.head(3))
print("\n")


#  --- NumPy Review ---

print("--- NumPy Review ---")
print("\n")

# Import NumPy
import numpy as np

#  --- NumPy Question 1 ---
print("NumPy Question 1:")

# Task:
# Create a 1D NumPy array from the list [10, 20, 30, 40, 50]. Print its shape, dtype, and ndim.

# Solution:
# Create a 1D NumPy array from the list [10, 20, 30, 40, 50]
arr = np.array([10, 20, 30, 40, 50])

# Print its shape, dtype, and ndim
print(f"Shape of the array: {arr.shape}")
print(f"Data type of the array: {arr.dtype}")
print(f"Number of dimensions of the array: {arr.ndim}")
print("\n")

#  --- NumPy Question 2 ---
print("NumPy Question 2:")

# Task:
# Create the following 2D array and print its shape and size (total number of elements).

# Given:
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# Solution:
# Print its shape and size
print(f"Shape of the array: {arr.shape}")
print(f"Size of the array: {arr.size}")
print("\n")

#  --- NumPy Question 3 ---
print("NumPy Question 3:")

# Task:
# Using the 2D array from Q2, slice out the top-left 2x2 block and print it. The expected result is [[1, 2], [4, 5]].

# Solution:
# Using the 2D array from Q2, slice out the top-left 2x2
slice_arr = arr[:2, :2]

# Print the sliced array
print("Top-left 2x2 slice of the array:")
print(slice_arr)
print("\n")

#  --- NumPy Question 4 ---
print("NumPy Question 4:")

# Task:
# Create a 3x4 array of zeros using a built-in command. Then create a 2x5 array of ones using a built-in command. Print both.

# Solution:
# Create a 3x4 array of zeros using a built-in command
arr_zeros = np.zeros((3, 4))

# Create a 2x5 array of ones using a built-in command
arr_ones = np.ones((2, 5))

# Print the array
print("Array of zeros:")
print(arr_zeros)
print("\n")

print("Array of ones:")
print(arr_ones)
print("\n")

# --- NumPy Question 5 ---
print("NumPy Question 5:")

# Task:
# Create an array using np.arange(0, 50, 5). First, think about what you expect it to look like. Then, print the array, its shape, mean, sum, and standard deviation.

# Solution:
# Create an array using np.arange(0, 50, 5)
arr_range = np.arange(0, 50, 5)

# Print the array, its shape, mean, sum, and standard deviation
print("Array created with np.arange(0, 50, 5):")
print(arr_range)
print(f"Shape: {arr_range.shape}")
print(f"Mean: {np.mean(arr_range)}")
print(f"Sum: {np.sum(arr_range)}")
print(f"Standard Deviation: {np.std(arr_range)}")
print("\n")

# --- NumPy Question 6 ---
print("NumPy Question 6:")

# Task:
# Generate an array of 200 random values drawn from a normal distribution with mean 0 and standard deviation 1 (use np.random.normal()). Print the mean and standard deviation of the result.

# Solution:
# Generate an array of 200 random values drawn from a normal distribution with mean 0 and standard deviation 1 (use np.random.normal())
arr_normal = np.random.normal(0, 1, 200)

# Print the mean and standard deviation of the result
print(f"Mean: {np.mean(arr_normal)}")
print(f"Standard Deviation: {np.std(arr_normal)}")
print("\n")


#  --- Matplotlib Review ---
print("--- Matplotlib Review ---")
print("\n")

#  --- Matplotlib Question 1 ---
print("Matplotlib Question 1:")
print("See the line plot 'Squares' in the pop-up window.")
print("\n")

import matplotlib.pyplot as plt

# Task:
# Plot the following data as a line plot. Add a title "Squares", x-axis label "x", and y-axis label "y".

# Given:
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

# Solution:
# Plot the data as a line plot
plt.plot(x, y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# --- Matplotlib Question 2 ---
print("Matplotlib Question 2:")
print("See the bar plot 'Subject Scores' in the pop-up window.")
print("\n")

# Task:
# Create a bar plot for the following subject scores. Add a title "Subject Scores" and label both axes.

# Given:
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]

# Solution:
# Create a bar plot for the subject scores
plt.bar(subjects, scores)
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.show()


# --- Matplotlib Question 3 ---
print("Matplotlib Question 3:")
print("See 'Scatter Plot of Two Datasets in the pop-up window.")
print("\n")

# Task:
# Plot the two datasets below as a scatter plot on the same figure. Use different colors for each, add a legend, and label both axes.

# Given:
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

# Solution:
# Plot the two datasets as a scatter plot on the same figure
plt.scatter(x1, y1, color='blue', label='Dataset 1')
plt.scatter(x2, y2, color='red', label='Dataset 2')
plt.title("Scatter Plot of Two Datasets")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()


# --- Matplotlib Question 4 ---
print("Matplotlib Question 4:")
print("See 'Squares' and 'Subject Scores' subplots in the pop-up window.")
print("\n")

# Task:
# Use plt.subplots() to create a figure with 1 row and 2 subplots side by side. In the left subplot, plot x vs y from Q1 as a line. In the right subplot, plot the subjects and scores from Q2 as a bar plot. Add a title to each subplot and call plt.tight_layout() before showing

# Solution:
# Create a figure with 1 row and 2 subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Left subplot: Line plot
ax1.plot(x, y)
ax1.set_title("Squares")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# Right subplot: Bar plot
ax2.bar(subjects, scores)
ax2.set_title("Subject Scores")
ax2.set_xlabel("Subjects")
ax2.set_ylabel("Scores")

plt.tight_layout()
plt.show()


# --- Descriptive Statistics Review ---
print("--- Descriptive Statistics Review ---")
print("\n")

# --- Descriptive Stats Question 1 ---
print("Descriptive Stats Question 1:")

# Task:
# Given the list below, use NumPy to compute and print the mean, median, variance, and standard deviation. Label each printed value.

# Given:
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]

# Solution:
# Use NumPy to compute and print the mean, median, variance, and standard deviation
data_np = np.array(data)
print(f"Mean: {np.mean(data_np)}")
print(f"Median: {np.median(data_np)}")
print(f"Variance: {np.var(data_np)}")
print(f"Standard Deviation: {np.std(data_np)}")
print("\n")

# --- Descriptive Stats Question 2 ---
print("Descriptive Stats Question 2:")
print("See the histogram 'Distribution of Scores' in the pop-up window.")
print("\n")

# Task:
# Generate 500 random values from a normal distribution with mean 65 and standard deviation 10 (use np.random.normal(65, 10, 500)). Plot a histogram with 20 bins. Add a title "Distribution of Scores" and label both axes.

# Solution:
# Generate random values
data_normal = np.random.normal(65, 10, 500)

# Plot a histogram with 20 bins. Add a title "Distribution of Scores" and label both axes.
plt.hist(data_normal, bins=20)
plt.title("Distribution of Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()


# --- Descriptive Stats Question 3 ---
print("Descriptive Stats Question 3:")
print("See the boxplot 'Score Comparison' in the pop-up window.")
print("\n")

# Task:
# Create a boxplot comparing the two groups below. Label each box ("Group A" and "Group B") and add a title "Score Comparison"

# Given:
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88] 

# Solution:
# Create a boxplot comparing the two groups. Label each box ("Group A" and "Group B") and add a title "Score Comparison"
plt.boxplot([group_a, group_b], tick_labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.ylabel("Scores")
plt.show()


# --- Descriptive Stats Question 4 ---
print("Descriptive Stats Question 4:")
print("See the boxplot 'Distribution Comparison' in the pop-up window.")
print("\n")

# Task:
# You are given two datasets: one normally distributed and one 'exponential' distribution.
# Create side-by-side boxplots comparing the two distributions. Label each boxplot appropriately ("Normal" and "Exponential") and add a title "Distribution Comparison".
# Then, add a comment in your code briefly noting which distribution is more skewed, and which descriptive statistic (mean or median) would provide a more appropriate measure of central tendency for each distribution.

# Given:
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

# Solution:
# Create side-by-side boxplots comparing the two distributions. Label each boxplot appropriately ("Normal" and "Exponential") and add a title "Distribution Comparison"
plt.boxplot([normal_data, skewed_data], tick_labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.ylabel("Values")
plt.show()

# Comment:
print("According to the box plots, the exponential distribution is more skewed than the normal distribution, as evidenced by the longer whisker and the presence of outliers in the exponential boxplot.")
print("Median is a more appropriate measure of central tendency in this case because the box plot shows data skewness with many outliers on one side, and the median is less affected by extreme values compared to the mean.")
print("\n")

# --- Descriptive Stats Question 5 ---
print("Descriptive Stats Question 5:")

# import stats module from scipy
from scipy import stats

# Task:
# Print the mean, median, and mode of the following data.
# Why are the median and mean so different for data2? Add your answer as a comment in the code.

# Given:
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

# Solution:
print(f"Data 1 - Mean: {np.mean(data1)}, \nMedian: {np.median(data1)}, \nMode: {stats.mode(data1).mode.item()}")
print(f"Data 2 - Mean: {np.mean(data2)}, \nMedian: {np.median(data2)}, \nMode: {stats.mode(data2).mode.item()}")

# Comment:
print("Mean shows the average value of the data, median shows the middle value when the data is sorted. If the data have extreme values, the mean is skewed by those values. The median is more robust to outliers. So the more extreme values data have the bigger difference will be between mean and median.")
print("\n")

# --- Hypothesis Testing Review ---
print("--- Hypothesis Testing Review ---")
print("\n")

# --- Hypothesis Testing Question 1 ---
print("Hypothesis Testing Question 1:")

# Task:
# Run an independent samples t-test on the two groups below. Print the t-statistic and p-value.

# Given:
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

# Solution:
# Run an independent samples t-test on the two groups
t_stat, p_value = stats.ttest_ind(group_a, group_b)

# Print the t-statistic and p-value
print(f"T-statistic: {t_stat}, \nP-value: {p_value}")
print("\n")

# --- Hypothesis Testing Question 2 ---
print("Hypothesis Testing Question 2:")

# Task:
# Using the p-value from Q1, write an if/else statement that prints whether the result is statistically significant at alpha = 0.05

# Solution:
alpha = 0.05
if p_value < alpha:
    print("The result is statistically significant.")
else:
    print("The result is not statistically significant.")
print("\n")

# --- Hypothesis Testing Question 3 ---
print("Hypothesis Testing Question 3:")

# Task:
# Run a paired t-test on the before/after scores below (the same students measured twice). Print the t-statistic and p-value.

# Given:
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

# Solution:
# Run a paired t-test on the before/after scores
t_stat, p_value = stats.ttest_rel(before, after)

# Print the t-statistic and p-value
print(f"T-statistic: {t_stat}, \nP-value: {p_value}")
print("\n")

# --- Hypothesis Testing Question 4 ---
print("Hypothesis Testing Question 4:")

# Task:
# Run a one-sample t-test to check whether the mean of scores is significantly different from a national benchmark of 70. Print the t-statistic and p-value.

# Given:
scores = [72, 68, 75, 70, 69, 74, 71, 73]

# Solution
# Run a one-sample t-test on scores dataset
t_stat, p_value = stats.ttest_1samp(scores, 70)

# Print the t-statistic and p-value
print(f"T-statistic: {t_stat}, \nP-value: {p_value}")
print("\n")

# --- Hypothesis Testing Question 5 ---
print("Hypothesis Testing Question 5:")

# Task:
# Re-run the test from Q1 as a one-tailed test to check whether group_a scores are less than group_b scores. Print the resulting p-value. Use the alternative parameter.

# Solution:
# Re-run the test from Q1 as a one-tailed test using the alternative parameter
t_stat, p_value = stats.ttest_ind(group_a, group_b, alternative='less')

# Print the resulting p-value
print(f"P-value: {p_value}")
print("\n")

# --- Hypothesis Testing Question 6 ---
print("Hypothesis Testing Question 6:")

# Task:
# Write a plain-language conclusion for the result of Q1 (do not just say "reject the null hypothesis"). Format it as a print() statement. Your conclusion should mention the direction of the difference and whether it is likely due to chance.

# Solution:
print("Conclusion: The independent samples t-test indicates that there is a statistically significant difference between the scores of group_a and group_b. Given the low p-value, it is unlikely that this difference is due to random chance, suggesting that the observed difference reflects a true effect.")

# --- Correlation Review ---
print("Correlation Review:")
print("\n")

# --- Correlation Question 1 ---
print("Correlation Question 1:")

# Task:
# Compute the Pearson correlation between x and y below using np.corrcoef(). Print the full correlation matrix, then print just the correlation coefficient (the value at position [0, 1])

# Given:
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Solution:
# Compute the correlation matrix
correlation_matrix = np.corrcoef(x, y)

# Extract the correlation coefficient
correlation_coefficient = correlation_matrix[0, 1]

# Print the results
print(f"Correlation matrix: \n{correlation_matrix}")
print(f"Correlation coefficient: {correlation_coefficient}")
print("\n")

# What do you expect the correlation to be, and why?
print("I expect the correlation to be 1.0, as the relationship between x and y is perfectly linear with a positive slope.")
print("\n")

# --- Correlation Question 2 ---
print("Correlation Question 2:")

# Task:
# Use pearsonr() from scipy.stats to compute the correlation between x and y below. Print both the correlation coefficient and the p-value.

# Given:
x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

# Solution:
# Compute the Pearson correlation and p-value
correlation_coefficient, p_value = stats.pearsonr(x, y)

# Print the results
print(f"Correlation coefficient: {correlation_coefficient}")
print(f"P-value: {p_value}")
print("\n")

# --- Correlation Question 3 ---
print("Correlation Question 3:")

# Task:
# Create the following DataFrame and use df.corr() to compute the correlation matrix. Print the result.

# Given:
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)

# Solution:
# Compute the correlation matrix
correlation_matrix = df.corr()

# Print the correlation matrix
print(f"Correlation matrix:\n{correlation_matrix}")
print("\n")

# --- Correlation Question 4 ---
print("Correlation Question 4:")
print("See the scatter plot 'Negative Correlation' in the pop-up window.")
print("\n")

# Task:
# Create a scatter plot of x and y below, which have a negative relationship. Add a title "Negative Correlation" and label both axes.

# Given:
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

# Solution:
# Create the scatter plot
plt.scatter(x, y)
plt.title("Negative Correlation")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()


# --- Correlation Question 5 ---
print("Correlation Question 5:")
print("See the correlation heatmap 'Correlation Heatmap' in the pop-up window.")
print("\n")

# Task:
# Using the correlation matrix from Q3, create a heatmap with sns.heatmap(). Pass annot=True so the correlation values appear in each cell, and add a title "Correlation Heatmap".

# Import seaborn
import seaborn as sns

# Solution:
# Create the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap")
plt.show()


# --- Piplines ---
print("--- Piplines ---")
print("\n")

# --- Pipeline Question 1 ---
print("Pipeline Question 1:")

# Task:
# Given the array below, which contains some missing values scattered throughout, implement the following three functions and then connect them in a data_pipeline() function.
# 1. create_series(arr) : takes a NumPy array and returns a pandas Series with the name "values".
# 2. clean_data(series) : takes the Series, removes any NaN values using .dropna(), and returns the cleaned Series.
# 3. summarize_data(series) -- takes the cleaned Series and returns a dictionary with four keys: "mean", "median", "std", and "mode". For mode, use series.mode()[0] to get a single value.

# data_pipeline(arr) -- calls the three functions above in sequence and returns the summary dictionary.

# Call data_pipeline(arr) and print each key and its value from the result.

# Given:
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

# Solution:
# 1.
def create_series(arr):
    """Takes a NumPy array and returns a pandas Series with the name 'values'."""
    return pd.Series(arr, name='values')

# 2.
def clean_data(series):
    """Takes a Series, removes any NaN values using .dropna(), and returns the cleaned Series."""
    return series.dropna()

# 3.
def summarize_data(series):
    """Takes the cleaned Series and returns a dictionary with mean, median, std, and mode."""
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

# Calls the three functions above in sequence and returns the summary dictionary
def data_pipeline(arr):
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary_dictionary = summarize_data(cleaned_series)
    return summary_dictionary

# Call data_pipeline(arr) and print each key and its value from the result.
summary = data_pipeline(arr)
print("Summary of the cleaned data:")
for key, value in summary.items():
    print(f"{key}: {value}")
print("\n")

