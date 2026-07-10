from prefect import task, flow, get_run_logger
import pandas as pd
import numpy as np

# Given array:
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

# Define tasks
@task(name="Create Series")
def create_series(arr):
    return pd.Series(arr, name="values")

@task(
    name="Clean Data",
    retries=2,
    retry_delay_seconds=3
)
def clean_data(series):
    logger = get_run_logger()
    logger.info(f"Rows before cleaning: {len(series)}")

    cleaned_series = series.dropna()

    logger.info(f"Rows after cleaning: {len(cleaned_series)}")

    return cleaned_series

@task(name="Summarize Data")
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

# Define a flow that uses the tasks
@flow(
    name="Data Cleaning Pipeline",
    description="Clean missing values and calculate summary statistics."
)
def data_pipeline(arr):
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)
    return summary

# To run the flow
if __name__ == "__main__":  
    result = data_pipeline(arr)
    print(result)

"""
Why might Prefect be more overhead than it is worth here?

This pipeline is simple—just three small functions operating on a small dataset. Using Prefect here adds extra setup and overhead compared to calling the functions directly, so it is not necessary for a simple script like this.
"""

"""
Describe some realistic scenarios where a framework like Prefect could still be useful, even if the pipeline logic itself stays simple like in this case.

Prefect becomes valuable when working with larger or automated data pipelines. For example, it can orchestrate tasks that load data from APIs or databases, process large datasets, retry failed tasks, schedule recurring workflows, monitor execution, and log pipeline progress. Even if each individual task is simple, Prefect makes the overall workflow more reliable, maintainable, and easier to monitor in production."""