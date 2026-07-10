from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from prefect import task, flow, get_run_logger

# path to run the program from either PYTHON200-HOMEWORK folder or assignment_01 folder
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "resources" / "happiness_project"
OUTPUT_DIR = BASE_DIR / "outputs"

ALPHA = 0.05

# Rename inconsistent column names
COLUMN_MAPPING = {
    "Happiness score": "happiness_score",
    "Ladder score": "happiness_score",
    "Regional indicator": "region",
    "GDP per capita": "gdp_per_capita",
}

@task(name="Task 1: Load and Merge World Happiness Data",
    retries=3, 
    retry_delay_seconds=2
)
def load_multiple_years_data():
    logger = get_run_logger()
    OUTPUT_DIR.mkdir(exist_ok=True)

    csv_files = sorted(DATA_DIR.glob("world_happiness_*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files.")

    dataframes = []

    for file_path in csv_files:
        year = int(file_path.stem.split("_")[-1])
        logger.info(f"Loading {file_path.name}")

        df = pd.read_csv(file_path, sep=";", decimal=",",  encoding="utf-8")

        logger.info(f"{year} columns: {df.columns.tolist()}")

        df = df.rename(columns=COLUMN_MAPPING)

        df["year"] = year

        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("-", "_")
        )

        logger.info(f"{year} normalized columns: {df.columns.tolist()}")

        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)

    output_path = OUTPUT_DIR / "merged_happiness.csv"
    merged_df.to_csv(output_path, index=False)

    logger.info(f"Merged data shape: {merged_df.shape}")
    logger.info(merged_df.dtypes)
    logger.info(f"Saved merged dataset to {output_path}")

    return merged_df


@task(name="Task 2: Compute Descriptive Statistics")
def descriptive_statistics(df):
    logger = get_run_logger()

    mean_happiness_overall = df["happiness_score"].mean()
    median_happiness_overall = df["happiness_score"].median()
    std_happiness_overall = df["happiness_score"].std()

    logger.info(f"Overall mean happiness score: {mean_happiness_overall:.2f}")
    logger.info(f"Overall median happiness score: {median_happiness_overall:.2f}")
    logger.info(f"Overall standard deviation: {std_happiness_overall:.2f}")

    mean_happiness_by_year = df.groupby("year")["happiness_score"].mean()
    logger.info(f"Mean happiness by year:\n{mean_happiness_by_year}")

    mean_happiness_by_region = (
        df.groupby("region")["happiness_score"]
          .mean()
          .sort_values(ascending=False)
    )
    logger.info(f"Mean happiness by region:\n{mean_happiness_by_region}")

    return {
        "overall_mean": mean_happiness_overall, 
        "overall_median": median_happiness_overall, 
        "overall_std": std_happiness_overall, 
        "by_year": mean_happiness_by_year, 
        "by_region": mean_happiness_by_region
    }

@task(name="Task 3: Generate Visualizations")
def visual_exploration(df):
    logger = get_run_logger()

    plt.figure()
    plt.hist(df["happiness_score"].dropna(), bins=20)
    plt.title("Distribution of Happiness Scores")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")
    plt.savefig(OUTPUT_DIR / "happiness_histogram.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved happiness_histogram.png")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="year", y="happiness_score")
    plt.title("Happiness Scores by Year")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    plt.xticks(rotation=45)
    plt.savefig(OUTPUT_DIR / "happiness_by_year.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved happiness_by_year.png")

    plt.figure()
    sns.scatterplot(data=df, x="gdp_per_capita", y="happiness_score")
    plt.title("GDP per Capita vs Happiness Score")
    plt.xlabel("GDP per Capita")
    plt.ylabel("Happiness Score")
    plt.savefig(OUTPUT_DIR / "gdp_vs_happiness.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved gdp_vs_happiness.png")

    numeric_df = df.select_dtypes(include="number")
    correlation_matrix = numeric_df.corr(method="pearson")

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", bbox_inches="tight")
    plt.close()
    logger.info("Saved correlation_heatmap.png")

@task(name="Task 4: Run Hypothesis Tests")
def hypothesis_testing(df):
    logger = get_run_logger()

    scores_2019 = df[df["year"] == 2019]["happiness_score"].dropna()
    scores_2020 = df[df["year"] == 2020]["happiness_score"].dropna()

    t_stat, p_value = stats.ttest_ind(scores_2019, scores_2020)

    mean_2019 = scores_2019.mean()
    mean_2020 = scores_2020.mean()

    logger.info(f"2019 mean happiness score: {mean_2019:.2f}")
    logger.info(f"2020 mean happiness score: {mean_2020:.2f}")
    logger.info(f"2019 vs 2020 t-statistic: {t_stat:.4f}")
    logger.info(f"2019 vs 2020 p-value: {p_value:.4f}")

    if p_value < ALPHA:
        if mean_2020 < mean_2019:
            interpretation = "The average global happiness score was lower in 2020 than in 2019. This decrease is statistically significant, suggesting that happiness changed after the pandemic began."
        else:
            interpretation = "The average global happiness score was higher in 2020 than in 2019.This increase is statistically significant."
    else:
        interpretation = "The average happiness scores for 2019 and 2020 differ slightly, but the difference is not statistically significant. Based on this data, we cannot conclude that the pandemic changed global happiness scores."

    logger.info(interpretation)

    regions = df["region"].dropna().unique()

    region_a = regions[0]
    region_b = regions[1]

    scores_a = df[df["region"] == region_a]["happiness_score"].dropna()
    scores_b = df[df["region"] == region_b]["happiness_score"].dropna()

    region_t, region_p = stats.ttest_ind(scores_a, scores_b)

    logger.info(f"Second test: {region_a} vs {region_b}")
    logger.info(f"{region_a} mean: {scores_a.mean():.2f}")
    logger.info(f"{region_b} mean: {scores_b.mean():.2f}")
    logger.info(f"t-statistic: {region_t:.4f}")
    logger.info(f"p-value: {region_p:.4f}")

    return {
        "mean_2019": mean_2019,
        "mean_2020": mean_2020,
        "t_stat": t_stat,
        "p_value": p_value,
        "interpretation": interpretation,
    }

@task(name="Task 5: Analyze Correlations")
def correlation_analysis(df):
    logger = get_run_logger()

    numeric_df = df.select_dtypes(include="number").drop(columns=["year"], errors="ignore")

    results = []

    for column in numeric_df.columns:
        if column == "happiness_score":
            continue

        test_df = df[["happiness_score", column]].dropna()

        correlation, p_value = stats.pearsonr(
            test_df["happiness_score"],
            test_df[column],
        )

        results.append(
            {
                "variable": column,
                "correlation": correlation,
                "p_value": p_value,
            }
        )

        logger.info(
            f"{column}: correlation={correlation:.4f}, p-value={p_value:.4f}"
        )

    results_df = pd.DataFrame(results)

    number_of_tests = len(results_df)
    adjusted_alpha = ALPHA / number_of_tests

    logger.info(f"Number of correlation tests: {number_of_tests}")
    logger.info(f"Bonferroni adjusted alpha: {adjusted_alpha:.6f}")

    significant_original = results_df[results_df["p_value"] < ALPHA]
    significant_adjusted = results_df[results_df["p_value"] < adjusted_alpha]

    logger.info(f"Significant at alpha=0.05:\n{significant_original}")
    logger.info(f"Significant after Bonferroni correction:\n{significant_adjusted}")

    output_path = OUTPUT_DIR / "correlation_results.csv"
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved correlation results to {output_path}")

    return {
        "results": results_df,
        "adjusted_alpha": adjusted_alpha
    }

@task(name="Task 6: Generate Summary Report")
def summary_report(df, descriptive_results, ttest_results, correlation_results):
    logger = get_run_logger()

    total_countries = df["country"].nunique()
    total_years = df["year"].nunique()

    region_means = descriptive_results["by_region"]
    top_regions = region_means.head(3)
    bottom_regions = region_means.tail(3).sort_values()

    results_df = correlation_results["results"]
    adjusted_alpha = correlation_results["adjusted_alpha"]

    significant_corrs = results_df[results_df["p_value"] < adjusted_alpha]

    if not significant_corrs.empty:
        strongest = significant_corrs.loc[
            significant_corrs["correlation"].abs().idxmax()
        ]
        strongest_variable = strongest["variable"]
        strongest_correlation = strongest["correlation"]
    else:
        strongest_variable = None
        strongest_correlation = None

    logger.info(
        f"Dataset includes {total_countries} countries across {total_years} years."
    )
    for rank, (region, score) in enumerate(top_regions.items(), start=1):
        logger.info(
            f"Top region #{rank}: {region} "
            f"with a mean happiness score of {score:.2f}"
        )
    for rank, (region, score) in enumerate(bottom_regions.items(), start=1):
        logger.info(
            f"Bottom region #{rank}: {region} "
            f"with a mean happiness score of {score:.2f}"
        )
    logger.info(f"Pre/post-2020 test result: {ttest_results['interpretation']}")

    if strongest_correlation is not None:
        logger.info(
            "Strongest significant correlation after Bonferroni correction: "
            f"{strongest_variable} with correlation {strongest_correlation:.4f}"
        )
    else:
        logger.info("No correlations remained statistically significant after Bonferroni correction.")

@flow(name="World Happiness Pipeline")
def happiness_pipeline():
    df = load_multiple_years_data()
    descriptive_results = descriptive_statistics(df)
    visual_exploration(df)
    ttest_results = hypothesis_testing(df)
    correlation_results = correlation_analysis(df)
    summary_report(df, descriptive_results, ttest_results, correlation_results)


if __name__ == "__main__":
    happiness_pipeline()