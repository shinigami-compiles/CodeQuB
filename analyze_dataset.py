#!/usr/bin/env python3
"""
analyze_dataset.py

Utility script to inspect and validate:
"code_quality_bug_risk_dataset_15k_balanced.csv"

What it does:
- Loads the dataset
- Prints:
    * shape, dtypes, NA counts
    * bug_risk_class distribution
    * summary stats for code_quality_score & bug_risk_score
- Computes:
    * feature correlation matrix and saves it as CSV
    * correlation heatmap (matplotlib) and saves as PNG
- Analyzes:
    * code_quality_score and bug_risk_score by bug_risk_class
    * simple boxplot of quality vs bug_risk_class

Usage:
    python analyze_dataset.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------

DATA_DIR = "data"
DATA_CSV = os.path.join(DATA_DIR, "code_quality_bug_risk_dataset_15k_balanced_v2.csv")

OUTPUT_DIR = "analysis_outputs"
CORR_CSV = os.path.join(OUTPUT_DIR, "feature_correlations.csv")
CORR_PNG = os.path.join(OUTPUT_DIR, "feature_correlation_heatmap.png")
BOXPLOT_PNG = os.path.join(OUTPUT_DIR, "quality_vs_bug_class_boxplot.png")


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            f"Run generate_dataset.py first."
        )
    df = pd.read_csv(path)
    return df


def basic_info(df: pd.DataFrame) -> None:
    print("\n=== BASIC INFO ===")
    print(f"Shape: {df.shape}")
    print("\nColumns and dtypes:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isna().sum())


def class_distribution(df: pd.DataFrame) -> None:
    print("\n=== BUG RISK CLASS DISTRIBUTION ===")
    print(df["bug_risk_class"].value_counts().sort_index())


def score_summaries(df: pd.DataFrame) -> None:
    print("\n=== CODE QUALITY SCORE SUMMARY ===")
    print(df["code_quality_score"].describe())

    print("\n=== BUG RISK SCORE SUMMARY ===")
    print(df["bug_risk_score"].describe())


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix for all numeric columns
    and return it.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(method="pearson")
    return corr


def save_correlation_outputs(corr: pd.DataFrame) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save CSV
    corr.to_csv(CORR_CSV, float_format="%.4f")
    print(f"\nCorrelation matrix saved to: {CORR_CSV}")

    # Plot heatmap using matplotlib only (no seaborn)
    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("Feature Correlation Heatmap", fontsize=12, pad=12)

    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(CORR_PNG, dpi=200)
    plt.close(fig)
    print(f"Correlation heatmap saved to: {CORR_PNG}")


def scores_by_class(df: pd.DataFrame) -> None:
    """
    Show how code_quality_score and bug_risk_score behave per bug_risk_class.
    """
    print("\n=== SCORES BY BUG RISK CLASS ===")
    grouped = df.groupby("bug_risk_class")[["code_quality_score", "bug_risk_score"]]
    summary = grouped.agg(["mean", "std", "min", "max"])
    print(summary)


def save_quality_boxplot(df: pd.DataFrame) -> None:
    """
    Boxplot: code_quality_score vs bug_risk_class.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    classes = sorted(df["bug_risk_class"].unique())
    data_per_class = [df[df["bug_risk_class"] == c]["code_quality_score"] for c in classes]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data_per_class, labels=classes, showfliers=True)
    ax.set_xlabel("bug_risk_class")
    ax.set_ylabel("code_quality_score")
    ax.set_title("Code Quality Score vs Bug Risk Class")

    plt.tight_layout()
    fig.savefig(BOXPLOT_PNG, dpi=200)
    plt.close(fig)
    print(f"Quality vs bug class boxplot saved to: {BOXPLOT_PNG}")


def main() -> None:
    print(f"Loading dataset from: {DATA_CSV}")
    df = load_dataset(DATA_CSV)

    # Basic checks
    basic_info(df)
    class_distribution(df)
    score_summaries(df)

    # Correlations
    corr = compute_correlations(df)
    save_correlation_outputs(corr)

    # Behavior of scores across classes
    scores_by_class(df)
    save_quality_boxplot(df)

    print("\nAnalysis complete. Check the 'analysis_outputs/' folder for plots and CSVs.")


if __name__ == "__main__":
    main()
