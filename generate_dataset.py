#!/usr/bin/env python3
"""
generate_dataset.py (v2)

Synthetic dataset generator for:
"ANN-Based Code Quality & Bug Risk Prediction System Using Static & Process Metrics"

This version is tuned for BETTER LEARNABILITY:
- Lower noise in bug_risk_score (less random label flipping).
- Stronger influence of bug-related metrics on bug_risk_score.
- Wider, cleaner thresholds for bug_risk_class (6 classes).
- Approximate class balance via down/oversampling.

Outputs:
    data/code_quality_bug_risk_dataset_15k_balanced.csv

Columns:
    - 22 feature columns (numeric)
    - code_quality_score (0–100, float)
    - bug_risk_score (0–100, float)
    - bug_risk_class (0–5, int)
"""

import os
from typing import Tuple, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
DEFAULT_NUM_SAMPLES = 15000  # 15k rows
OUTPUT_DIR = "data"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "code_quality_bug_risk_dataset_15k_balanced_v2.csv")

NUM_CLASSES = 6

# List of all 22 feature names in a fixed, consistent order
FEATURE_COLUMNS = [
    # Structural / quality-related
    "loc",
    "num_functions",
    "avg_function_length",
    "cyclomatic_complexity",
    "max_nesting_depth",
    "max_function_complexity",
    "std_dev_function_complexity",
    "lint_warning_count",
    "comment_to_code_ratio",
    "review_comment_count",
    "test_coverage_percent",
    "num_test_files_linked",
    "code_churn_recent",
    "num_contributors",
    "file_age_days",
    "fan_in",
    # Bug / instability-related
    "bug_count_total",
    "bug_density",
    "bugs_last_30_days",
    "bug_trend",
    "test_fail_count_related",
    "days_since_last_change",
]


# ---------------------------------------------------------------------------
# Synthetic feature generation
# ---------------------------------------------------------------------------

def generate_features(num_samples: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate realistic synthetic values for all 22 features.

    Returns:
        DataFrame with shape (num_samples, 22) and columns = FEATURE_COLUMNS.
    """

    # Structural / quality-related features
    loc = rng.integers(50, 3000, size=num_samples)  # lines of code
    num_functions = rng.integers(1, 80, size=num_samples)
    avg_function_length = np.clip(
        rng.normal(loc=30, scale=15, size=num_samples), 5, 200
    )
    cyclomatic_complexity = rng.integers(1, 31, size=num_samples)
    max_nesting_depth = rng.integers(1, 9, size=num_samples)
    max_function_complexity = rng.integers(1, 51, size=num_samples)
    std_dev_function_complexity = np.clip(
        rng.normal(loc=3, scale=2, size=num_samples), 0, 15
    )
    lint_warning_count = rng.integers(0, 201, size=num_samples)
    comment_to_code_ratio = np.clip(
        rng.normal(loc=0.2, scale=0.1, size=num_samples), 0.0, 0.6
    )
    review_comment_count = rng.integers(0, 101, size=num_samples)
    test_coverage_percent = np.clip(
        rng.normal(loc=65, scale=20, size=num_samples), 0, 100
    )
    num_test_files_linked = rng.integers(0, 11, size=num_samples)
    code_churn_recent = rng.integers(0, 501, size=num_samples)
    num_contributors = rng.integers(1, 16, size=num_samples)
    file_age_days = rng.integers(1, 3651, size=num_samples)
    fan_in = rng.integers(0, 51, size=num_samples)

    # Bug / instability-related features
    bug_count_total = rng.integers(0, 101, size=num_samples)
    bug_density = np.clip(
        rng.normal(loc=1.0, scale=0.8, size=num_samples), 0.0, 5.0
    )  # bugs per KLOC
    bugs_last_30_days = rng.integers(0, 21, size=num_samples)
    bug_trend = rng.choice([-1, 0, 1], size=num_samples)
    test_fail_count_related = rng.integers(0, 31, size=num_samples)
    days_since_last_change = rng.integers(0, 366, size=num_samples)

    data = {
        "loc": loc,
        "num_functions": num_functions,
        "avg_function_length": avg_function_length,
        "cyclomatic_complexity": cyclomatic_complexity,
        "max_nesting_depth": max_nesting_depth,
        "max_function_complexity": max_function_complexity,
        "std_dev_function_complexity": std_dev_function_complexity,
        "lint_warning_count": lint_warning_count,
        "comment_to_code_ratio": comment_to_code_ratio,
        "review_comment_count": review_comment_count,
        "test_coverage_percent": test_coverage_percent,
        "num_test_files_linked": num_test_files_linked,
        "code_churn_recent": code_churn_recent,
        "num_contributors": num_contributors,
        "file_age_days": file_age_days,
        "fan_in": fan_in,
        "bug_count_total": bug_count_total,
        "bug_density": bug_density,
        "bugs_last_30_days": bugs_last_30_days,
        "bug_trend": bug_trend,
        "test_fail_count_related": test_fail_count_related,
        "days_since_last_change": days_since_last_change,
    }

    df = pd.DataFrame(data, columns=FEATURE_COLUMNS)
    return df


# ---------------------------------------------------------------------------
# Label generation (heuristics)
# ---------------------------------------------------------------------------

def compute_code_quality_score(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """
    Compute code_quality_score in [0, 100] based on structural and quality metrics.

    Heuristic behavior:
        - Decreases with higher complexity, lint warnings, long functions, high churn,
          too many contributors, high fan_in.
        - Increases with better comments and higher test coverage.

    Produces a decent spread of quality values while still respecting intuitive
    relationships.
    """

    # Normalize some features into [0, 1] ranges for convenience
    cc_norm = df["cyclomatic_complexity"] / 30.0
    max_cc_norm = df["max_function_complexity"] / 50.0
    nesting_norm = df["max_nesting_depth"] / 8.0
    lint_norm = df["lint_warning_count"] / 200.0
    avg_fn_len_norm = df["avg_function_length"] / 200.0
    comments_norm = df["comment_to_code_ratio"] / 0.6
    coverage_norm = df["test_coverage_percent"] / 100.0
    churn_norm = df["code_churn_recent"] / 500.0
    contributors_norm = df["num_contributors"] / 15.0
    fan_in_norm = df["fan_in"] / 50.0

    # Base quality and adjustments
    quality = 75.0
    quality -= 18.0 * cc_norm
    quality -= 12.0 * max_cc_norm
    quality -= 8.0 * nesting_norm
    quality -= 15.0 * lint_norm
    quality -= 10.0 * avg_fn_len_norm
    quality += 15.0 * comments_norm
    quality += 22.0 * coverage_norm
    quality -= 7.0 * churn_norm
    quality -= 6.0 * contributors_norm
    quality -= 7.0 * fan_in_norm

    # Add small Gaussian noise for realism
    noise = rng.normal(loc=0.0, scale=5.0, size=len(df))
    quality = quality + noise

    # Clip to [0, 100]
    quality = np.clip(quality, 0.0, 100.0)
    return quality


def compute_bug_risk_score(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """
    Compute bug_risk_score in [0, 100] based on bug and instability-related metrics.

    v2 design goals:
        - Stronger influence of bug-specific metrics.
        - Moderate influence of churn, ownership, complexity.
        - Strong negative influence of review_comment_count.
        - Much lower Gaussian noise -> easier, more learnable labels.

    Main behavior:
        - Increases with bug counts, density, recent bugs, positive bug trend,
          test failures, high churn, many contributors, high fan_in, high complexity.
        - Decreases with older, stable files and higher days_since_last_change and
          stronger code reviews.
    """

    # Normalize metrics into [0, 1] for weighting
    bug_total_norm = df["bug_count_total"] / 100.0
    bug_density_norm = df["bug_density"] / 5.0
    recent_bugs_norm = df["bugs_last_30_days"] / 20.0

    # Map -1, 0, 1 → 0.0, 0.5, 1.0
    bug_trend_raw = df["bug_trend"]
    bug_trend_norm = (bug_trend_raw + 1.0) / 2.0

    test_fail_norm = df["test_fail_count_related"] / 30.0
    churn_norm = df["code_churn_recent"] / 500.0

    days_since_change = df["days_since_last_change"].clip(0, 365)
    recency_risk = 1.0 - (days_since_change / 365.0)  # recent change = higher risk

    file_age = df["file_age_days"].clip(1, 3650)
    age_risk = 1.0 - (file_age / 3650.0)  # younger files = higher risk

    contributors_norm = df["num_contributors"] / 15.0
    fan_in_norm = df["fan_in"] / 50.0
    cc_norm = df["cyclomatic_complexity"] / 30.0
    max_cc_norm = df["max_function_complexity"] / 50.0
    review_norm = df["review_comment_count"] / 100.0

    # Start from a low base so low-risk files are possible
    risk = 5.0

    # --- Strong bug signals ---
    risk += 30.0 * bug_total_norm
    risk += 25.0 * recent_bugs_norm
    risk += 20.0 * bug_density_norm
    risk += 18.0 * test_fail_norm
    risk += 15.0 * bug_trend_norm

    # --- Change & churn ---
    risk += 12.0 * churn_norm
    risk += 10.0 * recency_risk
    risk += 8.0 * age_risk

    # --- Ownership & architecture ---
    risk += 8.0 * contributors_norm
    risk += 8.0 * fan_in_norm

    # --- Complexity ---
    risk += 8.0 * cc_norm
    risk += 8.0 * max_cc_norm

    # --- Reviews mitigate risk strongly ---
    risk -= 15.0 * review_norm

    # Lower noise: we want cleaner, more learnable labels
    noise = rng.normal(loc=0.0, scale=2.0, size=len(df))
    risk = risk + noise

    # Clip to [0, 100]
    risk = np.clip(risk, 0.0, 100.0)
    return risk


def compute_bug_risk_class(bug_risk_score: np.ndarray) -> np.ndarray:
    """
    Convert continuous bug_risk_score to discrete bug_risk_class (0–5) based on
    wider, cleaner thresholds:

        [0, 20)   → 0 (Very Low)
        [20, 40)  → 1 (Low)
        [40, 55)  → 2 (Slightly Risky)
        [55, 70)  → 3 (Medium)
        [70, 85)  → 4 (High)
        [85, 100] → 5 (Critical)
    """
    classes = np.zeros_like(bug_risk_score, dtype=int)
    classes[(bug_risk_score >= 20) & (bug_risk_score < 40)] = 1
    classes[(bug_risk_score >= 40) & (bug_risk_score < 55)] = 2
    classes[(bug_risk_score >= 55) & (bug_risk_score < 70)] = 3
    classes[(bug_risk_score >= 70) & (bug_risk_score < 85)] = 4
    classes[bug_risk_score >= 85] = 5
    return classes


# ---------------------------------------------------------------------------
# Balanced dataset generation
# ---------------------------------------------------------------------------

def generate_balanced_dataset(
    total_samples: int,
    rng: np.random.Generator,
    batch_size: int = 5000,
    max_attempts: int = 20,
) -> pd.DataFrame:
    """
    Generate a dataset with approximately balanced bug_risk_class distribution.

    Strategy:
        - Decide per-class target = total_samples / NUM_CLASSES.
        - Iteratively generate batches of synthetic samples and collect them.
        - After generation, downsample or oversample each class to reach
          the per-class target:
            * if class has >= target rows -> sample without replacement
            * if class has  < target rows -> sample with replacement (oversampling)
    """

    if total_samples % NUM_CLASSES != 0:
        raise ValueError(
            f"total_samples ({total_samples}) must be divisible by {NUM_CLASSES} "
            "to ensure a perfectly balanced dataset."
        )

    per_class_target = total_samples // NUM_CLASSES
    print(f"Target samples per bug_risk_class: {per_class_target}")

    collected_frames: List[pd.DataFrame] = []
    class_counts = np.zeros(NUM_CLASSES, dtype=int)

    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        print(f"\nGeneration attempt {attempt} (batch size = {batch_size})")

        # Generate a batch of features
        df_batch = generate_features(num_samples=batch_size, rng=rng)

        # Compute labels for this batch
        code_quality_score = compute_code_quality_score(df_batch, rng=rng)
        bug_risk_score = compute_bug_risk_score(df_batch, rng=rng)
        bug_risk_class = compute_bug_risk_class(bug_risk_score)

        df_batch["code_quality_score"] = code_quality_score
        df_batch["bug_risk_score"] = bug_risk_score
        df_batch["bug_risk_class"] = bug_risk_class

        collected_frames.append(df_batch)

        # Update cumulative class counts
        counts_batch = df_batch["bug_risk_class"].value_counts().reindex(
            range(NUM_CLASSES), fill_value=0
        )
        class_counts += counts_batch.values

        print("Cumulative class counts after this batch:")
        for c in range(NUM_CLASSES):
            print(f"  Class {c}: {class_counts[c]}")

        # If all classes already reached or exceeded the target, we can stop early
        if (class_counts >= per_class_target).all():
            print("All classes reached the per-class target; stopping generation early.")
            break

    # Concatenate all batches into one big pool
    full_df = pd.concat(collected_frames, axis=0, ignore_index=True)
    print(f"\nTotal collected samples before balancing: {full_df.shape[0]}")

    # If some classes are still below target, we will oversample them
    if (class_counts < per_class_target).any():
        print(
            "Warning: some classes are below the per-class target; "
            "will oversample minority classes with replacement."
        )

    # Build a perfectly balanced dataset using down/over-sampling
    balanced_frames: List[pd.DataFrame] = []
    for c in range(NUM_CLASSES):
        cls_df = full_df[full_df["bug_risk_class"] == c]
        n_cls = len(cls_df)

        if n_cls == 0:
            raise RuntimeError(
                f"No samples generated for class {c}. "
                "Please adjust the bug risk heuristic to make this class possible."
            )

        if n_cls >= per_class_target:
            # Enough samples: sample without replacement
            cls_sampled = cls_df.sample(
                n=per_class_target,
                random_state=RANDOM_SEED,
                replace=False,
            )
        else:
            # Not enough samples: oversample with replacement
            print(
                f"Class {c} has only {n_cls} samples; "
                f"oversampling with replacement to {per_class_target}."
            )
            cls_sampled = cls_df.sample(
                n=per_class_target,
                random_state=RANDOM_SEED,
                replace=True,
            )

        balanced_frames.append(cls_sampled)

    final_df = pd.concat(balanced_frames, axis=0, ignore_index=True)
    # Shuffle final dataset
    final_df = final_df.sample(
        frac=1.0, random_state=RANDOM_SEED
    ).reset_index(drop=True)

    print(f"Final balanced dataset shape: {final_df.shape}")
    print("Final bug_risk_class distribution:")
    print(final_df["bug_risk_class"].value_counts().sort_index())

    return final_df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_dataset(num_samples: int = DEFAULT_NUM_SAMPLES) -> pd.DataFrame:
    """
    Generate the full synthetic dataset with features and labels, balanced by bug_risk_class.

    Returns:
        DataFrame with columns:
            - 22 feature columns
            - code_quality_score
            - bug_risk_score
            - bug_risk_class
    """
    rng = np.random.default_rng(RANDOM_SEED)
    df = generate_balanced_dataset(
        total_samples=num_samples,
        rng=rng,
        batch_size=5000,      # can be tuned
        max_attempts=20,      # safety guard
    )
    return df


def save_dataset(df: pd.DataFrame, output_path: str = OUTPUT_CSV) -> None:
    """
    Save the generated dataset to a CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print("Head:")
    print(df.head())


def main(num_samples: int = DEFAULT_NUM_SAMPLES) -> None:
    df = generate_dataset(num_samples=num_samples)
    save_dataset(df, OUTPUT_CSV)


if __name__ == "__main__":
    # You can modify DEFAULT_NUM_SAMPLES or call main(num_samples=...) explicitly.
    main()
