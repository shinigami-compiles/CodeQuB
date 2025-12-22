#!/usr/bin/env python3
"""
recommendations.py

Rule-based recommendation engine for:
"ANN-Based Code Quality & Bug Risk Prediction System Using Static & Process Metrics"

This module provides:
    - generate_recommendations(features, prediction, explanations)

Given:
    * features: dict of input feature values
    * prediction: model outputs (quality score, bug risk class, etc.)
    * explanations: feature-level contributions from explainability.py

It returns human-readable recommendations that tell the user WHAT they can improve.

The logic is intentionally rule-based and transparent, so it can work alongside
the ANN model and be adjusted easily.
"""

from typing import Dict, Any, List


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _get_feature_value(features: Dict[str, Any], name: str, default: float = 0.0) -> float:
    """Safely get a feature value from dict as float."""
    try:
        return float(features.get(name, default))
    except (TypeError, ValueError):
        return default


def _add_if_not_present(lst: List[str], message: str) -> None:
    """Append message to list only if not already present (avoid duplicates)."""
    if message not in lst:
        lst.append(message)


def _top_contributors(
    explanation_list: List[Dict[str, Any]],
    positive: bool = True,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Get top-k contributors by sign:
        positive=True  -> features that push score UP
        positive=False -> features that push score DOWN
    """
    if positive:
        filtered = [e for e in explanation_list if e["contribution"] > 0]
        filtered.sort(key=lambda x: x["contribution"], reverse=True)
    else:
        filtered = [e for e in explanation_list if e["contribution"] < 0]
        filtered.sort(key=lambda x: x["contribution"])  # more negative first
    return filtered[:top_k]


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------

def _generate_quality_recommendations(
    features: Dict[str, Any],
    prediction: Dict[str, Any],
    explanations: Dict[str, Any],
) -> List[str]:
    """
    Generate recommendations focused on improving Code Quality Score.
    """
    recs: List[str] = []

    quality_score = float(prediction.get("code_quality_score", 0.0))

    # Feature values
    cyclomatic_complexity = _get_feature_value(features, "cyclomatic_complexity")
    max_function_complexity = _get_feature_value(features, "max_function_complexity")
    max_nesting_depth = _get_feature_value(features, "max_nesting_depth")
    avg_function_length = _get_feature_value(features, "avg_function_length")
    lint_warning_count = _get_feature_value(features, "lint_warning_count")
    comment_to_code_ratio = _get_feature_value(features, "comment_to_code_ratio")
    test_coverage_percent = _get_feature_value(features, "test_coverage_percent")
    num_test_files_linked = _get_feature_value(features, "num_test_files_linked")
    code_churn_recent = _get_feature_value(features, "code_churn_recent")
    num_contributors = _get_feature_value(features, "num_contributors")
    fan_in = _get_feature_value(features, "fan_in")

    quality_expl = explanations.get("quality", [])

    # General messages based on score levels
    if quality_score < 40:
        _add_if_not_present(
            recs,
            "Overall code quality is low. Consider refactoring the module and addressing the key issues listed below."
        )
    elif quality_score < 70:
        _add_if_not_present(
            recs,
            "Code quality is moderate. With some targeted improvements, this file can become high quality."
        )
    else:
        _add_if_not_present(
            recs,
            "Code quality is relatively high. Maintain current practices and address minor issues as needed."
        )

    # Structural & complexity-related recommendations
    if cyclomatic_complexity >= 20:
        _add_if_not_present(
            recs,
            f"Cyclomatic complexity is high ({cyclomatic_complexity:.0f}). "
            "Refactor complex logic into smaller, simpler functions."
        )
    if max_function_complexity >= 25:
        _add_if_not_present(
            recs,
            f"The most complex function has very high complexity ({max_function_complexity:.0f}). "
            "Break it into smaller helper functions to improve readability and testability."
        )
    if max_nesting_depth >= 5:
        _add_if_not_present(
            recs,
            f"Maximum nesting depth is {max_nesting_depth:.0f}, which hurts readability. "
            "Consider early returns or restructuring nested conditionals."
        )
    if avg_function_length >= 50:
        _add_if_not_present(
            recs,
            f"Average function length is {avg_function_length:.0f} lines. "
            "Shorter functions are easier to understand and maintain; try splitting long functions."
        )

    # Style & documentation recommendations
    if lint_warning_count > 50:
        _add_if_not_present(
            recs,
            f"There are {lint_warning_count:.0f} lint/style warnings. "
            "Fixing these will improve consistency and reduce potential bugs."
        )
    if comment_to_code_ratio < 0.15:
        _add_if_not_present(
            recs,
            f"Comment-to-code ratio is low ({comment_to_code_ratio:.2f}). "
            "Add more meaningful comments and documentation for complex sections."
        )

    # Testing recommendations
    if test_coverage_percent < 60:
        _add_if_not_present(
            recs,
            f"Test coverage is only {test_coverage_percent:.0f}%. "
            "Add or expand unit tests, especially around complex and critical functions."
        )
    if num_test_files_linked == 0:
        _add_if_not_present(
            recs,
            "No test files are linked to this module. Create targeted test cases to cover its core functionality."
        )

    # Stability and ownership recommendations (quality perspective)
    if code_churn_recent > 200:
        _add_if_not_present(
            recs,
            f"Recent code churn is high ({code_churn_recent:.0f} lines changed). "
            "Stabilize the design and avoid large, frequent changes without tests."
        )
    if num_contributors > 8:
        _add_if_not_present(
            recs,
            f"The file has many contributors ({num_contributors:.0f}). "
            "Ensure clear ownership and consistent coding standards to avoid quality drift."
        )
    if fan_in > 20:
        _add_if_not_present(
            recs,
            f"This module has high fan-in ({fan_in:.0f} dependents). "
            "Be cautious when modifying it and consider extracting reusable pieces to reduce coupling."
        )

    # Use explanations to highlight top negative contributors
    if quality_expl:
        worst_factors = _top_contributors(quality_expl, positive=False, top_k=3)
        if worst_factors:
            factor_names = ", ".join(f["feature"] for f in worst_factors)
            _add_if_not_present(
                recs,
                "Top factors currently reducing quality: "
                + factor_names
                + ". Addressing these will have the biggest impact."
            )

    return recs


def _generate_risk_recommendations(
    features: Dict[str, Any],
    prediction: Dict[str, Any],
    explanations: Dict[str, Any],
) -> List[str]:
    """
    Generate recommendations focused on reducing Bug Risk.
    """
    recs: List[str] = []

    bug_risk_class = int(prediction.get("bug_risk_class", 0))
    bug_risk_label = prediction.get("bug_risk_label", "Unknown")

    # Feature values
    bug_count_total = _get_feature_value(features, "bug_count_total")
    bug_density = _get_feature_value(features, "bug_density")
    bugs_last_30_days = _get_feature_value(features, "bugs_last_30_days")
    bug_trend = _get_feature_value(features, "bug_trend")
    test_fail_count_related = _get_feature_value(features, "test_fail_count_related")
    code_churn_recent = _get_feature_value(features, "code_churn_recent")
    days_since_last_change = _get_feature_value(features, "days_since_last_change")
    file_age_days = _get_feature_value(features, "file_age_days")
    num_contributors = _get_feature_value(features, "num_contributors")
    fan_in = _get_feature_value(features, "fan_in")
    cyclomatic_complexity = _get_feature_value(features, "cyclomatic_complexity")
    max_function_complexity = _get_feature_value(features, "max_function_complexity")

    risk_expl = explanations.get("risk", [])

    # General messages based on bug risk class
    if bug_risk_class >= 4:
        _add_if_not_present(
            recs,
            f"Bug risk is {bug_risk_label} (class {bug_risk_class}). "
            "Treat this module as a hotspot and prioritize stabilization."
        )
    elif bug_risk_class == 3:
        _add_if_not_present(
            recs,
            "Bug risk is medium. Monitor this module closely and address the major contributors to risk."
        )
    else:
        _add_if_not_present(
            recs,
            f"Bug risk is {bug_risk_label.lower()}. Keep monitoring and maintain good practices to keep it low."
        )

    # Historical bug-related recommendations
    if bug_count_total > 20:
        _add_if_not_present(
            recs,
            f"This file has a history of {bug_count_total:.0f} bugs. "
            "Review previous defects and refactor the most error-prone areas."
        )
    if bug_density > 2.0:
        _add_if_not_present(
            recs,
            f"Bug density is high ({bug_density:.2f} bugs per KLOC). "
            "Consider thorough refactoring and adding defensive coding patterns."
        )
    if bugs_last_30_days >= 3:
        _add_if_not_present(
            recs,
            f"{bugs_last_30_days:.0f} bugs were reported in the last 30 days. "
            "Stabilize recent changes and focus on regression testing."
        )
    if bug_trend > 0:
        _add_if_not_present(
            recs,
            "Bug trend is increasing. Identify recent changes that might be introducing instability."
        )

    # Test-related recommendations
    if test_fail_count_related > 0:
        _add_if_not_present(
            recs,
            f"There are {test_fail_count_related:.0f} recent test failures related to this file. "
            "Investigate and fix these failing tests to improve reliability."
        )

    # Stability & change recommendations
    if code_churn_recent > 200:
        _add_if_not_present(
            recs,
            f"Recent code churn is high ({code_churn_recent:.0f} lines changed). "
            "Introduce smaller, incremental changes with proper tests to reduce risk."
        )
    if days_since_last_change < 7:
        _add_if_not_present(
            recs,
            f"This file was modified recently ({days_since_last_change:.0f} days ago). "
            "Be cautious of newly introduced logic and ensure adequate testing."
        )
    if file_age_days < 90:
        _add_if_not_present(
            recs,
            f"The file is relatively new ({file_age_days:.0f} days old). "
            "New modules typically have higher risk; stabilize the design and tests."
        )

    # Ownership & coupling recommendations
    if num_contributors > 8:
        _add_if_not_present(
            recs,
            f"This file has many contributors ({num_contributors:.0f}). "
            "Clarify ownership and enforce consistent coding and testing standards."
        )
    if fan_in > 20:
        _add_if_not_present(
            recs,
            f"This module is heavily depended on (fan-in = {fan_in:.0f}). "
            "Changes here can have wide impact; ensure changes are thoroughly reviewed and tested."
        )

    # Complexity & risk
    if cyclomatic_complexity >= 20 or max_function_complexity >= 25:
        _add_if_not_present(
            recs,
            "High complexity is contributing to bug risk. "
            "Refactor complex sections to make the logic easier to reason about and test."
        )

    # Use explanations to highlight top risk factors
    if risk_expl:
        worst_risk_factors = _top_contributors(risk_expl, positive=True, top_k=3)
        if worst_risk_factors:
            factor_names = ", ".join(f["feature"] for f in worst_risk_factors)
            _add_if_not_present(
                recs,
                "Top factors currently driving bug risk: "
                + factor_names
                + ". Focus on improving these first."
            )

    return recs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_recommendations(
    features: Dict[str, Any],
    prediction: Dict[str, Any],
    explanations: Dict[str, Any],
) -> Dict[str, List[str]]:
    """
    Public entry point for the recommendation engine.

    Args:
        features: dict of input features (raw values).
        prediction: dict containing at least:
            - "code_quality_score"
            - "bug_risk_class"
            - "bug_risk_label"
        explanations: dict from compute_feature_contributions(), containing:
            - "quality": list of feature explanations
            - "risk": list of feature explanations

    Returns:
        {
            "quality": [...],  # list of recommendation strings
            "risk": [...],     # list of recommendation strings
        }
    """
    quality_recs = _generate_quality_recommendations(features, prediction, explanations)
    risk_recs = _generate_risk_recommendations(features, prediction, explanations)

    return {
        "quality": quality_recs,
        "risk": risk_recs,
    }
