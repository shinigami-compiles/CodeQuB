#!/usr/bin/env python3
"""
explainability.py

Feature contribution / explanation utilities for:
"ANN-Based Code Quality & Bug Risk Prediction System Using Static & Process Metrics"

This module provides:
    - compute_feature_contributions(model, scaler, raw_feature_array, feature_names)

It uses a simple Gradient × Input style approach to approximate how much
each input feature contributes to:
    - Code Quality Score (regression head)
    - Bug Risk for the predicted class (classification head)

Designed to be used by app.py in the Flask backend.
"""

from typing import Dict, Any, List

import numpy as np
import tensorflow as tf


def _normalize_contributions(contribs: np.ndarray) -> np.ndarray:
    """
    Normalize contributions so that the sum of absolute values is 1.
    Keeps the sign (positive/negative impact).
    """
    abs_sum = np.sum(np.abs(contribs))
    if abs_sum == 0:
        return contribs
    return contribs / abs_sum


def compute_feature_contributions(
    model: tf.keras.Model,
    scaler,
    raw_feature_array: np.ndarray,
    feature_names: List[str],
    top_k: int = None,
) -> Dict[str, Any]:
    """
    Compute per-feature contribution scores for:
        - Code Quality Score (regression)
        - Bug Risk (classification, for predicted class)

    Args:
        model: Trained Keras model with outputs:
               {"quality_output": (batch, 1), "risk_output": (batch, num_classes)}
        scaler: Fitted StandardScaler used during training.
        raw_feature_array: np.ndarray of shape (1, num_features) with raw (unscaled) values.
        feature_names: List of feature names in the correct order.
        top_k: If not None, keep only top_k features by absolute contribution
               for each head (quality & risk).

    Returns:
        dict with:
        {
            "quality": [
                {
                    "feature": str,
                    "value": float,
                    "scaled_value": float,
                    "contribution": float,   # normalized contribution (sum |c| = 1)
                },
                ...
            ],
            "risk": [
                {
                    "feature": str,
                    "value": float,
                    "scaled_value": float,
                    "contribution": float,   # normalized for predicted class
                },
                ...
            ],
        }
    """
    if raw_feature_array.ndim != 2 or raw_feature_array.shape[0] != 1:
        raise ValueError(
            f"raw_feature_array must have shape (1, num_features), "
            f"got {raw_feature_array.shape}"
        )

    num_features = raw_feature_array.shape[1]
    if len(feature_names) != num_features:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) does not match "
            f"number of features ({num_features})"
        )

    # 1. Scale input (same as during training)
    x_scaled = scaler.transform(raw_feature_array)  # shape (1, num_features)

    # 2. Convert to TF tensor
    x_tf = tf.convert_to_tensor(x_scaled, dtype=tf.float32)

    # 3. Use GradientTape to compute gradients of outputs w.r.t inputs
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_tf)
        outputs = model(x_tf, training=False)

        if isinstance(outputs, dict):
            quality_output = outputs["quality_output"]   # shape (1, 1)
            risk_output = outputs["risk_output"]         # shape (1, num_classes)
        else:
            # Fallback: assume outputs is list/tuple [quality_output, risk_output]
            quality_output, risk_output = outputs[0], outputs[1]

        # For risk, we focus on the logit/prob for the predicted class
        risk_probs = risk_output[0]      # shape (num_classes,)
        pred_class = tf.argmax(risk_probs)  # scalar
        # Use the predicted class probability as scalar target
        risk_target = risk_probs[pred_class]  # scalar

    # Gradients of quality_output w.r.t inputs
    grad_quality = tape.gradient(quality_output, x_tf)  # shape (1, num_features)
    # Gradients of risk_target w.r.t inputs
    grad_risk = tape.gradient(risk_target, x_tf)        # shape (1, num_features)

    del tape  # free resources

    grad_quality_np = grad_quality.numpy()[0]  # (num_features,)
    grad_risk_np = grad_risk.numpy()[0]        # (num_features,)

    # 4. Gradient × Input contributions using scaled inputs
    x_scaled_np = x_scaled[0]  # (num_features,)

    contrib_quality = x_scaled_np * grad_quality_np
    contrib_risk = x_scaled_np * grad_risk_np

    # 5. Normalize contributions for interpretability
    contrib_quality_norm = _normalize_contributions(contrib_quality)
    contrib_risk_norm = _normalize_contributions(contrib_risk)

    # 6. Build structured explanation lists
    quality_explanations = []
    risk_explanations = []

    raw_values = raw_feature_array[0]

    for i, name in enumerate(feature_names):
        quality_explanations.append(
            {
                "feature": name,
                "value": float(raw_values[i]),
                "scaled_value": float(x_scaled_np[i]),
                "contribution": float(contrib_quality_norm[i]),
            }
        )
        risk_explanations.append(
            {
                "feature": name,
                "value": float(raw_values[i]),
                "scaled_value": float(x_scaled_np[i]),
                "contribution": float(contrib_risk_norm[i]),
            }
        )

    # 7. Sort by absolute contribution (descending)
    quality_explanations.sort(
        key=lambda x: abs(x["contribution"]), reverse=True
    )
    risk_explanations.sort(
        key=lambda x: abs(x["contribution"]), reverse=True
    )

    # 8. Optionally keep only top_k
    if top_k is not None and top_k > 0:
        quality_explanations = quality_explanations[:top_k]
        risk_explanations = risk_explanations[:top_k]

    return {
        "quality": quality_explanations,
        "risk": risk_explanations,
    }
