#!/usr/bin/env python3
"""
app.py

Flask web application for:
"ANN-Based Code Quality & Bug Risk Prediction System Using Static & Process Metrics"

Features:
- Landing page (cyberpunk themed)
- Input page with:
    * Manual metrics entry
    * Single-row CSV upload
- Result page: code quality score + bug risk class + probabilities + top factors
- Breakdown page: detailed explanations & recommendations
- About page: documentation + CSV structure + template download
- JSON APIs:
    * /api/predict
    * /api/report
- Health check: /health
"""

import os
import json
import uuid
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    redirect,
    url_for,
    flash,
    make_response,
    send_from_directory,
    abort,
)
import joblib
import tensorflow as tf

# Optional modules we wrote
try:
    from explainability import compute_feature_contributions
except ImportError:
    compute_feature_contributions = None

try:
    from recommendations import generate_recommendations
except ImportError:
    generate_recommendations = None

try:
    from report_generator import build_report_html
except ImportError:
    build_report_html = None


# ---------------------------------------------------------------------------
# Paths (must match train_model.py)
# ---------------------------------------------------------------------------

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "multitask_ann_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

TEMPLATE_CSV_REL_PATH = os.path.join("static", "files", "template_metrics.csv")

FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

# In-memory analysis store for this process.
# Maps analysis_id -> dict with:
#   features, prediction, explanations, recommendations
ANALYSIS_STORE: Dict[str, Dict[str, Any]] = {}



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def load_model_and_artifacts() -> Tuple[tf.keras.Model, Any, Dict[str, Any]]:
    """
    Load trained model, scaler, and metadata from disk.

    Returns:
        model: tf.keras.Model
        scaler: StandardScaler
        metadata: dict
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model directory not found at {MODEL_PATH}. "
            f"Train and save the model first using train_model.py."
        )

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler file not found at {SCALER_PATH}. "
            f"Train and save the scaler first using train_model.py."
        )

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata file not found at {METADATA_PATH}. "
            f"Train and save metadata first using train_model.py."
        )

    print("[Flask] Loading Keras model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[Flask] Model loaded.")

    print("[Flask] Loading scaler...")
    scaler = joblib.load(SCALER_PATH)
    print("[Flask] Scaler loaded.")

    print("[Flask] Loading metadata...")
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    print("[Flask] Metadata loaded.")

    return model, scaler, metadata


# Load artifacts once at startup
MODEL, SCALER, METADATA = load_model_and_artifacts()
FEATURE_COLUMNS = METADATA["feature_columns"]
FEATURE_DESCRIPTIONS: Dict[str, str] = METADATA.get("feature_descriptions", {})
FEATURE_STATS: Dict[str, Dict[str, float]] = METADATA.get("feature_stats", {})
LABEL_MAPPING: Dict[str, str] = METADATA.get("label_mapping", {})
NUM_BUG_CLASSES = METADATA.get("num_bug_classes", 6)


# ---------------------------------------------------------------------------
# Inference utilities
# ---------------------------------------------------------------------------

def validate_and_prepare_features(feature_dict: Dict[str, Any]) -> np.ndarray:
    """
    Validate provided feature dictionary and convert it into a numpy array
    with the correct ordering.

    Args:
        feature_dict: Dict mapping feature name -> numeric value.

    Returns:
        features_array: np.ndarray shape (1, num_features)
    """
    missing = [f for f in FEATURE_COLUMNS if f not in feature_dict]
    if missing:
        raise ValueError(
            f"Missing required features: {missing}. "
            f"Expected all of: {FEATURE_COLUMNS}"
        )

    # Warn about extras (ignored)
    extras = [f for f in feature_dict if f not in FEATURE_COLUMNS]
    if extras:
        print(
            f"[Flask] Warning: Ignoring extra features not used by model: {extras}"
        )

    values = []
    for f in FEATURE_COLUMNS:
        try:
            values.append(float(feature_dict[f]))
        except (TypeError, ValueError):
            raise ValueError(
                f"Feature '{f}' must be numeric; got: {feature_dict[f]!r}"
            )

    return np.array(values, dtype=np.float32).reshape(1, -1)


def run_model_inference(features_array: np.ndarray) -> Dict[str, Any]:
    """
    Run the model on a single sample of features (unscaled), returning predictions.

    Args:
        features_array: np.ndarray shape (1, num_features), original scale.

    Returns:
        dict with:
            - code_quality_score: float
            - bug_risk_class: int
            - bug_risk_label: str
            - bug_risk_probabilities: Dict[int, float]
    """
    # 1. Scale
    features_scaled = SCALER.transform(features_array)

    # 2. Predict
    preds = MODEL.predict(features_scaled, verbose=0)
    if isinstance(preds, dict):
        quality_pred = preds["quality_output"]
        risk_pred = preds["risk_output"]
    else:
        quality_pred, risk_pred = preds[0], preds[1]

    code_quality_score = float(quality_pred[0, 0])
    bug_probs = risk_pred[0]
    bug_risk_class = int(np.argmax(bug_probs))
    bug_risk_label = LABEL_MAPPING.get(str(bug_risk_class), "Unknown")

    bug_risk_probabilities = {
        int(i): float(p) for i, p in enumerate(bug_probs)
    }

    return {
        "code_quality_score": code_quality_score,
        "bug_risk_class": bug_risk_class,
        "bug_risk_label": bug_risk_label,
        "bug_risk_probabilities": bug_risk_probabilities,
    }


def compute_explanations_if_available(
    raw_features: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute feature contributions if explainability module is available.
    """
    if compute_feature_contributions is None:
        return {}

    try:
        return compute_feature_contributions(
            model=MODEL,
            scaler=SCALER,
            raw_feature_array=raw_features,
            feature_names=FEATURE_COLUMNS,
        )
    except Exception as e:
        print(f"[Flask] Warning: failed to compute explanations: {e}")
        return {}


def compute_recommendations_if_available(
    feature_dict: Dict[str, Any],
    prediction: Dict[str, Any],
    explanations: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute recommendations if recommendations module is available.
    """
    if generate_recommendations is None:
        return {}

    try:
        return generate_recommendations(
            features=feature_dict,
            prediction=prediction,
            explanations=explanations,
        )
    except Exception as e:
        print(f"[Flask] Warning: failed to compute recommendations: {e}")
        return {}


def store_analysis(
    features: Dict[str, Any],
    prediction: Dict[str, Any],
    explanations: Dict[str, Any],
    recommendations: Dict[str, Any],
) -> str:
    """
    Store a single analysis in memory and return a unique analysis_id.
    """
    analysis_id = str(uuid.uuid4())
    ANALYSIS_STORE[analysis_id] = {
        "features": features,
        "prediction": prediction,
        "explanations": explanations,
        "recommendations": recommendations,
    }

    # Optional: keep store from growing indefinitely (simple pruning)
    if len(ANALYSIS_STORE) > 200:
        # remove oldest 50 entries
        for key in list(ANALYSIS_STORE.keys())[:50]:
            ANALYSIS_STORE.pop(key, None)

    return analysis_id


def get_analysis(analysis_id: str) -> Dict[str, Any]:
    """
    Retrieve stored analysis by ID. Raises KeyError if not found.
    """
    if analysis_id not in ANALYSIS_STORE:
        raise KeyError(f"Unknown analysis_id: {analysis_id}")
    return ANALYSIS_STORE[analysis_id]


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = FLASK_SECRET_KEY

    # ---------------- Landing / Static Pages ----------------

    @app.route("/", methods=["GET"])
    def index():
        """
        Landing page.
        """
        return render_template("landing.html")

    @app.route("/input", methods=["GET"])
    def input_page():
        """
        Page for manual entry + single-row CSV upload.
        """
        return render_template("input.html")

    @app.route("/about", methods=["GET"])
    def about():
        """
        About page with docs, feature descriptions, CSV format.
        """
        return render_template("about.html")

    @app.route("/download-template-csv", methods=["GET"])
    def download_template_csv():
        """
        Download the template CSV file (header + one example row).
        Expects the file at: static/files/template_metrics.csv
        """
        template_path = os.path.join(app.root_path, TEMPLATE_CSV_REL_PATH)
        if not os.path.exists(template_path):
            return (
                "Template CSV not found. Please create "
                "'static/files/template_metrics.csv'.",
                404,
            )

        directory = os.path.dirname(template_path)
        filename = os.path.basename(template_path)

        return send_from_directory(
            directory=directory,
            path=filename,
            as_attachment=True,
            download_name="codepulse_template_metrics.csv",
        )

    # ---------------- Manual form prediction ----------------

    @app.route("/predict-form", methods=["POST"])
    def predict_form():
        """
        Handle manual form submission from input.html.
        """
        feature_dict: Dict[str, Any] = {}
        for f in FEATURE_COLUMNS:
            value = request.form.get(f, None)
            if value is None or value == "":
                flash(f"Missing value for feature '{f}'", "error")
                return redirect(url_for("input_page"))
            feature_dict[f] = value

        try:
            raw_features = validate_and_prepare_features(feature_dict)
            prediction = run_model_inference(raw_features)
            explanations = compute_explanations_if_available(raw_features)
            recommendations = compute_recommendations_if_available(
                feature_dict, prediction, explanations
            )

            analysis_id = store_analysis(
                features=feature_dict,
                prediction=prediction,
                explanations=explanations,
                recommendations=recommendations,
            )

        except ValueError as e:
            flash(str(e), "error")
            return redirect(url_for("input_page"))
        except Exception as e:
            flash(f"Internal server error: {str(e)}", "error")
            return redirect(url_for("input_page"))

        # Redirect to result page for this analysis
        return redirect(url_for("result_page", analysis_id=analysis_id))

    # ---------------- CSV single-row prediction ----------------

    @app.route("/predict-csv-single", methods=["POST"])
    def predict_csv_single():
        """
        Handle single-row CSV upload from input.html.
        CSV must contain:
            - header row with FEATURE_COLUMNS
            - exactly one data row
        """
        file = request.files.get("csv_file")
        if file is None or file.filename == "":
            flash("Please upload a CSV file.", "error")
            return redirect(url_for("input_page"))

        try:
            # Read CSV with pandas
            df = pd.read_csv(file)
            if df.shape[0] < 1:
                raise ValueError("CSV file contains no data rows.")
            if df.shape[0] > 1:
                flash(
                    "CSV contains multiple rows. Only the first row will be used.",
                    "warning",
                )

            row = df.iloc[0]
            feature_dict: Dict[str, Any] = {}
            for f in FEATURE_COLUMNS:
                if f not in df.columns:
                    raise ValueError(
                        f"CSV is missing required column '{f}'. "
                        f"Expected columns: {FEATURE_COLUMNS}"
                    )
                feature_dict[f] = row[f]

            raw_features = validate_and_prepare_features(feature_dict)
            prediction = run_model_inference(raw_features)
            explanations = compute_explanations_if_available(raw_features)
            recommendations = compute_recommendations_if_available(
                feature_dict, prediction, explanations
            )

            analysis_id = store_analysis(
                features=feature_dict,
                prediction=prediction,
                explanations=explanations,
                recommendations=recommendations,
            )

        except ValueError as e:
            flash(str(e), "error")
            return redirect(url_for("input_page"))
        except Exception as e:
            flash(f"Error processing CSV: {str(e)}", "error")
            return redirect(url_for("input_page"))

        return redirect(url_for("result_page", analysis_id=analysis_id))

    # ---------------- Result & Breakdown pages ----------------

    @app.route("/result/<analysis_id>", methods=["GET"])
    def result_page(analysis_id: str):
        """
        Show high-level result for a completed analysis.
        """
        try:
            data = get_analysis(analysis_id)
        except KeyError:
            flash("Analysis not found or expired. Please run a new analysis.", "error")
            return redirect(url_for("input_page"))

        prediction = data["prediction"]
        explanations = data.get("explanations", {})
        return render_template(
            "result.html",
            analysis_id=analysis_id,
            prediction=prediction,
            explanations=explanations,
            label_mapping=LABEL_MAPPING,
        )

    @app.route("/breakdown/<analysis_id>", methods=["GET"])
    def breakdown_page(analysis_id: str):
        """
        Show detailed breakdown for a completed analysis.
        """
        try:
            data = get_analysis(analysis_id)
        except KeyError:
            flash("Analysis not found or expired. Please run a new analysis.", "error")
            return redirect(url_for("input_page"))

        prediction = data["prediction"]
        explanations = data.get("explanations", {})
        recommendations = data.get("recommendations", {})
        features = data.get("features", {})

        return render_template(
            "breakdown.html",
            analysis_id=analysis_id,
            prediction=prediction,
            explanations=explanations,
            recommendations=recommendations,
            features=features,
            feature_descriptions=FEATURE_DESCRIPTIONS,
        )

    from flask import send_file
    import os

    @app.route("/download-report/<analysis_id>")
    def download_report(analysis_id):
        file_path = os.path.join("reports", f"{analysis_id}.pdf")  # or .csv / .zip

        if not os.path.exists(file_path):
            abort(404)

        return send_file(
            file_path,
            as_attachment=True,
            download_name=f"CodeQuB_Report_{analysis_id}.pdf"
        )

    # ---------------- JSON API: Predict ----------------

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        """
        JSON prediction API.

        Expected JSON body:
        {
            "features": {
                "loc": 800,
                ...
            }
        }
        """
        if not request.is_json:
            return jsonify({"error": "Request body must be JSON."}), 400

        data = request.get_json()
        if "features" not in data:
            return jsonify({"error": "JSON must contain 'features' object."}), 400

        feature_dict = data["features"]

        try:
            raw_features = validate_and_prepare_features(feature_dict)
            prediction = run_model_inference(raw_features)
            explanations = compute_explanations_if_available(raw_features)
            recommendations = compute_recommendations_if_available(
                feature_dict, prediction, explanations
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500

        return jsonify(
            {
                "prediction": prediction,
                "explanations": explanations,
                "recommendations": recommendations,
            }
        ), 200

    # ---------------- JSON API: Report ----------------

    @app.route("/api/report", methods=["POST"])
    def api_report():
        """
        Generate HTML report for given features.

        Expected JSON body:
        {
            "features": { ... }  # same as /api/predict
        }
        """
        if build_report_html is None:
            return jsonify({"error": "Report generation module not available."}), 501

        if not request.is_json:
            return jsonify({"error": "Request body must be JSON."}), 400

        data = request.get_json()
        if "features" not in data:
            return jsonify({"error": "JSON must contain 'features' object."}), 400

        feature_dict = data["features"]

        try:
            raw_features = validate_and_prepare_features(feature_dict)
            prediction = run_model_inference(raw_features)
            explanations = compute_explanations_if_available(raw_features)
            recommendations = compute_recommendations_if_available(
                feature_dict, prediction, explanations
            )

            report_html = build_report_html(
                features=feature_dict,
                prediction=prediction,
                explanations=explanations,
                recommendations=recommendations,
                metadata={
                    "feature_descriptions": FEATURE_DESCRIPTIONS,
                    "feature_stats": FEATURE_STATS,
                    "label_mapping": LABEL_MAPPING,
                },
            )

        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500

        resp = make_response(report_html)
        resp.headers["Content-Type"] = "text/html; charset=utf-8"
        return resp

    
    # ---------------- Health Check ----------------

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"}), 200

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# Entry point (Render / Gunicorn compatible)
# ---------------------------------------------------------------------------

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

