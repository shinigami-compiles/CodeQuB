#!/usr/bin/env python3
"""
report_generator.py

HTML report generator for:
"ANN-Based Code Quality & Bug Risk Prediction System Using Static & Process Metrics"

Provides:
    - build_report_html(features, prediction, explanations, recommendations, metadata)

Returns a full HTML string that can be served directly by Flask.
The report is designed to be:
    - Readable
    - Print-friendly
    - Easy to extend (CSS, charts, PDF conversion)
"""

from typing import Dict, Any, List
import html


def _escape(value: Any) -> str:
    """HTML-escape any value to safely insert into HTML."""
    return html.escape(str(value))


def _format_float(value: Any, digits: int = 2) -> str:
    """Safely format as float with digits decimal places."""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _render_feature_table(
    features: Dict[str, Any],
    feature_descriptions: Dict[str, str],
) -> str:
    """Render an HTML table of features with descriptions."""
    rows: List[str] = []
    for name, value in features.items():
        desc = feature_descriptions.get(name, "")
        rows.append(
            f"""
            <tr>
                <td><code>{_escape(name)}</code></td>
                <td>{_escape(_format_float(value, 3))}</td>
                <td>{_escape(desc)}</td>
            </tr>
            """
        )

    return (
        """
        <table class="table table-sm table-striped align-middle">
            <thead class="table-light">
                <tr>
                    <th>Feature</th>
                    <th>Value</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
        """
        + "\n".join(rows)
        + """
            </tbody>
        </table>
        """
    )


def _render_probabilities_table(
    bug_risk_probabilities: Dict[int, float],
    label_mapping: Dict[str, str],
) -> str:
    """Render an HTML table of bug risk class probabilities."""
    rows: List[str] = []
    for cls_idx, prob in sorted(bug_risk_probabilities.items()):
        label = label_mapping.get(str(cls_idx), "")
        rows.append(
            f"""
            <tr>
                <td>{cls_idx}</td>
                <td>{_escape(label)}</td>
                <td>{_format_float(prob * 100, 2)}%</td>
            </tr>
            """
        )

    return (
        """
        <table class="table table-sm table-striped align-middle">
            <thead class="table-light">
                <tr>
                    <th>Class</th>
                    <th>Label</th>
                    <th>Probability</th>
                </tr>
            </thead>
            <tbody>
        """
        + "\n".join(rows)
        + """
            </tbody>
        </table>
        """
    )


def _render_explanations_section(
    explanations: Dict[str, Any],
    section_key: str,
    title: str,
) -> str:
    """
    Render an HTML list of top feature contributions.

    section_key: "quality" or "risk"
    """
    expl_list = explanations.get(section_key, [])
    if not expl_list:
        return f"<p class='text-muted'>No explanation data available for {title.lower()}.</p>"

    rows: List[str] = []
    for e in expl_list:
        feature = e.get("feature", "")
        contribution = e.get("contribution", 0.0)
        value = e.get("value", "")
        direction = "increases" if contribution > 0 else "decreases"
        rows.append(
            f"""
            <tr>
                <td><code>{_escape(feature)}</code></td>
                <td>{_format_float(value, 3)}</td>
                <td>{_format_float(contribution, 4)}</td>
                <td>{direction} the {title.lower()}</td>
            </tr>
            """
        )

    return (
        f"""
        <h5 class="mt-3">{_escape(title)} – Top Feature Contributions</h5>
        <p class="text-muted small">
            Contributions are normalized (sum of absolute values ≈ 1). Positive values indicate
            features that push the score/class upward; negative values pull it downward.
        </p>
        <table class="table table-sm table-striped align-middle">
            <thead class="table-light">
                <tr>
                    <th>Feature</th>
                    <th>Value</th>
                    <th>Contribution</th>
                    <th>Effect</th>
                </tr>
            </thead>
            <tbody>
        """
        + "\n".join(rows)
        + """
            </tbody>
        </table>
        """
    )


def _render_recommendations_section(
    recommendations: Dict[str, List[str]],
) -> str:
    """Render recommendations grouped by quality & risk."""
    quality_recs = recommendations.get("quality", []) or []
    risk_recs = recommendations.get("risk", []) or []

    def _render_list(title: str, recs: List[str]) -> str:
        if not recs:
            return (
                f"<p class='text-muted'>No specific {title.lower()} recommendations generated.</p>"
            )
        items = "\n".join(f"<li>{_escape(r)}</li>" for r in recs)
        return (
            f"<h5 class='mt-3'>{_escape(title)}</h5>"
            f"<ul>{items}</ul>"
        )

    html_block = """
    <div class="row">
        <div class="col-md-6 mb-3">
            {quality_block}
        </div>
        <div class="col-md-6 mb-3">
            {risk_block}
        </div>
    </div>
    """.format(
        quality_block=_render_list("Code Quality Recommendations", quality_recs),
        risk_block=_render_list("Bug Risk Recommendations", risk_recs),
    )

    return html_block


def build_report_html(
    features: Dict[str, Any],
    prediction: Dict[str, Any],
    explanations: Dict[str, Any],
    recommendations: Dict[str, Any],
    metadata: Dict[str, Any],
) -> str:
    """
    Build a complete HTML report.

    Args:
        features: dict of input features.
        prediction: dict containing:
            - code_quality_score (float)
            - bug_risk_class (int)
            - bug_risk_label (str)
            - bug_risk_probabilities (dict[int -> float])
        explanations: dict from explainability.compute_feature_contributions()
        recommendations: dict from recommendations.generate_recommendations()
        metadata: dict containing:
            - feature_descriptions
            - feature_stats (optional)
            - label_mapping

    Returns:
        HTML string suitable for sending as a response.
    """
    feature_descriptions: Dict[str, str] = metadata.get("feature_descriptions", {})
    label_mapping: Dict[str, str] = metadata.get("label_mapping", {})

    code_quality_score = prediction.get("code_quality_score", 0.0)
    bug_risk_class = prediction.get("bug_risk_class", 0)
    bug_risk_label = prediction.get("bug_risk_label", "Unknown")
    bug_risk_probabilities = prediction.get("bug_risk_probabilities", {})

    feature_table_html = _render_feature_table(features, feature_descriptions)
    prob_table_html = _render_probabilities_table(
        bug_risk_probabilities, label_mapping
    )
    quality_expl_html = _render_explanations_section(
        explanations, section_key="quality", title="Code Quality"
    )
    risk_expl_html = _render_explanations_section(
        explanations, section_key="risk", title="Bug Risk"
    )
    recs_html = _render_recommendations_section(recommendations)

    # Full HTML document
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <title>Code Quality & Bug Risk Report</title>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>

    <!-- Bootstrap CSS (CDN) -->
    <link
        href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
        rel="stylesheet"
        integrity="sha384-T3c6CoIi6uLrA9TneNEoa7RxnatzjcDSCmG1MXxSR1GAsXEV/Dwwykc2MPK8M2HN"
        crossorigin="anonymous"
    />

    <style>
        body {{
            background-color: #f8f9fa;
        }}
        .report-container {{
            max-width: 1100px;
            margin: 20px auto 40px auto;
            background: #ffffff;
            border-radius: 0.75rem;
            box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.08);
            padding: 2rem 2.5rem;
        }}
        .section-title {{
            border-left: 4px solid #0d6efd;
            padding-left: 0.75rem;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }}
        .score-badge {{
            font-size: 1.25rem;
        }}
        @media print {{
            body {{
                background-color: #ffffff;
            }}
            .report-container {{
                box-shadow: none;
                border-radius: 0;
                margin: 0;
                padding: 0;
            }}
        }}
    </style>
</head>
<body>
<div class="report-container">
    <header class="mb-4">
        <h2>Code Quality & Bug Risk Prediction Report</h2>
        <p class="text-muted mb-1">
            Generated by the ANN-Based Code Quality & Bug Risk Prediction System
        </p>
        <p class="text-muted small">
            This report summarizes the predicted quality and risk for a single code file/module based on static & process metrics.
        </p>
    </header>

    <!-- Summary section -->
    <section>
        <h4 class="section-title">Summary</h4>
        <div class="row align-items-center">
            <div class="col-md-6 mb-3">
                <div class="card border-0">
                    <div class="card-body">
                        <h5>Code Quality Score</h5>
                        <p class="score-badge">
                            <span class="badge bg-primary">
                                {_format_float(code_quality_score, 2)} / 100
                            </span>
                        </p>
                        <p class="text-muted small mb-0">
                            Higher scores indicate cleaner, more maintainable, and better-tested code.
                        </p>
                    </div>
                </div>
            </div>
            <div class="col-md-6 mb-3">
                <div class="card border-0">
                    <div class="card-body">
                        <h5>Bug Risk Level</h5>
                        <p class="score-badge">
                            <span class="badge bg-danger">
                                Class {bug_risk_class} – {_escape(bug_risk_label)}
                            </span>
                        </p>
                        <p class="text-muted small mb-0">
                            Bug risk classes range from 0 (Very Low) to 5 (Critical),
                            based on modeled likelihood of defects given the code's history and metrics.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Probabilities section -->
    <section>
        <h4 class="section-title">Bug Risk Class Probabilities</h4>
        <p class="text-muted small">
            These probabilities represent the model's confidence for each bug risk class.
        </p>
        {prob_table_html}
    </section>

    <!-- Explanations section -->
    <section>
        <h4 class="section-title">Feature Impact Explanations</h4>
        <p class="text-muted small">
            The tables below highlight which features most strongly influence the predicted code quality
            and bug risk. Contributions are approximate and based on a Gradient × Input explanation method.
        </p>
        <div class="row">
            <div class="col-md-6">
                {quality_expl_html}
            </div>
            <div class="col-md-6">
                {risk_expl_html}
            </div>
        </div>
    </section>

    <!-- Recommendations section -->
    <section>
        <h4 class="section-title">Recommendations</h4>
        <p class="text-muted small">
            These recommendations combine model predictions and metric-based heuristics to suggest practical ways
            to improve code quality and reduce bug risk.
        </p>
        {recs_html}
    </section>

    <!-- Raw feature section -->
    <section>
        <h4 class="section-title">Input Features</h4>
        <p class="text-muted small">
            The following metrics were used as input to the prediction model for this code file/module.
        </p>
        {feature_table_html}
    </section>

    <footer class="mt-4 pt-3 border-top">
        <p class="text-muted small mb-0">
            Generated automatically. Predictions are based on synthetic training data and should be interpreted
            as guidance rather than absolute truth.
        </p>
    </footer>
</div>

<!-- Bootstrap JS (optional, for interactive behavior; not required for static report) -->
<script
    src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"
    integrity="sha384-C6RzsynM9kWDrMNeT87bh95OGNyZPhcTNXj1NW7RuBCsyN/o0jlpcV8Qyq46cDfL"
    crossorigin="anonymous">
</script>
</body>
</html>
"""
    return html_doc
