// static/js/main.js

// Wait for DOM to be fully loaded
document.addEventListener("DOMContentLoaded", function () {
    initModeToggle();
    initDownloadReport();
});

/**
 * Toggle between Manual Input mode and CSV mode on the input page.
 * Uses .mode-btn buttons and .mode-section panels.
 */
function initModeToggle() {
    const modeButtons = document.querySelectorAll(".mode-toggle .mode-btn");
    const sections = document.querySelectorAll(".mode-section");

    if (!modeButtons.length || !sections.length) return;

    modeButtons.forEach((btn) => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-target");
            if (!targetId) return;

            // Activate button
            modeButtons.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");

            // Show corresponding section
            sections.forEach((section) => {
                if (section.id === targetId) {
                    section.classList.add("active");
                } else {
                    section.classList.remove("active");
                }
            });
        });
    });
}

/**
 * Initialize Download Report button on breakdown page.
 * It scrapes the feature table to reconstruct the features dict,
 * sends it to /api/report, and opens the returned HTML in a new tab.
 */
function initDownloadReport() {
    const btn = document.getElementById("download-report-btn");
    if (!btn) return; // Not on breakdown page

    btn.addEventListener("click", async () => {
        try {
            btn.disabled = true;
            btn.textContent = "⏳ Generating Report...";

            const features = collectFeaturesFromTable();
            if (!features || Object.keys(features).length === 0) {
                alert("Could not read features from the breakdown table.");
                return;
            }

            const response = await fetch("/api/report", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ features: features })
            });

            if (!response.ok) {
                const errorText = await safeReadText(response);
                console.error("Report generation failed:", errorText);
                alert("Failed to generate report. Please try again.");
                return;
            }

            const html = await response.text();
            const reportWindow = window.open("", "_blank");
            if (reportWindow) {
                reportWindow.document.open();
                reportWindow.document.write(html);
                reportWindow.document.close();
            } else {
                // Popup blocked
                alert("Report generated, but browser blocked pop-up. Please allow pop-ups for this site.");
            }

        } catch (err) {
            console.error("Error generating report:", err);
            alert("An unexpected error occurred while generating the report.");
        } finally {
            btn.disabled = false;
            btn.textContent = "⬇ Download Full HTML Report";
        }
    });
}

/**
 * Collect features from the breakdown page's feature table.
 * Assumes the table has:
 *  - first column: feature name
 *  - second column: value
 */
function collectFeaturesFromTable() {
    const table = document.querySelector(".feature-table tbody");
    if (!table) return {};

    const features = {};
    const rows = table.querySelectorAll("tr");

    rows.forEach((row) => {
        const cells = row.querySelectorAll("td");
        if (cells.length < 2) return;

        // First cell contains <code>feature_name</code>
        const featureCell = cells[0];
        const valueCell = cells[1];

        const featureNameElement = featureCell.querySelector("code");
        const featureName = featureNameElement
            ? featureNameElement.textContent.trim()
            : featureCell.textContent.trim();

        const rawValue = valueCell.textContent.trim();
        const numericValue = parseFloat(rawValue);

        if (featureName) {
            // If parseFloat fails (NaN), we still send raw string; backend will handle.
            features[featureName] = isNaN(numericValue) ? rawValue : numericValue;
        }
    });

    return features;
}

/**
 * Safely read text from a response (used for debugging / error messages).
 */
async function safeReadText(response) {
    try {
        return await response.text();
    } catch (e) {
        return "<no body>";
    }
}


// -------- Overlay helpers (Start / Analyze) --------
function showOverlay(type = "start") {
    const overlayId = (type === "analyze") ? "overlay-analyze" : "overlay-start";
    const overlay = document.getElementById(overlayId);
    if (!overlay) return;
    overlay.classList.remove("hidden");
    // force reflow then make visible for transition
    requestAnimationFrame(() => overlay.classList.add("visible"));
}

function hideOverlay(type = "start") {
    const overlayId = (type === "analyze") ? "overlay-analyze" : "overlay-start";
    const overlay = document.getElementById(overlayId);
    if (!overlay) return;
    overlay.classList.remove("visible");
    // after transition hide completely
    setTimeout(() => overlay.classList.add("hidden"), 260);
}

// Close overlay on backdrop click
document.addEventListener("click", function (e) {
    const target = e.target;
    if (target && target.classList && target.classList.contains("overlay-backdrop")) {
        // determine which overlay
        const overlay = target.closest(".overlay");
        if (overlay && overlay.id) {
            if (overlay.id === "overlay-start") hideOverlay("start");
            if (overlay.id === "overlay-analyze") hideOverlay("analyze");
        }
    }
});

// Hook Start Analysis button on landing page
document.addEventListener("DOMContentLoaded", function () {

    const startBtn = document.getElementById("startAnalysisBtn");

    if (startBtn) {
        startBtn.addEventListener("click", function (e) {
            e.preventDefault();

            document.body.classList.add("loading-active");
            showOverlay("start");

            const targetUrl = startBtn.getAttribute("href");

            setTimeout(() => {
                window.location.href = targetUrl;
            }, 600);
        });
    }

});


    // Analyze This File button on input page: it is .btn-primary-cyber inside the form actions
document.addEventListener("DOMContentLoaded", function () {

    const analyzeBtn = document.querySelector(".metrics-form .btn-primary-cyber");
    const form = document.querySelector(".metrics-form");

    if (analyzeBtn && form) {
        analyzeBtn.addEventListener("click", function (e) {

            const inputs = form.querySelectorAll("input[type='number']");
            let valid = true;

            inputs.forEach(input => {
                if (input.value.trim() === "" || isNaN(input.value)) {
                    valid = false;
                    input.classList.add("input-error");
                } else {
                    input.classList.remove("input-error");
                }
            });

            if (!valid) {
                e.preventDefault();
                showFormError("Please fill all fields with numeric values only.");
                return;
            }

            // 🔥 FORCE blur + loader
            document.body.classList.add("loading-active");
            showOverlay("analyze");
        });
    }
});

function showFormError(msg) {
    let errorBox = document.getElementById("formErrorBox");

    if (!errorBox) {
        errorBox = document.createElement("div");
        errorBox.id = "formErrorBox";
        errorBox.style.color = "#ff4d4d";
        errorBox.style.marginTop = "10px";
        errorBox.style.fontSize = "0.9rem";

        const form = document.querySelector(".metrics-form");
        form.appendChild(errorBox);
    }

    errorBox.innerText = msg;
}


document.addEventListener("DOMContentLoaded", () => {
    const csvForm = document.getElementById("csvUploadForm");
    const csvInput = document.getElementById("csvFileInput");

    if (!csvForm || !csvInput) return;

    csvForm.addEventListener("submit", (e) => {
        const file = csvInput.files[0];
        if (!file) {
            alert("Please upload a CSV file.");
            e.preventDefault();
            return;
        }

        if (!file.name.endsWith(".csv")) {
            alert("Only .csv files are allowed.");
            e.preventDefault();
            return;
        }

        // 🔥 SHOW LOADER + BLUR
        document.body.classList.add("loading-active");
        showOverlay("analyze");

        const reader = new FileReader();
        reader.onload = function (event) {
            const text = event.target.result.trim();
            const lines = text.split("\n");

            if (lines.length < 2) {
                alert("CSV must contain header + one data row.");
                resetLoader();   // ❗ important
                e.preventDefault();
                return;
            }

            if (lines.length > 2) {
                alert("CSV must contain ONLY ONE data row.");
                resetLoader();
                e.preventDefault();
                return;
            }

            const headers = lines[0].split(",").map(h => h.trim());
            const expectedHeaders = [
                "loc","num_functions","avg_function_length","cyclomatic_complexity",
                "max_nesting_depth","max_function_complexity","std_dev_function_complexity",
                "lint_warning_count","comment_to_code_ratio","review_comment_count",
                "test_coverage_percent","num_test_files_linked","code_churn_recent",
                "num_contributors","file_age_days","fan_in","bug_count_total","bug_density",
                "bugs_last_30_days","bug_trend","test_fail_count_related","days_since_last_change"
            ];

            const missing = expectedHeaders.filter(h => !headers.includes(h));
            if (missing.length > 0) {
                alert("CSV missing columns:\n" + missing.join(", "));
                resetLoader();
                e.preventDefault();
                return;
            }

            // ✅ VALID → submit form
            csvForm.submit();
        };

        reader.readAsText(file);
        e.preventDefault();
    });

function resetLoader() {
    document.body.classList.remove("loading-active");

    document.querySelectorAll(".overlay").forEach(o => {
        o.classList.remove("visible");
        o.classList.add("hidden");
    });
}


});


// ✅ HARD RESET loading state on every page load (BACK / FORWARD safe)
window.addEventListener("pageshow", function () {
    document.body.classList.remove("loading-active");

    document.querySelectorAll(".overlay").forEach(o => {
        o.classList.remove("visible");
        o.classList.add("hidden");
    });
});

window.addEventListener("beforeunload", () => {
    document.body.classList.remove("loading-active");
});

window.addEventListener("pageshow", function () {
    resetLoader();
});
