"""B7: render the real dashboard views to PNG (defanged URLs only).

A live browser was unavailable in this environment (CUA pipe down), so
instead of a raw screen-grab we render the exact figures + page text the
three Streamlit tabs display, from the same DashboardCore data, into
results/screenshots/*.png. Genuine outputs of the real model + SHAP +
translator pipeline, not mockups.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phishxai.dashboard_core import DashboardCore, defang  # noqa: E402
from phishxai.features.extractor import FEATURE_NAMES  # noqa: E402

OUT = ROOT / "results" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
dc = DashboardCore("lr_v1")

PHISH_URL = "https://veryfied-business-number104.netlify.app/business-center"
LEGIT_URL = "https://shopee.com.br/images/hero.png"
REDC, GREENC = "#d62728", "#2ca02c"


def render_detection(url, fname):
    r = dc.analyze(url)
    v = r["verdict"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.axis("off")
    color = "#b03030" if v == "phishing" else "#2e7d32"
    ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.12, facecolor=color,
                                edgecolor="none", transform=ax.transAxes))
    ax.text(0.03, 0.94, "Predicted: " + v, color="white", fontsize=15,
            weight="bold", transform=ax.transAxes)
    ax.set_title("PhishXAI - Page 1: Detection", fontsize=14, weight="bold",
                 loc="left")
    lines = [
        "URL (defanged, do not click):  " + defang(url),
        "Phishing probability:  %.4f   (threshold %.4f, model %s)"
        % (r["probability"], r["threshold"], dc.model_name),
        "Model prediction - not verified ground truth. Do not visit this URL.",
        "",
        "Top contributing factors (model behavior):",
    ]
    lines += ["  - " + s for s in r["sentences"][:5]]
    lines += ["",
              ("Coverage: %.0f%% - These points describe the model; "
               "not proof phishing/safe.") % (r["coverage"] * 100)]
    ax.text(0.03, 0.84, "\n".join(lines), fontsize=10, family="monospace",
            va="top", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)


def render_explainability(url, fname):
    r = dc.analyze(url)
    df = pd.DataFrame({"feature": FEATURE_NAMES, "shap": r["shap"]}).sort_values(
        "shap", ascending=False)
    sign = np.where(df["shap"] >= 0, "toward phishing", "away from phishing")
    colors = [REDC if s == "toward phishing" else GREENC for s in sign]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    ax = axes[0]
    x = np.arange(len(df))
    ax.bar(x, df["shap"], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(df["feature"], rotation=90, fontsize=7)
    ax.set_ylabel("SHAP contribution (log-odds)")
    ax.set_title("Per-feature SHAP (lr_v1)", fontsize=11, weight="bold")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=REDC)],
              ["toward phishing"], loc="lower left", fontsize=8)
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_title("Plain-language explanation (registry-driven)",
                  fontsize=11, weight="bold")
    txt = "\n".join("- " + s for s in r["sentences"][:5])
    txt += ("\n\nCoverage: %.0f%%" % (r["coverage"] * 100)
            + "\nbase=%.4f + sum(SHAP)=%.4f (residual %.1e)"
            % (r["base_value"], r["decision_function"], r["additivity_residual"]))
    ax2.text(0.02, 0.98, txt, fontsize=9.5, va="top", family="monospace",
             transform=ax2.transAxes)
    fig.suptitle("Page 2: Explainability - " + defang(url)[:38],
                 fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)


def render_performance(fname):
    b4 = json.loads((ROOT / "artifacts" / "b4_summary.json").read_text(
        encoding="utf-8"))
    sel, test = b4["selection"], b4["test"]

    def table(ax, results, keys):
        rows = [{"model": n, "PR-AUC": results[n]["pr_auc"],
                 "F1": results[n]["f1"], "acc": results[n]["accuracy"],
                 "FPR": results[n]["fpr"], "thr": results[n]["threshold"],
                 "n": results[n]["n_samples"]} for n in keys]
        t = pd.DataFrame(rows).set_index("model")
        ax.axis("off")
        tbl = ax.table(cellText=t.values, colLabels=list(t.columns),
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.7)

    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.34)

    axv = fig.add_subplot(gs[0, 0])
    axv.set_title("Validation (n=61) - used for model selection",
                  fontsize=12, weight="bold", color="#8b0000")
    table(axv, sel["val_results"], sel["ranking"])
    fig.text(0.25, 0.585,
             "Selected: %s (decision recorded %s, BEFORE any test access;"
             " B7 clean re-run re-verified the same choice at %s)"
             % (sel["selected_model"], sel["recorded_at_utc"][:19],
                sel.get("b7_reverified_at_utc", "n/a")[:19]),
             fontsize=8.5, ha="center", color="#333")

    axt = fig.add_subplot(gs[0, 1])
    axt.set_title("Test (n=60) - final report only",
                  fontsize=12, weight="bold", color="#333")
    table(axt, test, list(test.keys()))
    fig.text(0.75, 0.585, "These numbers informed no decision.",
             fontsize=9, ha="center", color="#333")

    axc = fig.add_subplot(gs[1, 0])
    cm = sel["val_results"]["lr_v1"]
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    axc.imshow(mat, cmap="Blues", aspect="auto")
    axc.set_xticks([0, 1])
    axc.set_xticklabels(["Pred: legit", "Pred: phish"])
    axc.set_yticks([0, 1])
    axc.set_yticklabels(["Actual: legit", "Actual: phish"])
    for i in range(2):
        for j in range(2):
            axc.text(j, i, str(mat[i, j]), ha="center", va="center",
                     color="white" if mat[i, j] > mat.max() / 2 else "black",
                     fontsize=14, weight="bold")
    axc.set_title("Confusion matrix - lr_v1 (val, threshold 0.2994)",
                  fontsize=12, weight="bold")

    axl = fig.add_subplot(gs[1, 1])
    axl.axis("off")
    axl.set_title("Known limitations (must appear in report)",
                  fontsize=12, weight="bold", color="#8b0000")
    axl.text(
        0.0, 0.97,
        ("- Legitimate class is 100% CONSTRUCTED: Tranco top-1000 domains\n"
         "   + a documented path pool. Not independently crawled / verified.\n"
         "- Small splits (val 61 / test 60): one URL moves F1/FPR 2-5 pts.\n"
         "- Bare-domain phishing (uspostaldirect[.]com) is a known blind spot.\n"
         "- No probability calibration; no host-based features (deferred).\n"
         "- Reproducibility verified on one Windows machine; no CI."),
        fontsize=8.6, va="top", family="monospace", transform=axl.transAxes,
        wrap=True)
    fig.suptitle("Page 3: Model Performance", fontsize=15, weight="bold",
                 y=0.98)
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)


render_detection(PHISH_URL, "01_detection_phishing.png")
render_detection(LEGIT_URL, "02_detection_legitimate.png")
render_explainability(PHISH_URL, "03_explainability_phishing.png")
render_performance("04_model_performance.png")
print("SCREENSHOTS_WRITTEN:", [p.name for p in sorted(OUT.iterdir())])
