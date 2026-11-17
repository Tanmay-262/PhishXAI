r"""PhishXAI minimal dashboard (B6) - thin Streamlit wrapper.

All prediction/explanation logic lives in phishxai.dashboard_core (which
delegates to the B3/B4/B5 modules). This file only renders.

Run:
    .venv\Scripts\streamlit run app\streamlit_app.py

No network access: submitted URLs are never fetched, visited or resolved.
"""
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phishxai.dashboard_core import DashboardCore, MissingArtifactError  # noqa: E402
from phishxai.features.extractor import FEATURE_NAMES  # noqa: E402

st.set_page_config(page_title="PhishXAI", page_icon=":lock:", layout="wide")

# ------------------------------------------------------------------ model
@st.cache_resource
def core() -> DashboardCore:
    return DashboardCore("lr_v1")  # B4-selected model; do NOT re-select

core_model = None
try:
    core_model = core()
except MissingArtifactError as e:
    st.error(f"Model artifact unavailable: {e}")
    st.stop()

DISCLAIMER = ("PhishXAI's verdict is a model prediction, not a verified "
              "ground truth. A 'legitimate' verdict does **not** guarantee "
              "the URL is safe. Do not visit flagged URLs. Explanations "
              "describe why the model made its prediction; they do not "
              "establish causation or prove the prediction is correct.")

tab_det, tab_exp, tab_perf = st.tabs(["Detection", "Explainability",
                                       "Model Performance"])
st.caption(DISCLAIMER)

# ------------------------------------------------------------------ P1
with tab_det:
    st.header("Page 1 - Detection")
    url = st.text_input("Enter a URL to analyze", placeholder="https://example.com",
                        key="url_input")
    if st.button("Analyze URL"):
        u = url.strip()
        if not u:
            st.warning("Enter a URL first.")
        else:
            bad = re.match(r"^(javascript|vbscript|file|data|ftp):", u.lower())
            if bad:
                st.error(f"Unsupported URL scheme ('{bad.group(1)}'). "
                         "Enter an http/https or bare URL.")
            else:
                r = core_model.analyze(u)
                st.session_state.last_analysis = r
                v = r["verdict"]
                color = ":red-background" if v == "phishing" else ":green-background"
                if v == "legitimate":
                    st.success(f"Predicted: **{v}** (reduced phishing signals - "
                               f"not a safety guarantee)")
                else:
                    st.error(f"Predicted: **{v}**")
                st.markdown(f"URL shown (defanged, do not click): "
                            f"`{r['url_defanged']}`")
                st.markdown(f"Phishing probability: **{r['probability']:.4f}**  "
                            f"(decision threshold: {r['threshold']:.4f}, "
                            f"model: {core_model.model_name})")
                st.caption("Model prediction - not verified ground truth. "
                           "Do not visit this URL.")
                if r["sentences"]:
                    st.subheader("Top contributing factors (model behavior)")
                    for s in r["sentences"]:
                        st.markdown("- " + s)
                    st.caption(f"Explanation coverage: {r['coverage']:.0%} "
                               f"(top-5). {r['disclaimer']}")
                else:
                    st.info("No dominant features - the model was near "
                            "undecided for this URL.")
                with st.expander("All 19 feature values used by the model"):
                    fv = pd.DataFrame([r["feature_values"][n] for n in FEATURE_NAMES],
                                       index=FEATURE_NAMES)
                    st.dataframe(fv)
    st.markdown(f"> {DISCLAIMER}")

# ------------------------------------------------------------------ P2
with tab_exp:
    st.header("Page 2 - Explainability")
    if "last_analysis" in st.session_state:
        r = st.session_state.last_analysis
        st.markdown(f"Explaining: `{r['url_defanged']}`  "
                    f"(verdict: {r['verdict']}, p={r['probability']:.4f})")
        st.caption("SHAP contributions are in **log-odds space** "
                   "(they add to the model's log-odds score), not "
                   "probabilities. Positive = pushes toward phishing; "
                   "negative = pushes toward legitimate. A low score "
                   "does not mean the URL is safe.")
        # bar chart of contributions
        df = pd.DataFrame({"feature": FEATURE_NAMES,
                            "shap_log_odds": r["shap"]}).sort_values(
            "shap_log_odds", ascending=False)
        df["sign"] = ["toward phishing" if v >= 0 else "away from phishing"
                      for v in df["shap_log_odds"]]
        fig = px.bar(df, x="feature", y="shap_log_odds", color="sign",
                     color_discrete_map={"toward phishing": "#d62728",
                                          "away from phishing": "#2ca02c"},
                     title="Per-feature SHAP contribution (log-odds), "
                           "lr_v1",
                     labels={"feature": "feature", "shap_log_odds":
                             "SHAP contribution (log-odds)"})
        fig.update_layout(height=420, xaxis_tickangle=-45)
        st.plotly_chart(fig, width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Pushing toward phishing (positive)**")
            for n, v in r["top_positive"]:
                st.markdown(f"- `{n}`: {v:+.4f}")
            if not r["top_positive"]:
                st.caption("None in the top-5.")
        with c2:
            st.markdown("**Pushing away from phishing (negative)**")
            for n, v in r["top_negative"]:
                st.markdown(f"- `{n}`: {v:+.4f}")
            if not r["top_negative"]:
                st.caption("None in the top-5.")

        st.markdown("**Plain-language explanation (registry-driven)**")
        for s in r["sentences"]:
            st.markdown("- " + s)
        st.caption(f"Top-5 coverage: {r['coverage']:.0%} | {r['disclaimer']}")

        st.markdown("**Grouped concepts (signed SHAP sums - presentation "
                    "only; raw values above are authoritative)**")
        gc = pd.DataFrame(r["grouped_concepts"])
        st.dataframe(gc, width="stretch")
        add = r["additivity_residual"]
        st.caption(f"Additivity check: base ({r['base_value']:.4f}) + "
                   f"sum(SHAP) = decision function "
                   f"({r['decision_function']:.4f}); residual {add:.2e} "
                   f"(tolerance 1e-4).")
    else:
        st.info("Analyze a URL on Page 1 first - explanations appear here.")
        st.caption(DISCLAIMER)

# ------------------------------------------------------------------ P3
with tab_perf:
    st.header("Page 3 - Model Performance")
    st.caption("Recorded B4 results. Selection was made on the "
               "VALIDATION split BEFORE any test evaluation; the test "
               "numbers below are final-report-only and did not influence "
               "the chosen model.")
    try:
        perf = core_model.performance_data()
    except MissingArtifactError as e:
        st.error(str(e))
        st.stop()

    sel = perf["selection"]
    st.markdown(f"**Selected model:** `{sel['selected_model']}` - "
                f"{sel['rule']} (decision recorded "
                f"{sel['recorded_at_utc']}, pre-test). "
                f"{sel['caveat']}")

    st.subheader("Validation (used for model selection)")
    vtab = pd.DataFrame(
        [{k: m[k] for k in ("pr_auc", "accuracy", "precision", "recall",
                             "f1", "fpr", "threshold", "n_samples",
                             "n_phishing", "n_legitimate")}
         for m in sel["val_results"].values()])
    vtab = vtab.reindex(sel["ranking"]).reset_index().rename(
        columns={"index": "model (ranked)"})
    st.dataframe(vtab, width="stretch")
    st.caption("val n=61 (33 phishing / 28 legitimate). Differences of "
               "a few points are within uncertainty at this size.")

    st.subheader("Test (final report only - no tuning from this)")
    ttab = pd.DataFrame(
        [{k: m[k] for k in ("pr_auc", "accuracy", "precision", "recall",
                             "f1", "fpr", "threshold", "n_samples",
                             "n_phishing", "n_legitimate")}
         for m in perf["b4_summary"]["test"].values()])
    st.dataframe(ttab, width="stretch")
    st.caption("test n=60 (29 phishing / 31 legitimate).")

    st.subheader("Confusion matrices (selected model, at its threshold)")
    for which, key in (("Validation (selection)", "val_results"),
                       ("Test (report only)", "b4_summary_test")):
        if which.startswith("Validation"):
            cm = sel["val_results"]["lr_v1"]
        else:
            cm = perf["b4_summary"]["test"]["lr_v1"]
        mat = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        fig = px.imshow(mat, x=["Pred: phishing", "Pred: legitimate"],
                        y=["Actual: phishing", "Actual: legitimate"],
                        labels=dict(x="predicted", y="actual"),
                        text_auto=True, title=which + " - lr_v1")
        fig.update_xaxes(side="top")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"TP={cm['tp']} FP={cm['fp']} TN={cm['tn']} FN={cm['fn']}")

    st.subheader("Known limitations (must appear in your report)")
    st.markdown(
        "- **Legitimate class is 100% synthetically constructed** "
        "(Tranco top-1000 domains + documented path templates). FPR and "
        "the false positive shown in B4 are artifacts of that construction; "
        "no Tranco URL was crawled or verified.\n"
        "- **Small evaluation splits** (val n=61, test n=60): one URL "
        "moves F1/FPR by 2-5 points.\n"
        "- **Known blind spot:** bare-domain phishing URLs such as "
        "`uspostaldirect[.]com` are missed by all three B4 models.\n"
        "- No probability calibration; no host-based features "
        "(deferred).")
