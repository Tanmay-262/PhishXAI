# B4 Error Analysis (defanged, report-safe)

All metrics below: validation split n=61 (33 phishing / 28 legitimate),
test split n=60 (29 phishing / 31 legitimate). Selection was val-only and
recorded BEFORE test access (artifacts/selection_decision.json).

## Confusion matrices at each model's val-selected threshold

lr_v1 (thr 0.2994)
| split | TP | FP | TN | FN |
|---|---|---|---|---|
| val  | 33 | 1 | 27 | 0 |
| test | 28 | 1 | 30 | 1 |

rf_v1 (thr 0.45)
| split | TP | FP | TN | FN |
|---|---|---|---|---|
| val  | 33 | 2 | 26 | 0 |
| test | 28 | 0 | 31 | 1 |

xgb_v1 (thr 0.3424)
| split | TP | FP | TN | FN |
|---|---|---|---|---|
| val  | 33 | 2 | 26 | 0 |
| test | 28 | 0 | 31 | 1 |

## Representative errors (defanged; do NOT make clickable)

- **Shared false negative (all three models, test):** `uspostaldirect[.]com`
  (bare-domain phishing URL, no path). All models miss it: with no path and
  a plausible-looking brand-spoof domain, the 19 lexical features give it
  no strong phishing signal. This is a known limitation of lexical-only
  detection; host-based features (deferred) are the documented fix.
- **lr_v1 false positive (test):** `shopee[.]co[.]id/images/hero[.]png` -
  a SYNTHETICALLY CONSTRUCTED legitimate URL (path pool). The `/images/`
  pattern plus length pushed LR over threshold. Expected artifact of the
  constructed-legit design; disclosed, not hidden.
- **val false positives (RF/XGB):** `youtube-nocookie[.]com` (bare) and a
  constructed `t-s1-msedge[.]net/blog/2026/ann...` URL - same story:
  bare/short constructed legits near the threshold.

## Limitations (must appear in the final report)

1. Legit class is 100% constructed (B2 disclosure): FPR estimates and the
   FP examples above are artifacts of the construction method, not observed
   web behavior.
2. n=61/60 splits: FP/FN counts are single samples; a one-URL difference
   moves F1/FPR by 2-5 points. Do not over-read small metric gaps.
3. No probability calibration: verdicts use raw model probabilities at the
   val-F1 threshold.
4. Bare-domain phishing URLs are the systematic blind spot (1 FN).
