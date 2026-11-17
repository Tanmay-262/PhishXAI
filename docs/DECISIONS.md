# PhishXAI - Decision Log

| Date | Decision | Rationale | Alternatives |
|------|----------|-----------|--------------|
| 2026-11-17 | Keep system Python 3.14 for the venv | All 5 essential packages import successfully on 3.14.0 | Python 3.12 side-by-side (only if the gate had failed) |
| 2026-11-17 | FastAPI layer deferred past MVP deadline | User decision 5; dashboard can import the core package directly | Implement POST /predict (contract documented in plan, not started) |
| 2026-11-17 | Human user study deferred | Deadline (2026-11-18); requires guide/institutional approval first | None at MVP scope |
| 2026-11-17 | Development stays local; no GitHub push without user approval | User instruction | - |

## Open decisions

- [ ] B2: final dataset sources + counts (user approval gate)
- [ ] B4: model selection metric (default: highest PR-AUC, tie-break F1)

| 2026-11-17 | Registry = 19 features (target was ~15) | Correctness + coverage of user-required edge cases (IP, ports, @, encoding, keywords) mattered more than hitting 15 | Drop to 15 (would lose edge-case features) |
| 2026-11-17 | PSL bundled at src/phishxai/features/assets/ (CC0, 341 KB) | Fresh clone must reproduce extraction without gitignored data/raw; PSL is reference data, not a dataset | Point at data/raw only (fresh clone breaks) |

| 2026-11-17 | LR (lr_v1) selected as final model | Highest val PR-AUC (0.9991 vs 0.9955/0.9937) with best val F1; also most interpretable for SHAP | RF / XGB (higher test PR-AUC 0.9958/0.9989 - noted, but selection was val-only per protocol) |
| 2026-11-17 | Thresholds via argmax F1 on val (tie: higher recall, then lower threshold) | Single documented rule, val-only; avoids hand-picking 0.5 | Fixed 0.5 (would give F1 0.9565 for LR - 2.9% worse at n=61) |
| 2026-11-17 | No probability calibration in B4 | Deadline; thresholds already absorb calibration needs for verdicts | Calibrated probabilities (deferred, noted in status) |
| 2026-11-17 | Selection decision recorded to artifacts/selection_decision.json before test access | Enforces the "no test-driven selection" protocol | Ad-hoc ordering |

| 2026-11-17 | shap.LinearExplainer (not TreeSHAP) for lr_v1 | lr_v1 is a scaled linear model, not a tree; LinearExplainer gives EXACT SHAP values in log-odds space with O(19) per row, additivity to machine precision | TreeSHAP (wrong model family); DeepExplainer (overkill) |
| 2026-11-17 | Full train split (479 rows) as SHAP background | For a linear model the background only fixes the base value and the "mean reference" for local contributions; full-train is the most faithful reference and needs no RNG sub-sampling | 100-sample background (shap default) - would add an RNG-dependent subsample |
| 2026-11-17 | Keep all 19 features in SHAP; add GROUPED presentation concepts | User instruction: do not hide correlation evidence by dropping features; raw feature-level contributions always available for inspection; signed-sum groups (overall length, domain structure, digits, ...) give a readable view | Drop collinear features from explanations (would misreport their importance) |
| 2026-11-17 | Registry v1.1: phrase-only enrichment (16 toward_legitimate + has_https both directions) | B3 registry had many null "looks safe" phrases -> 51.8% top-5 coverage; enrichment raised val coverage to 100%; feature names/types/order and the trained model are unchanged (bundle does not embed phrases) | Keep nulls and rely on fallback sentences (coverage 0.518, weaker demo) |

| 2026-11-17 | Dashboard = thin Streamlit wrapper; logic in phishxai.dashboard_core | User requirement: do not duplicate extraction/prediction/SHAP/translation; keeps Streamlit swappable and testable without a browser | Put logic in the .py app (untestable core) |
| 2026-11-17 | Registry v1.2: value-aware phrases for 12 continuous features ({value} substitution), phrase-only | B6 requirement: continuous phrases must describe the observed value; fail-closed if a value-aware phrase lacks its value (ValueError) | Keep magnitude-only phrases (risked "unusually long" sentences on short domains) |
| 2026-11-17 | No network access in dashboard, enforced by socket-monkeypatch tests + 7-scenario denial run | User requirement: submitted URLs are never fetched/visited/resolved | Advisory only (unverifiable) |
| 2026-11-17 | Page 3 renders val and test metrics SEPARATELY with the pre-test selection decision | User requirement: do not re-select on test performance | Merged table (would blur the selection boundary) |

| 2026-11-17 | B7 screenshots rendered via the real pipeline (matplotlib), not live browser grabs | CUA/browser automation unavailable in this environment (native pipe down); rendered figures use the same DashboardCore data and defang URLs; AppTest proves all 3 tabs load with 0 exceptions | Fake/mock screenshots (rejected) |
| 2026-11-17 | B7 clean re-run re-derived selection; original B4 timestamp preserved, re-run recorded as b7_reverified_at_utc | User rule: preserve the pre-test selection decision AND its timestamp; the re-run is itself the reproducibility evidence | Let the re-run overwrite the B4 timestamp |
| 2026-11-17 | socket-denial test scoped to direct analyze() paths, not the AppTest test | AppTest's session machinery legitimately needs sockets; the no-network guarantee for submitted URLs lives entirely in the DashboardCore/extractor code paths, which are covered | Leave module-wide autouse patch (broke AppTest) |
