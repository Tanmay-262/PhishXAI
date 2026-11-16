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
