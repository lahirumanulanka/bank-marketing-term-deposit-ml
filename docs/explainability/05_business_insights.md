# Consolidated Insights: Features and Business Actions

This note integrates findings across explainability techniques and maps them to practical actions.

## Top features (consistent across methods)

1) duration (post-call) – strongest global signal; use only for QA and training.
2) poutcome – successful prior outcomes strongly increase conversion odds.
3) Macro indicators (euribor3m, emp.var.rate, nr.employed) – timing is material.
4) campaign / pdays / previous – saturation and recency dynamics.
5) month – seasonal lift patterns.
6) contact channel – channel effects.
7) demographics (age, job, education) – moderate effects; monitor for fairness.

## Do’s and Don’ts

- Do maintain two workflows:
  - Pre-call targeting (exclude duration)
  - Post-call analytics (include duration for engagement coaching)
- Do cap contact frequency and suppress stale leads (policy on campaign/pdays).
- Do time campaigns with macro conditions; monitor drift and re-train on regime shifts.
- Don’t use duration for pre-call scoring (information leakage and causality concerns).
- Don’t over-index on demographics; include fairness monitoring and guardrails.

## Policy recommendations

- Contact policy: cap campaign contacts at a small number; re-queue or pause after threshold.
- Recency policy: suppress high-pdays leads; prioritize recent interactions and successful poutcome.
- Timing policy: maintain a macro dashboard; enable/disable campaigns by macro regime.
- Governance: publish model cards; track group-level performance; set retraining SLAs.

## Ops checklist

- Refresh macro variables periodically; trigger re-train when regime changes.
- Keep encoders/scalers versioned (see `models/preprocessing/`).
- Log SHAP summaries for each monthly refresh; compare drift in top features.
