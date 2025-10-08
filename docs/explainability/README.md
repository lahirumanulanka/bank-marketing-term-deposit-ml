# Model Explainability Reports (Bank Marketing – Term Deposit)

This folder consolidates model explainability analyses produced from the Model Development notebook. It translates technical outputs into actionable insights for business and risk stakeholders.

Included techniques:
- SHAP (global and local)
- LIME (local)
- Permutation importance (model-agnostic)
- Partial dependence plots (feature effects)

The figures referenced below are saved under `reports/figures` when the notebook is executed.

## Executive summary

Across methods (feature importance from tree models, SHAP, permutation importance, PDPs), the most consistently influential features for predicting term deposit subscription are:

- duration (call duration)
- poutcome (outcome of previous marketing campaign)
- euribor3m, emp.var.rate, nr.employed (macroeconomic indicators)
- month (contact month)
- campaign (number of contacts in this campaign)
- pdays / previous (recency/volume of prior contacts)
- contact (contact channel)
- age, job, education (demographics; moderate, second-order effects)

Notes:
- duration is a strong post-call predictor but isn’t available pre-call (don’t use it for pre-call targeting). 
- Economic indicators materially shift baseline propensity; timing matters.

## Key findings by technique

- SHAP (global):
  - duration typically dominates global importance but should be excluded for pre-call targeting.
  - Positive SHAP contributions: successful poutcome, favorable macro (e.g., lower emp.var.rate, appropriate euribor3m regime), certain months.
  - Negative SHAP contributions: excessive campaign contacts, long pdays (stale leads), unfavorable macro regimes.
  - Figures: `reports/figures/shap_summary_plot.png`, `reports/figures/shap_bar_plot.png`, waterfall examples for local cases.

- LIME (local):
  - Provides per-customer rationale (top 10 features with signed weights).
  - Useful to audit borderline predictions and communicate case-level decisions.
  - Figures: `reports/figures/lime_explanation_tp.png`, `reports/figures/lime_explanation_tn.png`.

- Permutation importance (model-agnostic):
  - Confirms the above ordering; features that cause the largest AUC drop when shuffled: duration, poutcome, macro indicators, campaign/pdays.
  - Figure: `reports/figures/permutation_importance.png`.

- Partial dependence plots (PDPs):
  - Non-linear effects observed: diminishing returns or thresholds in campaign contacts; macro variables show regime-like behavior; month shows seasonality; age often U-shaped or monotonic mild effects.
  - Figure: `reports/figures/partial_dependence_plots.png`.

## Business implications and recommendations

- Campaign timing
  - Schedule pushes around favorable macro conditions (signals from euribor3m, emp.var.rate, nr.employed).
  - Prioritize historically strong months (e.g., Mar/Sep/Oct/Dec—validate with your data refresh).

- Lead prioritization
  - Boost priority for customers with successful poutcome; build re-engagement plays for warm leads.
  - Curb over-contacting (campaign) and stale lists (high pdays); set policy thresholds.

- Sales enablement
  - Emphasize engagement quality in scripts (duration correlates with conversion but is post-call; use it for QA and coaching).
  - Address macro concerns proactively in talking points.

- Governance and fairness
  - Monitor demographic influences (age, job, education) for fairness; impose usage and monitoring policies.
  - Keep duration out of pre-call targeting to avoid leakage and to comply with causal integrity.

- Operations
  - Maintain two models/workflows:
    1) Pre-call targeting model (exclude duration) for list scoring.
    2) Post-call analytics (include duration) for QA, training, and journey optimization.

- Risk mitigation
  - Implement contact-frequency caps and stale-lead suppression.
  - Track macro drift; set re-training triggers when macro indicators shift regimes.

## Traceability

These findings were generated from the notebook `notebooks/03_model_development.ipynb` under the section “Model Explainability and Interpretability.”
