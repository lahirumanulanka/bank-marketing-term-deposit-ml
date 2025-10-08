# LIME: Local Explanations for Individual Predictions

This report summarizes the LIME explanations generated for the tuned XGBoost model.

Figures:
- `reports/figures/lime_explanation_tp.png`
- `reports/figures/lime_explanation_tn.png`

## What LIME adds

- Case-level transparency: shows top 10 features (with signed weights) that locally explain a single prediction.
- Complements SHAP by providing an alternative local linear surrogate near the instance.

## Observations

- True Positive case: higher propensity is explained by positive local weights for variables like successful poutcome, appropriate macro ranges, and engagement proxies.
- True Negative case: reduced propensity tied to saturation/recency signals (high campaign, long pdays) and unfavorable macro periods.

## Usage recommendations

- Use LIME during model QA and when you need concise, human-readable local rationales.
- Keep feature definitions consistent and ensure categorical encoding mapping is included (see `models/preprocessing/label_encoders.pkl`).
