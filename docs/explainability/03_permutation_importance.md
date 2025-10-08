# Permutation Importance (Model-Agnostic)

Computed on the tuned XGBoost model using ROC-AUC as the scoring metric.

Figure: `reports/figures/permutation_importance.png`

## Highlights

- Largest AUC drops when shuffled: duration, poutcome, macro indicators (euribor3m, emp.var.rate, nr.employed), followed by campaign/pdays and month.
- Confirms the ranking and themes seen in SHAP and built-in feature importance.

## Notes

- Because permutation importance measures performance degradation, it reflects actual predictive reliance, independent of scale or cardinality.
- For pre-call use-cases, recompute permutation importance with duration removed to avoid leakage into targeting.
