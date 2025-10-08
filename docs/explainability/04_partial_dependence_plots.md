# Partial Dependence Plots (PDPs)

Figure: `reports/figures/partial_dependence_plots.png`

PDPs show the marginal effect of a feature on the model’s predicted probability, averaging over other features.

## Observed patterns (typical for this dataset)

- campaign (contact count): diminishing returns; beyond a small number of contacts, probability often drops.
- pdays / previous: long recency reduces propensity; recent prior contact helps.
- month: seasonal effects (higher propensities in certain months); verify with current data.
- Macro indicators:
  - emp.var.rate, euribor3m, nr.employed: regime-like relationships; certain ranges favor conversion.
- age: mild monotonic or U-shaped patterns; second-order compared to engagement/macro.

## Practical use

- Set policy thresholds (e.g., max campaign contacts before pausing).
- Time campaigns when macro PDPs indicate favorable regimes.
- Calibrate messages and eligibility rules using PDP breakpoints.
