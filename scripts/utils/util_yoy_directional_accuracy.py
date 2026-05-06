from statsmodels.stats.proportion import proportion_confint
import pandas as pd

# Overall YoY: 5/7
for label, k, n in [("Overall YoY", 5, 7), ("Pre-COVID", 1, 1), ("COVID", 1, 1), ("Post-COVID", 3, 4)]:
    lo, hi = proportion_confint(k, n, alpha=0.05, method='wilson')
    print(f"  {label}: {k}/{n} = {k/n:.1%}  95% CI [{lo:.1%}, {hi:.1%}]")