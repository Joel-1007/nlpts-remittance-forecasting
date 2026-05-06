import pandas as pd
fc = pd.read_csv('/kaggle/working/baseline_forecasts.csv')
cov = ((fc['actual'] >= fc['sarima_lower_95']) & (fc['actual'] <= fc['sarima_upper_95'])).mean()
print(f"95% PI coverage: {cov:.1%}")

# Also check width
fc['pi_width'] = fc['sarima_upper_95'] - fc['sarima_lower_95']
print(f"Mean PI width: ${fc['pi_width'].mean():,.0f}M")
print(f"Mean actual: ${fc['actual'].mean():,.0f}M")
print(f"PI width as % of actual: {(fc['pi_width']/fc['actual']).mean():.1%}")