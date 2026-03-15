import pandas as pd

pred = pd.read_csv('customer_predictions.csv')

new_row = {
    'customer_id':  'CUST99999',
    'churn_prob':   0.96,
    'risk_level':   'HIGH',
    'top_reason_1': 'friction_score (impact: +2.17)',
    'top_reason_2': 'tenure_months (impact: +1.35)',
    'top_reason_3': 'watch_hours_per_week (impact: +1.05)',
    'clv_score':    500.0,
    'actual_churn': 1
}

pred = pd.concat([pred, pd.DataFrame([new_row])], ignore_index=True)
pred.to_csv('customer_predictions.csv', index=False)
print('Done! CUST99999 added to predictions.csv')
