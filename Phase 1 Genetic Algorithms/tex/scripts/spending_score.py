score_mapping = {
    'Low': 0,
    'Average': 1,
    'High': 2
}
features['Spending_Score'] = features['Spending_Score'].map(score_mapping)
features.head()