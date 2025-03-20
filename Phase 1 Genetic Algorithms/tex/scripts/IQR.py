df = pd.read_csv('Customer Classification dataset/Train.csv')

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

numeric_cols.remove('ID')

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
