columns_to_encode = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Var_1']

features = pd.get_dummies(
    features,
    columns=columns_to_encode,
    prefix=columns_to_encode,
    drop_first=True 
)
features.head()