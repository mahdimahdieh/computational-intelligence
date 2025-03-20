from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=8)
imputed_X = imputer.fit_transform(features)
features = pd.DataFrame(imputed_X, columns=features.columns)