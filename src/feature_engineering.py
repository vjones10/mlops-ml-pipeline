def feature_engineering(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    print("Feature engineering complete")
    return X, y
