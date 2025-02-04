from sklearn.ensemble import ExtraTreesRegressor


def regression_model() -> ExtraTreesRegressor:
    ml_model = ExtraTreesRegressor(
        criterion="squared_error",
        n_estimators=200,
        max_depth=15,
        max_features=None,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    return ml_model
