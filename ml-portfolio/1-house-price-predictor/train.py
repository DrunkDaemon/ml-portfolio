"""Compare, tune and evaluate models that predict California house prices."""
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from features import add_features

SEED = 42
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target  # target is in units of $100,000
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
fe = lambda: FunctionTransformer(add_features)

# 1) Compare several models with 3-fold cross-validation
models = {
    "Baseline (mean)": DummyRegressor(),
    "Ridge": make_pipeline(fe(), StandardScaler(), Ridge()),
    "RandomForest": make_pipeline(fe(), RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=SEED)),
    "HistGradientBoosting": make_pipeline(fe(), HistGradientBoostingRegressor(random_state=SEED)),
}
print("Cross-validated RMSE (lower is better)")
for name, m in models.items():
    cv = -cross_val_score(m, X_train, y_train, cv=3, scoring="neg_root_mean_squared_error")
    print(f"  {name:22s} {cv.mean():.3f} +/- {cv.std():.3f}")

# 2) Hyperparameter-tune the best family (gradient boosting)
search = RandomizedSearchCV(
    models["HistGradientBoosting"],
    {
        "histgradientboostingregressor__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "histgradientboostingregressor__max_iter": [200, 400, 600],
        "histgradientboostingregressor__max_leaf_nodes": [15, 31, 63],
        "histgradientboostingregressor__min_samples_leaf": [10, 20, 40],
        "histgradientboostingregressor__l2_regularization": [0.0, 0.1, 1.0],
    },
    n_iter=15, cv=3, scoring="neg_root_mean_squared_error", random_state=SEED, n_jobs=-1,
)
search.fit(X_train, y_train)
best = search.best_estimator_
print("\nBest params:", {k.split("__")[1]: v for k, v in search.best_params_.items()})

# 3) Final evaluation on held-out test data
pred = best.predict(X_test)
metrics = {
    "MAE_usd": float(mean_absolute_error(y_test, pred) * 100_000),
    "RMSE_usd": float(np.sqrt(mean_squared_error(y_test, pred)) * 100_000),
    "R2": float(r2_score(y_test, pred)),
}
print("\nTEST RESULTS:", json.dumps(metrics, indent=2))

# 4) Plots: predictions vs actual, and which features matter (permutation importance)
plt.figure(figsize=(5, 5))
plt.scatter(y_test, pred, s=4, alpha=0.3)
plt.plot([0, 5], [0, 5], "r--")
plt.xlabel("Actual ($100k)"); plt.ylabel("Predicted ($100k)"); plt.title("Predicted vs Actual")
plt.tight_layout(); plt.savefig("pred_vs_actual.png", dpi=120); plt.close()

imp = permutation_importance(best, X_test, y_test, n_repeats=5, random_state=SEED, n_jobs=-1)
order = imp.importances_mean.argsort()
plt.figure(figsize=(6, 4))
plt.barh(X_test.columns[order], imp.importances_mean[order])
plt.title("Feature importance (permutation)"); plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120); plt.close()

# 5) Save model + slider ranges for the app
stats = {c: [float(X[c].quantile(.01)), float(X[c].median()), float(X[c].quantile(.99))] for c in X.columns}
joblib.dump({"model": best, "stats": stats, "metrics": metrics}, "model.joblib")
json.dump(metrics, open("results.json", "w"), indent=2)
print("Saved model.joblib, results.json, pred_vs_actual.png, feature_importance.png")
