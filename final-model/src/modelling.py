"""Model building, tuning, validation, OLS interpretation, multicollinearity screening and the
assessment-ratio (bias) study.

Target transform
----------------
Annual Value is strongly right-skewed (Rs. ~1,000 to ~300,000), so every model is fitted on
log(Annual Value) through sklearn's TransformedTargetRegressor. Predictions are converted back
to rupees automatically, and ALL reported metrics (RMSE, MAE, R2, ratios) are in rupees.
A raw-target sensitivity run is reported in the comparison notebook.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy.stats import randint
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBRegressor

from . import config as C

# ----------------------------------------------------------------------------
# Pipelines (all imputation / encoding inside the pipeline -> fitted on training folds only)
# ----------------------------------------------------------------------------
def condition_categories(observed):
    return [c for c in C.CONDITION_ORDER if c in set(pd.Series(observed).dropna())]


def build_preprocessor(key, features, categories):
    num = [f for f in features if f not in C.CATEGORICAL_FEATURES]
    cat = [f for f in features if f in C.CATEGORICAL_FEATURES]
    steps = [("num", SimpleImputer(strategy="median"), num)]
    if cat:
        if key == "mlr":   # dummies, 'Fair' = reference -> coefficients read "difference from Fair"
            enc = OneHotEncoder(categories=[categories], drop=[C.CONDITION_REFERENCE],
                                handle_unknown="ignore", sparse_output=False)
        else:              # ordered grade -> one ordinal column (Poor=0 ... Good=4)
            enc = OrdinalEncoder(categories=[categories], handle_unknown="use_encoded_value", unknown_value=np.nan)
        steps.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", enc)]), cat))
    return ColumnTransformer(steps, remainder="drop", verbose_feature_names_out=False)


def build_estimator(key, params=None):
    params = dict(params or {})
    if key == "mlr":
        return LinearRegression()
    if key == "rf":
        return RandomForestRegressor(random_state=C.RANDOM_STATE, n_jobs=1, **params)
    if key == "xgb":
        return XGBRegressor(random_state=C.RANDOM_STATE, n_jobs=1, objective="reg:squarederror",
                            tree_method="hist", **params)
    raise ValueError(key)


def build_model(key, features, categories, params=None, log_target=True):
    pipe = Pipeline([("preprocess", build_preprocessor(key, features, categories)),
                     ("model", build_estimator(key, params))])
    if not log_target:
        return pipe
    return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp, check_inverse=False)


def fitted_pipeline(model):
    """The fitted sklearn Pipeline inside a (possibly target-transformed) model."""
    return model.regressor_ if isinstance(model, TransformedTargetRegressor) else model


def feature_names_out(model):
    return list(fitted_pipeline(model).named_steps["preprocess"].get_feature_names_out())


# ----------------------------------------------------------------------------
# Hyperparameter tuning (training partition only, 5-fold CV)
# ----------------------------------------------------------------------------
RF_PARAM_DIST = {
    "n_estimators": [200, 300, 400, 500, 700],
    "max_depth": [None, 4, 6, 8, 10, 15, 20],
    "min_samples_split": randint(2, 11),
    "min_samples_leaf": randint(1, 9),
    "max_features": ["sqrt", "log2", 0.33, 0.5, 0.7, 1.0],
}
XGB_PARAM_DIST = {
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1],
    "max_depth": [2, 3, 4, 5, 6],
    "min_child_weight": [1, 2, 3, 5, 7],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
    "gamma": [0, 0.01, 0.05, 0.1],
    "n_estimators": [200, 300, 500, 700, 900],
    "reg_alpha": [0, 0.01, 0.1, 1],
    "reg_lambda": [1, 2, 5, 10],
}


def _py(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def tune(key, X, y, features, categories, log_target=True):
    """RandomizedSearchCV on the training rows. Score = RMSE in rupees. MLR has no hyperparameters."""
    if key == "mlr":
        return {}, pd.DataFrame()
    dist = RF_PARAM_DIST if key == "rf" else XGB_PARAM_DIST
    prefix = "regressor__model__" if log_target else "model__"
    search = RandomizedSearchCV(
        build_model(key, features, categories, log_target=log_target),
        {prefix + k: v for k, v in dist.items()},
        n_iter=C.RF_SEARCH_ITER if key == "rf" else C.XGB_SEARCH_ITER,
        scoring="neg_root_mean_squared_error",
        cv=KFold(C.CV_SPLITS, shuffle=True, random_state=C.RANDOM_STATE),
        random_state=C.RANDOM_STATE, n_jobs=-1, refit=False)
    search.fit(X[features], y)
    res = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    best = {k.replace(prefix, ""): _py(v) for k, v in search.best_params_.items()}
    return best, res


# ----------------------------------------------------------------------------
# Metrics and validation
# ----------------------------------------------------------------------------
def metrics(y_true, y_pred):
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred))}


def _fit_fold(i, tr, va, key, X, y, features, categories, params, permutation, log_target):
    m = build_model(key, features, categories, params, log_target).fit(X.iloc[tr][features], y.iloc[tr])
    pred = m.predict(X.iloc[va][features])
    rec = {"split": i, "repeat": i // C.CV_SPLITS + 1, "fold": i % C.CV_SPLITS + 1, **metrics(y.iloc[va], pred)}
    perm = None
    if permutation:
        r = permutation_importance(m, X.iloc[va][features], y.iloc[va], scoring="neg_root_mean_squared_error",
                                   n_repeats=C.PERMUTATION_REPEATS, random_state=C.RANDOM_STATE + i, n_jobs=1)
        perm = r.importances_mean
    oof = pd.DataFrame({"row": X.index[va], "repeat": rec["repeat"], "prediction": pred})
    return rec, oof, perm


def repeated_cv(key, X, y, features, categories, params=None, permutation=False, log_target=True):
    """Repeated 5-fold CV (10 repeats) on the training rows. Identical folds for every model and
    for Model A and Model B (same seed, same rows) -> paired comparison is possible."""
    folds = list(RepeatedKFold(n_splits=C.CV_SPLITS, n_repeats=C.CV_REPEATS,
                               random_state=C.RANDOM_STATE).split(X))
    out = Parallel(n_jobs=-1)(delayed(_fit_fold)(i, tr, va, key, X, y, features, categories, params,
                                                 permutation, log_target) for i, (tr, va) in enumerate(folds))
    fold_df = pd.DataFrame([o[0] for o in out])
    oof = pd.concat([o[1] for o in out], ignore_index=True)
    perm = pd.DataFrame([o[2] for o in out], columns=features) if permutation else None
    return fold_df, oof, perm


def pooled_oof_r2(oof, y):
    """R2 over all out-of-fold predictions of each repeat (every training row predicted once per
    repeat), averaged over the 10 repeats. Less sensitive than the per-fold mean to a fold that
    happens to contain one of the few very large properties."""
    return float(np.mean([r2_score(y.loc[g["row"]].values, g["prediction"].values)
                          for _, g in oof.groupby("repeat")]))


def summarise_cv(fold_df):
    return {"CV MAE": fold_df["MAE"].mean(), "CV MAE SD": fold_df["MAE"].std(ddof=1),
            "CV RMSE": fold_df["RMSE"].mean(), "CV RMSE SD": fold_df["RMSE"].std(ddof=1),
            "CV R2": fold_df["R2"].mean(), "CV R2 SD": fold_df["R2"].std(ddof=1)}


def oof_mean_prediction(oof, index):
    """Average out-of-fold prediction per training row (over the 10 repeats)."""
    return oof.groupby("row")["prediction"].mean().reindex(index)


def spatial_groups(X, n_clusters=C.SPATIAL_CLUSTERS):
    """No road field exists in the dataset, so properties are grouped into spatial clusters using
    their 11 GIS distances (which together fix a property's position)."""
    Z = StandardScaler().fit_transform(X[C.GIS_FEATURES])
    return KMeans(n_clusters=n_clusters, random_state=C.RANDOM_STATE, n_init=10).fit_predict(Z)


def spatial_cv(key, X, y, features, categories, params, groups, log_target=True):
    rows = []
    for i, (tr, va) in enumerate(GroupKFold(n_splits=C.CV_SPLITS).split(X, y, groups)):
        m = build_model(key, features, categories, params, log_target).fit(X.iloc[tr][features], y.iloc[tr])
        rows.append({"fold": i + 1, "n_valid": len(va), **metrics(y.iloc[va], m.predict(X.iloc[va][features]))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Assessment-ratio study (IAAO): ratio = model estimate / CMC-recorded Annual Value
# ----------------------------------------------------------------------------
def ratio_statistics(actual, predicted):
    a = np.asarray(actual, float)
    p = np.asarray(predicted, float)
    ratio = p / a
    med = np.median(ratio)
    cod = 100 * np.mean(np.abs(ratio - med)) / med
    prd = ratio.mean() / (p.sum() / a.sum())
    # PRB (IAAO 2013): regress % deviation from median ratio on log2 of a value proxy
    value = 0.5 * (p / med) + 0.5 * a
    if np.all(value > 0):
        X = sm.add_constant(np.log(value) / np.log(2))
        prb_fit = sm.OLS((ratio - med) / med, X).fit()
        prb, ci = float(prb_fit.params[1]), prb_fit.conf_int()[1]
    else:                       # only possible for the raw-target sensitivity run (negative estimates)
        prb, ci = np.nan, (np.nan, np.nan)
    return {"n": len(ratio), "Median ratio": med, "Mean ratio": ratio.mean(),
            "Weighted mean ratio": p.sum() / a.sum(), "COD (%)": cod, "PRD": prd,
            "PRB": prb, "PRB 95% CI low": float(ci[0]), "PRB 95% CI high": float(ci[1]),
            "Negative estimates": int((p <= 0).sum()),
            "Within +/-10% (%)": 100 * np.mean(np.abs(ratio - 1) <= 0.10),
            "Within +/-20% (%)": 100 * np.mean(np.abs(ratio - 1) <= 0.20)}


def iaao_verdict(row):
    ok = []
    lo, hi = C.IAAO_MEDIAN_RATIO
    ok.append("level OK" if lo <= row["Median ratio"] <= hi else "level outside 0.90-1.10")
    ok.append("uniformity OK" if row["COD (%)"] <= C.IAAO_COD_MAX else f"COD > {C.IAAO_COD_MAX:.0f}")
    lo, hi = C.IAAO_PRD
    if row["PRD"] > hi:
        ok.append("PRD regressive")
    elif row["PRD"] < lo:
        ok.append("PRD progressive")
    else:
        ok.append("PRD OK")
    return "; ".join(ok)


def ratio_by_value_band(actual, predicted, n_bands=4):
    d = pd.DataFrame({"actual": np.asarray(actual, float), "pred": np.asarray(predicted, float)})
    d["ratio"] = d["pred"] / d["actual"]
    d["Band"] = pd.qcut(d["actual"].rank(method="first"), n_bands, labels=[f"Q{i}" for i in range(1, n_bands + 1)])
    g = d.groupby("Band", observed=True).agg(n=("ratio", "size"), AV_min=("actual", "min"), AV_max=("actual", "max"),
                                             Median_ratio=("ratio", "median"))
    return g.reset_index()


# ----------------------------------------------------------------------------
# Multicollinearity (VIF) and the screened GIS set used by MLR in Model B
# ----------------------------------------------------------------------------
def vif_table(d, numeric, condition=True):
    X = d[numeric].copy()
    X = X.fillna(X.median())
    if condition:
        cond = d["Condition"].fillna(d["Condition"].mode().iloc[0])
        for lvl in [c for c in C.CONDITION_ORDER if c != C.CONDITION_REFERENCE and (cond == c).any()]:
            X[f"Condition_{lvl}"] = (cond == lvl).astype(float)
    X.insert(0, "const", 1.0)
    rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        try:
            v = variance_inflation_factor(X.values, i)
        except Exception:
            v = np.inf
        rows.append({"Variable": col, "VIF": v, "VIF > 5": bool(v > C.VIF_THRESHOLD)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def screened_gis_set(train):
    """Rule (training rows only):
    1. rank GIS variables by |Pearson r| with log(Annual Value);
    2. keep a variable only if |r| < 0.70 with every variable already kept;
    3. while any kept GIS variable has VIF > 5 (with the property variables), drop the highest.
    Property variables are always kept. Used for MLR only; RF / XGBoost use all 11 distances."""
    ly = np.log(train[C.TARGET])
    r_t = {g: abs(np.corrcoef(train[g], ly)[0, 1]) for g in C.GIS_FEATURES}
    ranked = sorted(C.GIS_FEATURES, key=lambda g: -r_t[g])
    corr = train[C.GIS_FEATURES].corr()
    kept, dec = [], {}
    for g in ranked:
        clash = [k for k in kept if abs(corr.loc[g, k]) >= C.CORR_THRESHOLD]
        if clash:
            dec[g] = ("Dropped", f"|r| = {abs(corr.loc[g, clash[0]]):.2f} with retained {clash[0]}")
        else:
            kept.append(g)
            dec[g] = ("Retained", "No |r| >= 0.70 with a retained variable; VIF <= 5")
    prop_num = ["Age", "Log_Total_Floor_Area", "Num_Floors"]
    while True:
        v = vif_table(train, prop_num + kept)
        vg = v[v["Variable"].isin(kept)]
        if vg.empty or vg["VIF"].max() <= C.VIF_THRESHOLD:
            break
        worst = vg.iloc[0]
        kept.remove(worst["Variable"])
        dec[worst["Variable"]] = ("Dropped", f"VIF {worst['VIF']:.1f} > 5 after correlation screening")
    table = pd.DataFrame([{"Variable": g, "Group": C.GIS_GROUPS[g],
                           "|r| with log(Annual Value)": r_t[g], "Decision": dec[g][0], "Reason": dec[g][1]}
                          for g in ranked])
    return [g for g in C.GIS_FEATURES if g in kept], table


def feature_sets(train):
    """Predictor lists per (model set, algorithm)."""
    screened, screen_table = screened_gis_set(train)
    sets = {}
    for key in C.ALGORITHMS:
        prop = C.PROPERTY_FEATURES_MLR if key == "mlr" else C.PROPERTY_FEATURES
        sets[("A", key)] = list(prop)
        sets[("B", key)] = prop + (screened if key == "mlr" else C.GIS_FEATURES)
    return sets, screened, screen_table


# ----------------------------------------------------------------------------
# OLS (statsmodels) for interpretation of the MLR - same coding as the sklearn MLR
# ----------------------------------------------------------------------------
def ols_fit(train, features, categories):
    num = [f for f in features if f not in C.CATEGORICAL_FEATURES]
    X = train[num].copy()
    X = X.fillna(X.median())
    if "Condition" in features:
        cond = train["Condition"].fillna(train["Condition"].mode().iloc[0])
        for lvl in categories:
            if lvl != C.CONDITION_REFERENCE:
                X[f"Condition_{lvl}"] = (cond == lvl).astype(float)
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(np.log(train[C.TARGET].values), X).fit()
    ci = model.conf_int()
    table = pd.DataFrame({"Term": X.columns, "Coefficient (log scale)": model.params.values,
                          "Std. error": model.bse.values, "t": model.tvalues.values, "p-value": model.pvalues.values,
                          "95% CI low": ci[0].values, "95% CI high": ci[1].values})
    table["Effect on Annual Value (%)"] = 100 * (np.exp(table["Coefficient (log scale)"]) - 1)
    table.loc[table["Term"] == "const", "Effect on Annual Value (%)"] = np.nan
    # readable units: distances per 100 m, log floor area per +10 % area
    table["Readable effect"] = ""
    for i, t in table["Term"].items():
        b = table.at[i, "Coefficient (log scale)"]
        if t.startswith("Dist_"):
            table.at[i, "Readable effect"] = f"{100 * (np.exp(100 * b) - 1):+.2f} % per +100 m"
        elif t == "Log_Total_Floor_Area":
            table.at[i, "Readable effect"] = f"{100 * (1.1 ** b - 1):+.2f} % per +10 % floor area (elasticity {b:.3f})"
        elif t == "Age":
            table.at[i, "Readable effect"] = f"{100 * (np.exp(10 * b) - 1):+.2f} % per +10 years"
        elif t == "Num_Floors":
            table.at[i, "Readable effect"] = f"{100 * (np.exp(b) - 1):+.2f} % per extra floor (same total area)"
        elif t.startswith("Condition_"):
            table.at[i, "Readable effect"] = f"{100 * (np.exp(b) - 1):+.2f} % vs {C.CONDITION_REFERENCE}"
    stats = {"n": int(model.nobs), "R2 (log scale)": model.rsquared, "Adj. R2 (log scale)": model.rsquared_adj,
             "F p-value": model.f_pvalue, "AIC": model.aic, "Condition number": float(np.linalg.cond(X.values))}
    return model, table, stats
