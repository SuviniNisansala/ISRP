"""Runs one model set (A = property only, B = property + GIS) end to end with identical
procedures: tuning on the training rows, repeated 5-fold CV, final fit, ONE hold-out test
evaluation, ratio study, and saving of models / tables / predictions.

Notebook 01 calls run_set("A"), notebook 02 calls run_set("B"); notebook 03 loads both.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import modelling as M


def run_set(set_key, data, features_by_alg, categories, log_target=True, permutation=True, save=True, verbose=True):
    train, test = data["train"], data["test"]
    ytr, yte = train[C.TARGET], test[C.TARGET]
    results = {}
    for key in C.ALGORITHMS:
        t0 = time.time()
        feats = features_by_alg[(set_key, key)]
        params, search = M.tune(key, train, ytr, feats, categories, log_target)
        folds, oof, perm = M.repeated_cv(key, train, ytr, feats, categories, params, permutation, log_target)
        model = M.build_model(key, feats, categories, params, log_target).fit(train[feats], ytr)
        p_tr, p_te = model.predict(train[feats]), model.predict(test[feats])
        results[key] = dict(features=feats, params=params, search=search, folds=folds, oof=oof, perm=perm,
                            model=model, pred_train=p_tr, pred_test=p_te,
                            cv={**M.summarise_cv(folds), "CV R2 pooled": M.pooled_oof_r2(oof, ytr)},
                            train=M.metrics(ytr, p_tr), test=M.metrics(yte, p_te),
                            ratio=M.ratio_statistics(yte, p_te),
                            ratio_oof=M.ratio_statistics(ytr, M.oof_mean_prediction(oof, train.index)))
        if verbose:
            r = results[key]
            print(f"{C.MODEL_SETS[set_key]} | {C.MODEL_NAMES[key]:27s} CV RMSE {r['cv']['CV RMSE']:>8,.0f} "
                  f"CV R2 pooled {r['cv']['CV R2 pooled']:.3f} | test RMSE {r['test']['RMSE']:>8,.0f} R2 {r['test']['R2']:.3f} "
                  f"| {time.time() - t0:.0f}s")
    out = {"set": set_key, "results": results, "log_target": log_target,
           "train_ids": list(train["Property_ID"]), "test_ids": list(test["Property_ID"])}
    if save:
        save_set(out, data)
    return out


def summary_tables(out):
    s = out["set"]
    perf, tuning, ratio = [], [], []
    for key, r in out["results"].items():
        base = {"Model set": C.MODEL_SETS[s], "Algorithm": C.MODEL_NAMES[key], "Predictors": len(r["features"])}
        perf.append({**base, **{k: v for k, v in r["cv"].items()},
                     "Train RMSE": r["train"]["RMSE"], "Train R2": r["train"]["R2"],
                     "Test MAE": r["test"]["MAE"], "Test RMSE": r["test"]["RMSE"], "Test R2": r["test"]["R2"]})
        tuning.append({**base, "Predictor list": ", ".join(r["features"]),
                       "Tuned hyperparameters": json.dumps(r["params"]) if r["params"] else "none (OLS)"})
        ratio.append({**base, "Sample": "Hold-out test", **r["ratio"], "IAAO check": M.iaao_verdict(r["ratio"])})
        ratio.append({**base, "Sample": "Training (out-of-fold CV)", **r["ratio_oof"],
                      "IAAO check": M.iaao_verdict(r["ratio_oof"])})
    return pd.DataFrame(perf), pd.DataFrame(tuning), pd.DataFrame(ratio)


def save_set(out, data):
    s = out["set"]
    perf, tuning, ratio = summary_tables(out)
    perf.to_csv(C.TABLES_DIR / f"{s}_performance.csv", index=False)
    tuning.to_csv(C.TABLES_DIR / f"{s}_predictors_and_hyperparameters.csv", index=False)
    ratio.to_csv(C.TABLES_DIR / f"{s}_ratio_study.csv", index=False)
    pd.concat([r["folds"].assign(Algorithm=C.MODEL_NAMES[k]) for k, r in out["results"].items()]) \
        .to_csv(C.TABLES_DIR / f"{s}_cv_folds.csv", index=False)
    test = data["test"]
    pred = test[["Property_ID", "Assessment_No", C.TARGET]].copy()
    for k, r in out["results"].items():
        pred[f"Pred_{C.MODEL_SHORT[k]}"] = r["pred_test"]
        pred[f"Ratio_{C.MODEL_SHORT[k]}"] = r["pred_test"] / test[C.TARGET].values
    pred.to_csv(C.PREDICTIONS_DIR / f"{s}_test_predictions.csv", index=False)
    for k, r in out["results"].items():
        joblib.dump(r["model"], C.MODELS_DIR / f"{s}_{k}.joblib")
    slim = {**out, "results": {k: {kk: vv for kk, vv in r.items() if kk not in ("search",)}
                               for k, r in out["results"].items()}}
    joblib.dump(slim, C.MODELS_DIR / f"{s}_results.joblib")


def load_set(set_key):
    return joblib.load(C.MODELS_DIR / f"{set_key}_results.joblib")


def permutation_summary(perm_df):
    out = pd.DataFrame({"Feature": perm_df.columns, "Mean": perm_df.mean().values,
                        "SD": perm_df.std(ddof=1).values, "Share of folds > 0": (perm_df > 0).mean().values})
    pos = out["Mean"].clip(lower=0)
    out["Relative importance (%)"] = 100 * pos / pos.sum() if pos.sum() > 0 else 0.0
    out = out.sort_values("Mean", ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, len(out) + 1))
    out.insert(2, "Label", out["Feature"].map(C.label))
    return out
