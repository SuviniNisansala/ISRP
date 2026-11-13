"""Interpretation (section 3.10 / 4.6): SHAP for tree models, partial dependence, and an
itemised explanation of one property's estimate ("worked example").

All models are fitted on log(Annual Value), so feature contributions are additive on the log
scale. They are shown to a reader in two equivalent ways:
  * as a multiplicative effect (%), e.g. "+12 % for Good vs Fair condition";
  * as rupee steps in a waterfall that starts at the average estimate and ends exactly at the
    property's estimate (contributions are applied in order of size).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import partial_dependence  # noqa: E402

from . import config as C  # noqa: E402
from . import modelling as M  # noqa: E402
from .plots import MUTED, PRIMARY, rs_fmt  # noqa: E402


def _transformed(model, X):
    pipe = M.fitted_pipeline(model)
    names = M.feature_names_out(model)
    return pd.DataFrame(pipe.named_steps["preprocess"].transform(X), columns=names, index=X.index), names


def shap_values(model, X_background, X_explain):
    """TreeExplainer for RF / XGBoost; exact linear SHAP for MLR. Values are on the log scale."""
    import shap
    pipe = M.fitted_pipeline(model)
    est = pipe.named_steps["model"]
    Xb, names = _transformed(model, X_background)
    Xe, _ = _transformed(model, X_explain)
    if hasattr(est, "coef_"):
        mean = Xb.mean()
        sv = (Xe - mean).values * est.coef_
        base = float(est.intercept_ + (mean.values * est.coef_).sum())
    else:
        explainer = shap.TreeExplainer(est)
        sv = explainer.shap_values(Xe)
        base = float(np.ravel(explainer.expected_value)[0])
    return pd.DataFrame(sv, columns=names, index=X_explain.index), base, Xe


def shap_importance(sv: pd.DataFrame):
    m = sv.abs().mean()
    t = pd.DataFrame({"Feature": m.index, "Label": [C.label(f) for f in m.index],
                      "Mean |SHAP| (log scale)": m.values,
                      "Typical effect on Annual Value (%)": 100 * (np.exp(m.values) - 1)})
    return t.sort_values("Mean |SHAP| (log scale)", ascending=False).reset_index(drop=True)


def shap_summary_plot(sv, Xe, title):
    import shap
    plt.figure()
    shap.summary_plot(sv.values, Xe.values, feature_names=[C.SHORT_LABELS.get(c, C.label(c)) for c in sv.columns],
                      show=False, plot_size=(9, 0.4 * sv.shape[1] + 2))
    plt.title(title, fontsize=11.5)
    plt.xlabel("SHAP value (effect on log Annual Value; 0.10 ~ +10.5 %)")
    return plt.gcf()


def pdp_figure(model, X, features, title):
    n = len(features)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows), squeeze=False)
    for ax, f in zip(axes.flat, features):
        pd_res = partial_dependence(model, X, [f], kind="average", grid_resolution=40, percentiles=(0.02, 0.98))
        ax.plot(pd_res["grid_values"][0], pd_res["average"][0], color=PRIMARY, lw=2)
        ymin = ax.get_ylim()[0]
        ax.plot(X[f], np.full(len(X), ymin), "|", color=MUTED, ms=8, alpha=0.5)
        ax.set_xlabel(C.label(f))
        ax.set_ylabel("Average predicted AV (Rs.)")
        ax.yaxis.set_major_formatter(rs_fmt)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


def reference_property(train, features):
    """A typical property: median numeric values, the reference condition grade (Fair)."""
    ref = {f: float(train[f].median()) for f in features if f not in C.CATEGORICAL_FEATURES}
    if "Condition" in features:
        ref["Condition"] = C.CONDITION_REFERENCE
    return pd.DataFrame([ref])[features]


def mlr_itemised(model, train, row, features):
    """Rule-like MLR explanation: this property vs the reference property. Each term is
    coefficient x (value - reference value) on the log scale; the condition dummies collapse into
    one 'Condition grade' term. Sums exactly to the model estimate."""
    ref = reference_property(train, features)
    est = M.fitted_pipeline(model).named_steps["model"]
    Xr, names = _transformed(model, ref)
    Xp, _ = _transformed(model, row[features])
    contrib = pd.Series((Xp.values[0] - Xr.values[0]) * est.coef_, index=names)
    cond = [n for n in names if n.startswith("Condition_")]
    if cond:
        contrib["Condition"] = contrib[cond].sum()
        contrib = contrib.drop(cond)
    base_log = float(np.log(model.predict(ref)[0]))
    return contrib, base_log, ref


def _step_label(f, raw_values):
    if f == "Log_Total_Floor_Area":
        f = "Total_Floor_Area"
    val = raw_values.get(f, "")
    if isinstance(val, (float, np.floating)):
        val = f"{val:,.0f}"
    return f"{C.label(f)} = {val}"


def waterfall_steps(contrib: pd.Series, base_log: float, raw_values: dict, top=None,
                    base_label="Average estimate (all training properties)"):
    """contrib: log-scale contributions of one property (index = transformed feature names)."""
    order = contrib.abs().sort_values(ascending=False).index
    if top and len(order) > top:
        keep, rest = list(order[:top]), list(order[top:])
    else:
        keep, rest = list(order), []
    rows = [{"Label": base_label, "Kind": "base",
             "Start (Rs.)": 0.0, "End (Rs.)": float(np.exp(base_log)), "Effect (%)": np.nan}]
    cur = base_log
    for f in keep:
        nxt = cur + contrib[f]
        rows.append({"Label": _step_label(f, raw_values), "Kind": "step", "Start (Rs.)": float(np.exp(cur)),
                     "End (Rs.)": float(np.exp(nxt)), "Effect (%)": 100 * (np.exp(contrib[f]) - 1)})
        cur = nxt
    if rest:
        tot = contrib[rest].sum()
        rows.append({"Label": f"{len(rest)} other variables", "Kind": "step", "Start (Rs.)": float(np.exp(cur)),
                     "End (Rs.)": float(np.exp(cur + tot)), "Effect (%)": 100 * (np.exp(tot) - 1)})
        cur += tot
    rows.append({"Label": "Model estimate for this property", "Kind": "final", "Start (Rs.)": 0.0,
                 "End (Rs.)": float(np.exp(cur)), "Effect (%)": np.nan})
    t = pd.DataFrame(rows)
    t["Change (Rs.)"] = np.where(t["Kind"] == "step", t["End (Rs.)"] - t["Start (Rs.)"], np.nan)
    return t
