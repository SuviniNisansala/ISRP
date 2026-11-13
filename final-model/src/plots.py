"""Thesis figures (matplotlib, plain academic styling, 200 dpi PNG)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from . import config as C  # noqa: E402

INK, MUTED, GRID = "#1F2933", "#6B7280", "#E5E7EB"
PRIMARY, ACCENT = "#2F5D8A", "#B5523B"
SET_COLORS = {"A": "#9AA5B1", "B": "#2F5D8A"}
ALG_COLORS = {"mlr": "#5B6C8F", "rf": "#3A7D5C", "xgb": "#C2772B"}
rs_fmt = FuncFormatter(lambda x, _: f"{x:,.0f}")


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
        "legend.frameon": False, "font.family": "DejaVu Sans",
    })


def save(fig, name):
    path = C.FIGURES_DIR / name
    fig.savefig(path, facecolor="white")
    return path


# ----------------------------------------------------------------------------
# Dataset description (4.2)
# ----------------------------------------------------------------------------
def annual_value_distribution(d):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(d[C.TARGET], bins=40, color=PRIMARY, edgecolor="white")
    ax[0].set_xlabel("CMC Annual Value (Rs.)")
    ax[0].set_ylabel("Number of properties")
    ax[0].xaxis.set_major_formatter(rs_fmt)
    ax[0].set_title("(a) Annual Value")
    ax[1].hist(np.log(d[C.TARGET]), bins=30, color=PRIMARY, edgecolor="white")
    ax[1].set_xlabel("log(Annual Value)")
    ax[1].set_title("(b) log(Annual Value) - modelling scale")
    fig.tight_layout()
    return fig


def property_relationships(d):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    cols = {c: plt.cm.Blues(0.35 + 0.6 * i / 4) for i, c in enumerate(C.CONDITION_ORDER)}
    for c in C.CONDITION_ORDER:
        s = d[d["Condition"] == c]
        ax[0].scatter(s["Total_Floor_Area"], s[C.TARGET], s=14, alpha=0.75, color=cols[c], label=c)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Total floor area (sq ft, log scale)")
    ax[0].set_ylabel("Annual Value (Rs., log scale)")
    ax[0].legend(title="Condition", fontsize=8)
    ax[0].set_title("(a) Floor area vs Annual Value")
    ax[1].scatter(d["Age"], d[C.TARGET], s=14, alpha=0.6, color=PRIMARY)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Building age (years)")
    ax[1].set_title("(b) Age vs Annual Value")
    order = [c for c in C.CONDITION_ORDER if (d["Condition"] == c).any()]
    data = [d.loc[d["Condition"] == c, "AV_per_sqft"] for c in order]
    ax[2].boxplot(data, labels=[f"{c}\n(n={len(x)})" for c, x in zip(order, data)], showfliers=False,
                  medianprops={"color": ACCENT})
    ax[2].set_ylabel("Annual Value per sq ft (Rs.)")
    ax[2].set_title("(c) Annual Value per sq ft by condition")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# GIS variables (4.3)
# ----------------------------------------------------------------------------
def gis_distributions(d):
    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    for ax, g in zip(axes.flat, C.GIS_FEATURES):
        ax.hist(d[g], bins=25, color=PRIMARY, edgecolor="white")
        ax.set_title(C.SHORT_LABELS[g], fontsize=10)
        ax.set_xlabel("metres", fontsize=9)
    axes.flat[-1].axis("off")
    fig.tight_layout()
    return fig


def gis_target_correlation(tc, ax=None, title="GIS distances vs CMC Annual Value", xlabel=None):
    t = tc.sort_values("Spearman rho")
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    colors = [ACCENT if v < 0 else PRIMARY for v in t["Spearman rho"]]
    ax.barh([C.SHORT_LABELS[v] for v in t["Variable"]], t["Spearman rho"], color=colors)
    ax.axvline(0, color=MUTED, lw=0.8)
    for y, (v, p) in enumerate(zip(t["Spearman rho"], t["Spearman p-value"])):
        ax.text(v + (0.01 if v >= 0 else -0.01), y, f"{v:.2f}{'*' if p < 0.05 else ''}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.set_xlabel(xlabel or "Spearman correlation (* p < 0.05)")
    ax.set_title(title)
    lim = max(0.1, t["Spearman rho"].abs().max() * 1.35)
    ax.set_xlim(-lim, lim)
    return fig or ax.figure


def correlation_heatmap(corr, title):
    labels = [C.SHORT_LABELS.get(c, c) for c in corr.columns]
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.6 else INK)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson r")
    ax.set_title(title)
    return fig


# ----------------------------------------------------------------------------
# Model results
# ----------------------------------------------------------------------------
def actual_vs_predicted(y_test, preds: dict, set_label):
    fig, axes = plt.subplots(1, len(preds), figsize=(5 * len(preds), 4.6), sharex=True, sharey=True)
    lo, hi = y_test.min() * 0.8, y_test.max() * 1.25
    for ax, (key, p) in zip(np.atleast_1d(axes), preds.items()):
        ax.scatter(y_test, p, s=16, alpha=0.7, color=ALG_COLORS[key])
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(C.MODEL_NAMES[key])
        ax.set_xlabel("CMC Annual Value (Rs.)")
        ax.xaxis.set_major_formatter(rs_fmt)
        ax.yaxis.set_major_formatter(rs_fmt)
    np.atleast_1d(axes)[0].set_ylabel("Predicted Annual Value (Rs.)")
    fig.suptitle(f"{set_label}: hold-out test set (log scales)", y=1.02)
    fig.tight_layout()
    return fig


def model_comparison_bars(comp):
    """comp: rows per (Set, key) with Test RMSE / Test MAE / Test R2."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    x = np.arange(len(C.ALGORITHMS))
    w = 0.38
    for ax, m, lab in zip(axes, ["RMSE", "MAE", "R2"], ["RMSE (Rs.)", "MAE (Rs.)", "R²"]):
        for i, s in enumerate(["A", "B"]):
            vals = [comp.loc[(comp["Set"] == s) & (comp["key"] == k), f"Test {m}"].iloc[0] for k in C.ALGORITHMS]
            bars = ax.bar(x + (i - 0.5) * w, vals, w, color=SET_COLORS[s], label=C.MODEL_SETS[s])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}" if m == "R2" else f"{v:,.0f}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([C.MODEL_SHORT[k] for k in C.ALGORITHMS])
        ax.set_ylabel(lab)
        ax.set_title(f"Test {lab.split(' ')[0]}" + (" (lower is better)" if m != "R2" else " (higher is better)"))
        if m != "R2":
            ax.yaxis.set_major_formatter(rs_fmt)
    axes[0].legend(fontsize=8.5)
    fig.tight_layout()
    return fig


def paired_cv_differences(fold_diff):
    """fold_diff: columns Model, split, dRMSE (B - A)."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    data = [fold_diff.loc[fold_diff["key"] == k, "RMSE change (B - A)"] for k in C.ALGORITHMS]
    ax.boxplot(data, labels=[C.MODEL_NAMES[k] for k in C.ALGORITHMS], medianprops={"color": ACCENT})
    ax.axhline(0, color=MUTED, lw=1, ls="--")
    ax.set_ylabel("RMSE Model B - RMSE Model A (Rs.)")
    ax.yaxis.set_major_formatter(rs_fmt)
    ax.set_title("Paired CV folds: effect of adding GIS variables (below 0 = GIS improves)")
    return fig


def ratio_by_band(band_tables: dict):
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for (lab, color, ls), t in band_tables.items():
        ax.plot(t["Band"].astype(str), t["Median_ratio"], marker="o", color=color, ls=ls, label=lab)
    ax.axhspan(*C.IAAO_MEDIAN_RATIO, color=GRID, alpha=0.6, label="IAAO 0.90-1.10")
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_xlabel("CMC Annual Value quartile (Q1 = lowest)")
    ax.set_ylabel("Median ratio (predicted / CMC)")
    ax.set_title("Assessment ratio by value band (hold-out test)")
    ax.legend(fontsize=8)
    return fig


def importance_bars(perm_sum, title):
    t = perm_sum.sort_values("Mean")
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(t) + 1.5))
    ax.barh([C.SHORT_LABELS.get(f, f) for f in t["Feature"]], t["Mean"], xerr=t["SD"], color=PRIMARY,
            error_kw={"ecolor": MUTED, "lw": 0.8})
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Increase in CV RMSE when shuffled (Rs.)")
    ax.xaxis.set_major_formatter(rs_fmt)
    ax.set_title(title)
    return fig


def waterfall(steps: pd.DataFrame, title):
    """steps: columns Label, Start (Rs.), End (Rs.) - from interpret.waterfall_steps."""
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(steps) + 1.8))
    y = np.arange(len(steps))[::-1]
    for yi, (_, r) in zip(y, steps.iterrows()):
        if r["Kind"] in ("base", "final"):
            ax.barh(yi, r["End (Rs.)"], color=MUTED if r["Kind"] == "base" else INK, height=0.6)
            ax.text(r["End (Rs.)"], yi, f"  Rs. {r['End (Rs.)']:,.0f}", va="center", fontsize=8.5)
        else:
            delta = r["End (Rs.)"] - r["Start (Rs.)"]
            ax.barh(yi, delta, left=r["Start (Rs.)"], color=PRIMARY if delta >= 0 else ACCENT, height=0.6)
            ax.text(max(r["Start (Rs.)"], r["End (Rs.)"]), yi, f"  {delta:+,.0f}", va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(steps["Label"], fontsize=9)
    ax.set_xlabel("Annual Value (Rs.)")
    ax.xaxis.set_major_formatter(rs_fmt)
    ax.set_title(title)
    return fig
