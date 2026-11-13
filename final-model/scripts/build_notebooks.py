"""Generates the three analysis notebooks (run from the final-model folder):

    python scripts/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/01_Model_A_property_only.ipynb  (then 02, 03)
"""
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
KERNEL = {"name": "cmc-final", "display_name": "Python (CMC final model)", "language": "python"}

SETUP = """import sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
from scipy import stats

from src import config as C, data as D, modelling as M, runner as R, plots as P, interpret as I
P.setup_style()
pd.set_option("display.max_columns", 40, "display.width", 200, "display.float_format", lambda v: f"{v:,.3f}")
T, F = C.TABLES_DIR, C.FIGURES_DIR
def show(fig, name):
    P.save(fig, name); plt.show()"""


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": KERNEL, "language_info": {"name": "python"}}
    NB_DIR.mkdir(exist_ok=True)
    nbf.write(nb, NB_DIR / name)
    print("wrote", name)


# =============================================================================================
# 01  Model A - property only
# =============================================================================================
nb1 = [
    md("""
# Notebook 01 - Model A: property-only predictors
**A Machine Learning-Based Approach to Residential Property Tax Assessment in a Data-Constrained Environment:
Dematagoda Ward, Colombo Municipal Council**

Model A uses only the property characteristics recorded by CMC (building age, condition grade, total floor
area, number of floors) and **no location information**. It sets the baseline performance (methodology 3.8).
Model B (notebook 02) adds the OSM/GIS distances; notebook 03 compares the two.

* Target: **CMC-recorded Annual Value (2025)**. It is not market value.
* Algorithms: Multiple Linear Regression (baseline), Random Forest, XGBoost.
* Validation: a fixed 80:20 split (seed 42, stratified by Annual Value). Tuning and 5-fold CV (repeated 10 times) use
  only the training set. The test set is used once, at the end.
* All models are fitted on **log(Annual Value)**; predictions are converted back to rupees, and every metric is in rupees.

This notebook also contains the **dataset description** (Chapter 4.2)."""),
    code(SETUP),
    md("## 1. Load and clean the data (methodology 3.6-3.7)"),
    code("""data = D.load_and_prepare()
lab, train, test = data["labelled"], data["train"], data["test"]
(T / "AB_cleaning_log.txt").write_text("\\n".join(data["log"]) + "\\n")
data["accounting"].to_csv(T / "AB_record_accounting.csv", index=False)
data["accounting"]"""),
    md("### Missing values in the raw file"),
    code("""miss = D.raw_missing_table(data["raw"])
miss = miss[miss["Variable"].isin(["Age", "Condition", "Ground_Floor_Area", "First_Floor_Area", "Second_Floor_Area",
                                   "Third_Floor_Area", "Annual_Value"] + C.GIS_FEATURES)]
# missingness among the labelled (modelling) records
lab_miss = lab[["Age", "Condition", "Total_Floor_Area"]].isna().sum().rename("Missing in modelling data")
miss.to_csv(T / "AB_missing_values.csv", index=False)
display(miss)
lab_miss.to_frame()"""),
    md("""**Interpretation.** Upper-floor fields are blank when a building has no such floor, so a blank is read as 0
(the rule from the interim prototype, to be confirmed with CMC). No predictor in the modelling data is above the 30 %
exclusion threshold. The few missing ages and condition grades of labelled properties are imputed inside each model
pipeline (median / mode), using training data only."""),
    md("### Condition grades and repeated assessment numbers"),
    code("""data["condition_table"].to_csv(T / "AB_condition_cleaning.csv", index=False)
display(data["condition_table"])
dr = data["duplicate_review"]
dr.to_csv(T / "AB_duplicate_assessment_numbers.csv", index=False)
dr["Classification"].value_counts().to_frame("Records")"""),
    md("""Assessment numbers repeat because the same house number exists on different roads. The records share a number
but have different locations, attributes and Unique Property IDs, so all are retained. There are no exact duplicate
records."""),
    md("## 2. Dataset description (Chapter 4.2)"),
    code("""desc_vars = [C.TARGET, "Total_Floor_Area", "Ground_Floor_Area", "Num_Floors", "Age", "AV_per_sqft"]
rows = []
for v in desc_vars:
    s = lab[v].dropna()
    rows.append({"Variable": C.label(v), "n": len(s), "Mean": s.mean(), "Median": s.median(), "SD": s.std(),
                 "Min": s.min(), "Max": s.max(), "Skewness": stats.skew(s)})
desc = pd.DataFrame(rows)
desc.to_csv(T / "A_descriptive_statistics.csv", index=False)
desc"""),
    code("""cond = (lab.groupby("Condition")
          .agg(n=(C.TARGET, "size"), Median_AV=(C.TARGET, "median"), Median_floor_area=("Total_Floor_Area", "median"),
               Median_AV_per_sqft=("AV_per_sqft", "median"))
          .reindex([c for c in C.CONDITION_ORDER if c in set(lab["Condition"])]).reset_index())
cond.to_csv(T / "A_condition_summary.csv", index=False)
floors = lab.groupby("Num_Floors").agg(n=(C.TARGET, "size"), Median_AV=(C.TARGET, "median")).reset_index()
floors.to_csv(T / "A_floors_summary.csv", index=False)
display(cond); floors"""),
    code("""rows = []
for v in ["Total_Floor_Area", "Num_Floors", "Age"]:
    s = lab[[v, C.TARGET]].dropna()
    rho, p = stats.spearmanr(s[v], s[C.TARGET])
    r, pr = stats.pearsonr(np.log(s[v]) if v == "Total_Floor_Area" else s[v], np.log(s[C.TARGET]))
    rows.append({"Variable": C.label(v), "Spearman rho with AV": rho, "p-value": p,
                 "Pearson r (log AV)" + (" vs log area" if v == "Total_Floor_Area" else ""): r})
prop_corr = pd.DataFrame(rows)
prop_corr.to_csv(T / "A_property_correlations.csv", index=False)
prop_corr"""),
    code("""show(P.annual_value_distribution(lab), "4_1_annual_value_distribution.png")
show(P.property_relationships(lab), "4_2_property_relationships.png")"""),
    md("""**Interpretation.** Annual Value is strongly right-skewed: a few large multi-storey houses have values far
above the median. The log scale (panel b) is close to symmetric, which is why the models use log(Annual Value). Floor
area is by far the strongest single correlate. Annual Value per sq ft is quite stable, which is consistent with CMC
applying a rate per sq ft. Its level rises with the condition grade, so condition explains part of the variation in
rate."""),
    code("""flag = lab[lab["Flag_Extreme_AV_per_sqft"] == 1][["Property_ID", "Assessment_No", "Total_Floor_Area", C.TARGET,
                                               "AV_per_sqft", "Partition"]]
flag.to_csv(T / "AB_flagged_extreme_rate_records.csv", index=False)
flag"""),
    md("""These records have an Annual Value per sq ft far outside the usual range (beyond 3 IQR on the log scale).
Possible causes are entry errors, partly non-residential use, or a floor area that does not match the assessed unit.
They are **kept** in the main analysis, because the data must not be cleaned to make the model look better. A
sensitivity check without them is reported in notebook 03."""),
    md("## 3. Model A predictors and validation design"),
    code("""sets, screened, screen_table = M.feature_sets(train)
categories = M.condition_categories(lab["Condition"])
pd.DataFrame([{"Algorithm": C.MODEL_NAMES[k], "Model A predictors": ", ".join(sets[("A", k)])} for k in C.ALGORITHMS])"""),
    md("""* **MLR** uses log(floor area). With log(Annual Value) as the target, this is the standard log-log hedonic form
  (Annual Value ~ rate x area), and its coefficient is an elasticity. Condition is dummy-coded with **Fair** as the
  reference grade.
* **RF / XGBoost** use floor area in sq ft (tree models are not affected by monotone transforms). Condition is an
  ordinal grade: Poor < Fair- < Fair < Fair+ < Good.
* Hyperparameters are tuned with `RandomizedSearchCV` (5-fold CV on the training set; RF: 40 candidates, XGBoost:
  60 candidates, scored by RMSE in rupees)."""),
    code("""print(f"Training records: {len(train)}, test records: {len(test)}")
pd.concat([train[C.TARGET].describe().rename("train"), test[C.TARGET].describe().rename("test")], axis=1)"""),
    md("## 4. Train, tune and validate Model A (MLR, RF, XGBoost)"),
    code("""outA = R.run_set("A", data, sets, categories, permutation=True)
perfA, tuneA, ratioA = R.summary_tables(outA)"""),
    md("### Cross-validation (training set) and hold-out test performance"),
    code("""cols = ["Algorithm", "CV RMSE", "CV RMSE SD", "CV MAE", "CV R2 pooled", "Train R2", "Test RMSE", "Test MAE", "Test R2"]
perfA[cols]"""),
    code("""tuneA"""),
    code("""show(P.actual_vs_predicted(test[C.TARGET], {k: r["pred_test"] for k, r in outA["results"].items()},
                           C.MODEL_SETS["A"]), "A_actual_vs_predicted.png")"""),
    md("## 5. MLR coefficients (statistical interpretation of Model A)"),
    code("""olsA, olsA_table, olsA_stats = M.ols_fit(train, sets[("A", "mlr")], categories)
olsA_table.to_csv(T / "A_mlr_ols.csv", index=False)
pd.DataFrame([olsA_stats]).to_csv(T / "A_mlr_ols_fit.csv", index=False)
display(pd.DataFrame([olsA_stats]))
olsA_table[["Term", "Coefficient (log scale)", "Std. error", "p-value", "Readable effect"]]"""),
    md("## 6. Assessment-ratio analysis for Model A (ratio = model estimate / CMC Annual Value)"),
    code("""ratioA[["Algorithm", "Sample", "n", "Median ratio", "COD (%)", "PRD", "PRB", "Within +/-10% (%)", "IAAO check"]]"""),
    md("""IAAO reference ranges (Standard on Ratio Studies, residential): median ratio 0.90-1.10, COD <= 15, PRD 0.98-1.03,
PRB between -0.05 and +0.05. A PRD above 1.03 or a negative PRB means **regressivity**: lower-value properties are
estimated relatively higher than higher-value ones. These measure agreement with the CMC assessments, not
market-value accuracy. All three models are interpreted together with Model B in notebook 03."""),
    code("""assert len(train) + len(test) == len(lab)
assert set(outA["test_ids"]) == set(pd.read_csv(C.SPLIT_FILE).query("Partition == 'test'")["Property_ID"])
print("Model A complete - outputs saved with prefix A_ in outputs/")"""),
]

# =============================================================================================
# 02  Model B - property + GIS
# =============================================================================================
nb2 = [
    md("""
# Notebook 02 - Model B: property + OSM/GIS location predictors
Model B adds the GIS-derived distances (QGIS, OpenStreetMap) to the Model A property variables. It uses the same
target, the same cleaned records, **the same train/test split** (loaded from `split_ids.csv`), the same CV folds and
the same metrics. The difference from Model A shows how much of the location effect the OSM variables capture.

This notebook also contains the **GIS variable results** (Chapter 4.3)."""),
    code(SETUP),
    code("""data = D.load_and_prepare(verbose=False)
lab, train, test = data["labelled"], data["train"], data["test"]
print(len(lab), "records;", len(train), "train /", len(test), "test (same split as Model A)")"""),
    md("## 1. GIS-derived distances (Chapter 4.3)"),
    code("""rows = []
for g in C.GIS_FEATURES:
    s = data["all"][g]
    rows.append({"Variable": C.label(g), "Group": C.GIS_GROUPS[g], "In Table y (3.5)": g in C.GIS_IN_METHODOLOGY,
                 "Records": s.notna().sum(), "Missing": s.isna().sum(), "Min": s.min(), "Median": s.median(),
                 "Max": s.max(), "Range": s.max() - s.min()})
gis_desc = pd.DataFrame(rows)
gis_desc.to_csv(T / "B_gis_descriptive.csv", index=False)
gis_desc"""),
    code("""show(P.gis_distributions(lab), "4_3_gis_distributions.png")"""),
    md("""All 11 distances are available for every property, so no OSM gaps had to be treated as missing.
The distances span a few hundred metres to about 2 km within the ward."""),
    md("### Association with Annual Value and with Annual Value per sq ft"),
    code("""def gis_corr(target):
    out = []
    for g in C.GIS_FEATURES:
        rho, p = stats.spearmanr(lab[g], lab[target])
        out.append({"Variable": g, "Label": C.label(g), "Spearman rho": rho, "Spearman p-value": p})
    return pd.DataFrame(out)
tc_av, tc_rate = gis_corr(C.TARGET), gis_corr("AV_per_sqft")
gis_corr_table = tc_av.merge(tc_rate, on=["Variable", "Label"], suffixes=(" (AV)", " (AV per sq ft)"))
gis_corr_table.to_csv(T / "B_gis_correlations.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
P.gis_target_correlation(tc_av, ax[0], "(a) with Annual Value")
P.gis_target_correlation(tc_rate, ax[1], "(b) with Annual Value per sq ft")
ax[1].set_yticklabels([])
ax[1].set_yticks(range(len(tc_rate)))
ax[1].set_yticklabels([C.SHORT_LABELS[v] for v in tc_rate.sort_values("Spearman rho")["Variable"]])
fig.tight_layout()
show(fig, "4_4_gis_correlations.png")
gis_corr_table"""),
    md("""**Why AV per sq ft?** Annual Value mostly scales with floor area. Location should therefore appear in the
**rate per sq ft** that CMC applies through the road-wise chart. Panel (b) shows whether the distances relate to that
rate."""),
    md("### Multicollinearity among the distances"),
    code("""corr = lab[C.GIS_FEATURES].corr()
corr.to_csv(T / "B_gis_corr_matrix.csv")
show(P.correlation_heatmap(corr, "Pearson correlation between GIS distances"), "4_5_gis_correlation_matrix.png")
vif_all = M.vif_table(train, ["Age", "Log_Total_Floor_Area", "Num_Floors"] + C.GIS_FEATURES)
vif_all.to_csv(T / "B_vif_all_gis.csv", index=False)
vif_all"""),
    md("""Within one ward, the distances to the CBD, railway station, schools, supermarkets and so on largely encode the
**same position**, so many pairs are highly correlated and the VIFs are very large. A linear regression with all 11
would give unstable, uninterpretable coefficients. MLR therefore uses a **screened subset** (rule below, applied to
the training rows only). RF and XGBoost are not affected in the same way and use all 11 distances."""),
    code("""sets, screened, screen_table = M.feature_sets(train)
screen_table.to_csv(T / "B_mlr_gis_screening.csv", index=False)
vif_screened = M.vif_table(train, ["Age", "Log_Total_Floor_Area", "Num_Floors"] + screened)
vif_screened.to_csv(T / "B_vif_screened.csv", index=False)
display(screen_table)
print("GIS variables kept for MLR:", screened)
vif_screened"""),
    md("## 2. Model B predictors"),
    code("""categories = M.condition_categories(lab["Condition"])
pd.DataFrame([{"Algorithm": C.MODEL_NAMES[k], "n predictors": len(sets[("B", k)]),
               "Model B predictors": ", ".join(sets[("B", k)])} for k in C.ALGORITHMS])"""),
    md("## 3. Train, tune and validate Model B"),
    code("""outB = R.run_set("B", data, sets, categories, permutation=True)
perfB, tuneB, ratioB = R.summary_tables(outB)
cols = ["Algorithm", "CV RMSE", "CV RMSE SD", "CV MAE", "CV R2 pooled", "Train R2", "Test RMSE", "Test MAE", "Test R2"]
perfB[cols]"""),
    code("""tuneB"""),
    code("""show(P.actual_vs_predicted(test[C.TARGET], {k: r["pred_test"] for k, r in outB["results"].items()},
                           C.MODEL_SETS["B"]), "B_actual_vs_predicted.png")"""),
    md("## 4. MLR coefficients (Model B, screened GIS set)"),
    code("""olsB, olsB_table, olsB_stats = M.ols_fit(train, sets[("B", "mlr")], categories)
olsB_table.to_csv(T / "B_mlr_ols.csv", index=False)
pd.DataFrame([olsB_stats]).to_csv(T / "B_mlr_ols_fit.csv", index=False)
display(pd.DataFrame([olsB_stats]))
olsB_table[["Term", "Coefficient (log scale)", "Std. error", "p-value", "Readable effect"]]"""),
    md("## 5. Assessment-ratio analysis for Model B"),
    code("""ratioB[["Algorithm", "Sample", "n", "Median ratio", "COD (%)", "PRD", "PRB", "Within +/-10% (%)", "IAAO check"]]"""),
    code("""outA_test = set(pd.read_csv(C.SPLIT_FILE).query("Partition == 'test'")["Property_ID"])
assert set(outB["test_ids"]) == outA_test, "Model B must be tested on the same records as Model A"
print("Model B complete - outputs saved with prefix B_ in outputs/")"""),
]

# =============================================================================================
# 03  Comparison + interpretation
# =============================================================================================
nb3 = [
    md("""
# Notebook 03 - Model A vs Model B: comparison, bias analysis and interpretation
This notebook loads the saved results of notebooks 01 and 02 (no retraining of the main models). It then:
1. compares the six configurations (Chapter 4.4);
2. measures the gain from the OSM/GIS variables on the same test records and on paired CV folds;
3. runs the assessment-ratio (consistency and bias) analysis (Chapter 4.5);
4. interprets the models: coefficients, permutation importance, SHAP, partial dependence and a worked example for one
   property (Chapter 4.6);
5. runs supplementary checks (spatially grouped CV, a raw-target model, and results without the flagged records);
6. builds the Chapter 4 draft (`report/Chapter4_draft.docx`)."""),
    code(SETUP + """
from sklearn.inspection import permutation_importance"""),
    code("""data = D.load_and_prepare(verbose=False)
lab, train, test = data["labelled"], data["train"], data["test"]
outA, outB = R.load_set("A"), R.load_set("B")
assert outA["test_ids"] == outB["test_ids"] == list(test["Property_ID"]), "A and B must share the test records"
sets, screened, _ = M.feature_sets(train)
categories = M.condition_categories(lab["Condition"])
yte, ytr = test[C.TARGET].values, train[C.TARGET]
OUT = {"A": outA, "B": outB}"""),
    md("## 1. Six-configuration comparison (Chapter 4.4)"),
    code("""rows = []
for s, out in OUT.items():
    for k, r in out["results"].items():
        rows.append({"Set": s, "key": k, "Model set": C.MODEL_SETS[s], "Algorithm": C.MODEL_NAMES[k],
                     "Predictors": len(r["features"]), "CV RMSE": r["cv"]["CV RMSE"], "CV RMSE SD": r["cv"]["CV RMSE SD"],
                     "CV MAE": r["cv"]["CV MAE"], "CV R2 pooled": r["cv"]["CV R2 pooled"],
                     "Test RMSE": r["test"]["RMSE"], "Test MAE": r["test"]["MAE"], "Test R2": r["test"]["R2"],
                     "Test MAPE (%)": 100 * np.mean(np.abs(r["pred_test"] - yte) / yte)})
comp = pd.DataFrame(rows)
best = comp.sort_values("CV RMSE").iloc[0]
comp["Selected (lowest CV RMSE)"] = np.where((comp["Set"] == best["Set"]) & (comp["key"] == best["key"]), "Yes", "")
comp.drop(columns=["Set", "key"]).to_csv(T / "AB_six_model_comparison.csv", index=False)
comp.drop(columns=["Set", "key"])"""),
    code("""show(P.model_comparison_bars(comp), "AB_model_comparison.png")
print(f"Best configuration (selected on cross-validation, not on the test set): {best['Model set']} - {best['Algorithm']}")"""),
    md("""The best configuration is chosen on **cross-validated RMSE in the training set**, not on the test set. The
test set therefore stays an unbiased check. With about 100 test properties, small differences between models (a few
hundred rupees of RMSE) are within sampling noise. The paired analysis below is the fairer test of the GIS effect."""),
    md("## 2. Does adding OSM/GIS variables improve the estimates? (A vs B)"),
    code("""gain, fold_rows = [], []
for k in C.ALGORITHMS:
    a, b = outA["results"][k], outB["results"][k]
    fa, fb = a["folds"].set_index("split"), b["folds"].set_index("split")
    d_rmse = fb["RMSE"] - fa["RMSE"]
    # absolute errors on the same test records -> Wilcoxon signed-rank test
    ea, eb = np.abs(a["pred_test"] - yte), np.abs(b["pred_test"] - yte)
    w_p = stats.wilcoxon(eb, ea).pvalue
    t_p = stats.ttest_rel(fb["RMSE"], fa["RMSE"]).pvalue
    gain.append({"Algorithm": C.MODEL_NAMES[k],
                 "Test RMSE A": a["test"]["RMSE"], "Test RMSE B": b["test"]["RMSE"],
                 "Test RMSE change (%)": 100 * (b["test"]["RMSE"] / a["test"]["RMSE"] - 1),
                 "Test MAE A": a["test"]["MAE"], "Test MAE B": b["test"]["MAE"],
                 "Test MAE change (%)": 100 * (b["test"]["MAE"] / a["test"]["MAE"] - 1),
                 "Test R2 A": a["test"]["R2"], "Test R2 B": b["test"]["R2"],
                 "Test: properties with smaller error in B (%)": 100 * np.mean(eb < ea),
                 "Wilcoxon p (test abs. errors)": w_p,
                 "CV RMSE change, mean (Rs.)": d_rmse.mean(), "CV RMSE change, SD": d_rmse.std(),
                 "CV RMSE change (%)": 100 * (fb["RMSE"].mean() / fa["RMSE"].mean() - 1),
                 "CV folds where B better (%)": 100 * np.mean(d_rmse < 0),
                 "Paired t-test p (50 CV folds)": t_p})
    fold_rows.append(pd.DataFrame({"key": k, "split": d_rmse.index, "RMSE change (B - A)": d_rmse.values}))
gain = pd.DataFrame(gain)
gain.to_csv(T / "AB_gis_gain.csv", index=False)
fold_diff = pd.concat(fold_rows)
show(P.paired_cv_differences(fold_diff), "AB_paired_cv_differences.png")
gain.T"""),
    md("""**How to read this.** A negative change means Model B (with GIS) has a lower error than Model A. Because the CV
folds are identical for A and B, the 50 paired fold differences show both the **size** and the **consistency** of
any GIS gain (methodology 3.9). The p-values (paired t-test on folds, Wilcoxon on the test-set absolute errors) are
indicative only, because the repeated CV folds overlap."""),
    md("## 3. Consistency and bias: assessment-ratio study (Chapter 4.5)"),
    code("""rr = []
for s, out in OUT.items():
    for k, r in out["results"].items():
        for sample, st in [("Hold-out test", r["ratio"]), ("Training (out-of-fold CV)", r["ratio_oof"])]:
            rr.append({"Model set": C.MODEL_SETS[s], "Algorithm": C.MODEL_NAMES[k], "Sample": sample, **st,
                       "IAAO check": M.iaao_verdict(st)})
ratio_all = pd.DataFrame(rr)
ratio_all.to_csv(T / "AB_ratio_study.csv", index=False)
ratio_all[["Model set", "Algorithm", "Sample", "n", "Median ratio", "COD (%)", "PRD", "PRB", "Within +/-20% (%)", "IAAO check"]]"""),
    code("""bands, band_rows = {}, []
styles = {("A", "mlr"): ("#9AA5B1", "--"), ("B", "mlr"): ("#5B6C8F", "-"), ("A", best["key"]): ("#E0B07A", "--"),
          ("B", best["key"]): ("#C2772B", "-")}
for (s, k), (col, ls) in styles.items():
    t = M.ratio_by_value_band(yte, OUT[s]["results"][k]["pred_test"])
    bands[(f"{C.MODEL_SETS[s].split(' - ')[0]} {C.MODEL_SHORT[k]}", col, ls)] = t
    band_rows.append(t.assign(**{"Model set": C.MODEL_SETS[s], "Algorithm": C.MODEL_NAMES[k]}))
pd.concat(band_rows).to_csv(T / "AB_ratio_by_value_band.csv", index=False)
show(P.ratio_by_band(bands), "AB_ratio_by_value_band.png")
pd.concat(band_rows)"""),
    md("### Sensitivity: ratio statistics without the flagged extreme-rate records"),
    code("""keep = (test["Flag_Extreme_AV_per_sqft"] == 0).values
sens = []
for s, out in OUT.items():
    for k, r in out["results"].items():
        st = M.ratio_statistics(yte[keep], r["pred_test"][keep])
        sens.append({"Model set": C.MODEL_SETS[s], "Algorithm": C.MODEL_NAMES[k], **M.metrics(yte[keep], r["pred_test"][keep]),
                     **{x: st[x] for x in ["n", "Median ratio", "COD (%)", "PRD", "PRB"]}, "IAAO check": M.iaao_verdict(st)})
sens = pd.DataFrame(sens)
sens.to_csv(T / "AB_ratio_without_flagged.csv", index=False)
print(f"Test records excluded: {(~keep).sum()}")
sens"""),
    md("""**Reading the ratio study.** The median ratio tests the overall *level* (do the models reproduce the CMC
values on average?). COD tests *uniformity*: how far individual ratios are from the median. PRD and PRB test
*vertical equity*. A PRD above 1.03 or a negative PRB means low-value properties receive relatively higher estimates
than high-value ones (regressive). The models are trained on CMC values, so they reproduce any systematic pattern in
those values. These figures show agreement and consistency with CMC assessments; they do not show market-value
accuracy."""),
    md("## 4. Interpretation (Chapter 4.6)"),
    md("### 4.1 MLR coefficients - Model A vs Model B side by side"),
    code("""olsA = pd.read_csv(T / "A_mlr_ols.csv"); olsB = pd.read_csv(T / "B_mlr_ols.csv")
fitA = pd.read_csv(T / "A_mlr_ols_fit.csv"); fitB = pd.read_csv(T / "B_mlr_ols_fit.csv")
coef = olsA[["Term", "Coefficient (log scale)", "p-value", "Readable effect"]].merge(
    olsB[["Term", "Coefficient (log scale)", "p-value", "Readable effect"]], on="Term", how="outer", suffixes=(" A", " B"))
order = {t: i for i, t in enumerate(olsB["Term"])}
coef = coef.sort_values("Term", key=lambda s: s.map(order)).reset_index(drop=True)
coef.to_csv(T / "AB_mlr_coefficients.csv", index=False)
display(pd.concat([fitA.assign(Model="A"), fitB.assign(Model="B")]))
coef"""),
    md("### 4.2 Permutation importance (increase in CV RMSE when a variable is shuffled)"),
    code("""imp_tables = []
for s in ["A", "B"]:
    for k in C.ALGORITHMS:
        ps = R.permutation_summary(OUT[s]["results"][k]["perm"])
        imp_tables.append(ps.assign(**{"Model set": C.MODEL_SETS[s], "Algorithm": C.MODEL_NAMES[k]}))
imp = pd.concat(imp_tables)
imp.to_csv(T / "AB_permutation_importance.csv", index=False)
for s in ["A", "B"]:
    k = best["key"]
    ps = imp[(imp["Model set"] == C.MODEL_SETS[s]) & (imp["Algorithm"] == C.MODEL_NAMES[k])]
    show(P.importance_bars(ps, f"Permutation importance - {C.MODEL_SETS[s]}, {C.MODEL_NAMES[k]}"),
         f"AB_permutation_importance_{s}_{k}.png")
imp[imp["Model set"] == C.MODEL_SETS["B"]].pivot_table(index="Label", columns="Algorithm", values="Relative importance (%)") \\
   .sort_values(C.MODEL_NAMES[best["key"]], ascending=False)"""),
    code("""grp = imp[imp["Model set"] == C.MODEL_SETS["B"]].copy()
grp["Group"] = np.where(grp["Feature"].isin(C.GIS_FEATURES), "GIS / location (all distances)", "Property characteristics")
gshare = grp.groupby(["Algorithm", "Group"])["Relative importance (%)"].sum().unstack()
gshare.to_csv(T / "AB_importance_by_group.csv")
gshare"""),
    md("### 4.3 SHAP - best tree model, Model B"),
    code("""tree_key = best["key"] if best["key"] != "mlr" else comp[comp["key"] != "mlr"].sort_values("CV RMSE").iloc[0]["key"]
mB = OUT["B"]["results"][tree_key]["model"]
fB = sets[("B", tree_key)]
sv, base, Xe = I.shap_values(mB, train[fB], train[fB])
shap_imp = I.shap_importance(sv)
shap_imp.to_csv(T / "AB_shap_importance_B.csv", index=False)
fig = I.shap_summary_plot(sv, Xe, f"SHAP summary - Model B, {C.MODEL_NAMES[tree_key]} (training properties)")
show(fig, "AB_shap_summary_B.png")
shap_imp"""),
    md("""Each dot is a property. Its horizontal position is that variable's contribution to the (log) estimate,
relative to the average. The colour is the variable's value (red = high). The table converts the mean absolute
contribution into a typical percentage effect on Annual Value."""),
    md("### 4.4 Partial dependence (average effect of one variable, others held at their observed values)"),
    code("""ps_b = imp[(imp["Model set"] == C.MODEL_SETS["B"]) & (imp["Algorithm"] == C.MODEL_NAMES[tree_key])]
pdp_feats = [f for f in ps_b["Feature"] if f != "Condition"][:6]
show(I.pdp_figure(mB, train[fB], pdp_feats, f"Partial dependence - Model B, {C.MODEL_NAMES[tree_key]}"), "AB_pdp_B.png")
pdp_feats"""),
    md("""### 4.5 Worked example: explaining one property's assessment
The example is a typical test property: the one whose CMC Annual Value is closest to the test-set median. Two
explanations are shown.
* **MLR (Model B)** gives an itemised, rule-like explanation. It starts from a *reference property* (median floor area,
  one floor, median age, Fair condition, median distances). Each recorded characteristic then moves the estimate by a
  fixed percentage that is the same for every property, much like an adjustment on a valuation sheet.
* **The best tree model** gives a SHAP explanation for this property."""),
    code("""ex_idx = int(np.argmin(np.abs(yte - np.median(yte))))
ex = test.iloc[[ex_idx]]
raw_vals = {c: ex.iloc[0][c] for c in ["Age", "Condition", "Total_Floor_Area", "Log_Total_Floor_Area", "Num_Floors"] + C.GIS_FEATURES}
raw_vals["Log_Total_Floor_Area"] = f"log({ex.iloc[0]['Total_Floor_Area']:,.0f} sq ft)"
ex_info = ex[["Property_ID", "Assessment_No", "Age", "Condition", "Total_Floor_Area", "Num_Floors", C.TARGET] + C.GIS_FEATURES].T
ex_info.to_csv(T / "AB_worked_example_property.csv", header=["Value"])
display(ex_info)

mlrB = OUT["B"]["results"]["mlr"]["model"]; fM = sets[("B", "mlr")]
contrib_m, base_m, ref = I.mlr_itemised(mlrB, train, ex, fM)
ref_desc = (f"Reference property ({train['Total_Floor_Area'].median():,.0f} sq ft, {ref['Num_Floors'].iloc[0]:.0f} floor, "
            f"{ref['Age'].iloc[0]:.0f} yrs, {C.CONDITION_REFERENCE}, median distances)")
steps_m = I.waterfall_steps(contrib_m, base_m, raw_vals, base_label=ref_desc)
steps_m.to_csv(T / "AB_worked_example_mlr.csv", index=False)
ref.assign(Total_Floor_Area=train["Total_Floor_Area"].median()).to_csv(T / "AB_worked_example_reference.csv", index=False)
show(P.waterfall(steps_m, f"MLR (Model B) explanation - property {ex.iloc[0]['Property_ID']}\\n"
                          f"CMC Annual Value Rs. {ex.iloc[0][C.TARGET]:,.0f}"), "AB_worked_example_mlr.png")

sv_t, base_t, _ = I.shap_values(mB, train[fB], ex[fB])
steps_t = I.waterfall_steps(sv_t.iloc[0], base_t, raw_vals, top=6)
steps_t.to_csv(T / "AB_worked_example_tree.csv", index=False)
show(P.waterfall(steps_t, f"{C.MODEL_NAMES[tree_key]} (Model B) SHAP explanation - property {ex.iloc[0]['Property_ID']}\\n"
                          f"CMC Annual Value Rs. {ex.iloc[0][C.TARGET]:,.0f}"), "AB_worked_example_tree.png")
assert abs(steps_m["End (Rs.)"].iloc[-1] - mlrB.predict(ex[fM])[0]) < 1
assert abs(steps_t["End (Rs.)"].iloc[-1] - mB.predict(ex[fB])[0]) / mB.predict(ex[fB])[0] < 0.01
steps_m[["Label", "Effect (%)", "Change (Rs.)", "End (Rs.)"]]"""),
    md("## 5. Supplementary checks"),
    md("""### 5.1 Spatially grouped cross-validation
The dataset has no road field, so a road-grouped validation (methodology 3.8) is approximated. The training
properties are grouped into 10 spatial clusters (k-means on their standardised GIS distances), and whole clusters are
held out. This tests whether the models carry over to **parts of the ward they were not trained on**, which random
CV does not test."""),
    code("""groups = M.spatial_groups(train)
sp = []
for s in ["A", "B"]:
    for k in C.ALGORITHMS:
        r = OUT[s]["results"][k]
        f = M.spatial_cv(k, train, ytr, r["features"], categories, r["params"], groups)
        sp.append({"Model set": C.MODEL_SETS[s], "Algorithm": C.MODEL_NAMES[k], "Spatial CV RMSE": f["RMSE"].mean(),
                   "Spatial CV MAE": f["MAE"].mean(), "Spatial CV R2": f["R2"].mean(),
                   "Random CV RMSE": r["cv"]["CV RMSE"]})
spatial = pd.DataFrame(sp)
spatial.to_csv(T / "AB_spatial_cv.csv", index=False)
print("Cluster sizes:", np.bincount(groups))
spatial"""),
    md("### 5.2 Raw-target sensitivity (models fitted on Annual Value in rupees instead of log)"),
    code("""raw = []
for s in ["A", "B"]:
    for k in C.ALGORITHMS:
        r = OUT[s]["results"][k]
        m = M.build_model(k, r["features"], categories, r["params"], log_target=False).fit(train[r["features"]], ytr)
        p = m.predict(test[r["features"]])
        st = M.ratio_statistics(yte, p)
        raw.append({"Model set": C.MODEL_SETS[s], "Algorithm": C.MODEL_NAMES[k], **{f"Test {a}": b for a, b in M.metrics(yte, p).items()},
                    "Log-target test RMSE": r["test"]["RMSE"], "COD (%)": st["COD (%)"], "PRD": st["PRD"],
                    "Negative estimates": st["Negative estimates"]})
rawt = pd.DataFrame(raw)
rawt.to_csv(T / "AB_raw_target_sensitivity.csv", index=False)
rawt"""),
    md("""The tuned hyperparameters from the log-target search are reused here, so this compares only the target scale.
It shows whether the main conclusions depend on the log transform."""),
    md("## 6. Build the Chapter 4 draft"),
    code("""from src import report
path = report.build()
print("Chapter 4 draft written to", path)"""),
]

write("01_Model_A_property_only.ipynb", nb1)
write("02_Model_B_property_plus_GIS.ipynb", nb2)
write("03_Model_A_vs_B_comparison_and_interpretation.ipynb", nb3)
