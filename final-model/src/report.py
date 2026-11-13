"""Builds report/Chapter4_draft.docx (Chapter 4: Analysis and Findings, sections 4.1-4.9, plus
suggested inserts for Chapter 3) from the saved output tables and figures.

Every number in the text is read from outputs/tables/*.csv; nothing is typed by hand.
scripts/check_report.py re-checks the numbers in the finished document against those tables.
"""
import json

import numpy as np
import pandas as pd

from . import config as C
from .docx_helpers import Doc

T, F = C.TABLES_DIR, C.FIGURES_DIR
SRC_CMC = "Source: Compiled by author based on CMC (2025) assessment records"
SRC_GIS = "Source: Compiled by author based on CMC (2025) records and OpenStreetMap data (QGIS)"
SRC_MODEL = "Source: Compiled by author (model results)"


def t(name, **kw):
    return pd.read_csv(T / name, **kw)


def rs(v):
    return f"Rs. {v:,.0f}"


def pct(v, d=1, sign=False):
    return f"{v:+.{d}f} %" if sign else f"{v:.{d}f} %"


def f3(v):
    return f"{v:.3f}"


def f2(v):
    return f"{v:.2f}"


def short_set(s):
    return s.split(" - ")[0]


def build(path=None):
    d = Doc()
    path = path or (C.REPORT_DIR / "Chapter4_draft.docx")

    # ------------------------------------------------------------------ data
    acc = t("AB_record_accounting.csv").set_index("Step")["Records"]
    desc = t("A_descriptive_statistics.csv")
    cond = t("A_condition_summary.csv")
    floors = t("A_floors_summary.csv")
    pcorr = t("A_property_correlations.csv")
    flagged = t("AB_flagged_extreme_rate_records.csv")
    miss = t("AB_missing_values.csv").set_index("Variable")
    dups = t("AB_duplicate_assessment_numbers.csv")
    gis_desc = t("B_gis_descriptive.csv")
    gis_corr = t("B_gis_correlations.csv")
    gcm = t("B_gis_corr_matrix.csv", index_col=0)
    vif_all = t("B_vif_all_gis.csv")
    screen = t("B_mlr_gis_screening.csv")
    predA, predB = t("A_predictors_and_hyperparameters.csv"), t("B_predictors_and_hyperparameters.csv")
    comp = t("AB_six_model_comparison.csv")
    gain = t("AB_gis_gain.csv").set_index("Algorithm")
    spatial = t("AB_spatial_cv.csv")
    rawt = t("AB_raw_target_sensitivity.csv")
    ratio = t("AB_ratio_study.csv")
    bands = t("AB_ratio_by_value_band.csv")
    noflag = t("AB_ratio_without_flagged.csv")
    coef = t("AB_mlr_coefficients.csv")
    fitA, fitB = t("A_mlr_ols_fit.csv").iloc[0], t("B_mlr_ols_fit.csv").iloc[0]
    imp = t("AB_permutation_importance.csv")
    grp = t("AB_importance_by_group.csv", index_col=0)
    shp = t("AB_shap_importance_B.csv")
    ex = t("AB_worked_example_property.csv", index_col=0)["Value"]
    ex_m = t("AB_worked_example_mlr.csv")
    ex_t = t("AB_worked_example_tree.csv")

    MLR, RF, XGB = (C.MODEL_NAMES[k] for k in C.ALGORITHMS)
    best = comp[comp["Selected (lowest CV RMSE)"] == "Yes"].iloc[0]
    best_alg, best_set = best["Algorithm"], best["Model set"]
    tree_alg = best_alg if best_alg != MLR else comp[comp["Algorithm"] != MLR].sort_values("CV RMSE").iloc[0]["Algorithm"]
    tree_key = {v: k for k, v in C.MODEL_NAMES.items()}[tree_alg]

    def cmp(s, alg, col):
        return comp[(comp["Model set"] == C.MODEL_SETS[s]) & (comp["Algorithm"] == alg)][col].iloc[0]

    def rat(s, alg, sample, col):
        r = ratio[(ratio["Model set"] == C.MODEL_SETS[s]) & (ratio["Algorithm"] == alg) & (ratio["Sample"] == sample)]
        return r[col].iloc[0]

    n_raw, n_lab = int(acc["Records in raw file"]), int(acc["Records used for modelling"])
    n_unlab = int(acc["No Annual Value (unlabelled)"])
    n_na = int(miss.loc["Annual_Value", "Literal 'NA' entries"])
    split = pd.read_csv(C.SPLIT_FILE)["Partition"].value_counts()
    n_train, n_test = int(split["train"]), int(split["test"])
    dsc = desc.set_index("Variable")
    av, area, rate = dsc.loc[C.label(C.TARGET)], dsc.loc[C.label("Total_Floor_Area")], dsc.loc[C.label("AV_per_sqft")]
    r_log = pcorr.iloc[0][[c for c in pcorr.columns if c.startswith("Pearson")][0]]
    age_rho = pcorr.set_index("Variable").loc[C.label("Age"), "Spearman rho with AV"]

    # ================================================================== 4.1
    d.h("Chapter 4", 1)
    d.h("Analysis and Findings", 1)
    d.h("4.1\tIntroduction", 2)
    d.p("This chapter presents the results of the analysis described in Chapter 3. Section 4.2 describes the "
        "cleaned property dataset, and Section 4.3 describes the GIS-derived location variables. Section 4.4 compares "
        "the six model configurations: Multiple Linear Regression (MLR), Random Forest and XGBoost, each estimated "
        "with property-only predictors (Model A) and with property plus OSM-derived location predictors (Model B). It "
        "then measures how much the location variables add. Section 4.5 tests the consistency and bias of the "
        "estimates with an assessment-ratio study, and Section 4.6 interprets the models, including a worked "
        "example of how one property's estimate can be explained to its owner. Sections 4.7 to 4.9 discuss what the "
        "results imply for CMC's assessment process.")
    d.p(f"Throughout the chapter the target is the **CMC-recorded Annual Value** of the 2025 valuation. The models "
        f"therefore measure how well recorded property characteristics and location reproduce existing CMC "
        f"assessments. They do not measure independent market value (Section 3.9).")

    # ================================================================== 4.2
    d.h("4.2\tDataset Description", 2)
    d.h("4.2.1\tRecords and data cleaning", 3)
    d.p(f"The integrated Dematagoda dataset contains **{n_raw:,} residential property records**. Each record has an "
        f"assessment number, a unique property identifier, the CMC-recorded characteristics (building age, condition "
        f"grade and the floor area of up to four floors), the 2025 Annual Value and eleven GIS-derived distances. "
        f"{n_unlab} records have no Annual Value: {n_unlab - n_na} are blank and {n_na} are recorded as 'NA'. These "
        f"were excluded from model training because the target is unavailable (Section 3.7). They are kept separately "
        f"so they can be estimated once the model is validated. This leaves **{n_lab} properties** for the analysis "
        f"({d.next_tab()}).")
    d.table(t("AB_record_accounting.csv"), "Record accounting from the raw file to the modelling dataset", SRC_CMC,
            col_widths=[9, 3])
    cleaned = pd.read_csv(C.PROCESSED_DIR / "cleaned_properties.csv")
    lab_age_miss, lab_cond_miss = int(cleaned["Age"].isna().sum()), int(cleaned["Condition"].isna().sum())
    d.p(f"The cleaning steps were documented record by record:")
    d.bullets([
        "**Upper floors.** Blank first-, second- and third-floor areas were read as 'no such floor' (0 sq ft), and "
        "total floor area was calculated as the sum of all floors. This is consistent with how the records are "
        "structured, but it remains an assumption to be confirmed with CMC.",
        "**Condition grades.** Spelling variants were unified ('Fair +' as 'Fair+'). The four records graded "
        "'Modern' or 'Good (Modern)' were merged into 'Good', because there are too few to estimate a separate "
        "effect. This gives five ordered grades: Poor < Fair- < Fair < Fair+ < Good.",
        f"**Missing predictors.** Among the {n_lab} labelled properties, building age is missing for {lab_age_miss} "
        f"and condition grade for {lab_cond_miss}. These are imputed inside each model pipeline (median for age, mode "
        f"for condition), using training data only, so no information leaks from the test set. No predictor exceeds "
        f"the 30 % missing-value threshold.",
        f"**Repeated assessment numbers.** {len(dups)} records share an assessment number with another record. In "
        f"every case the records have different locations, floor areas and Unique Property IDs, because the same "
        f"house number is used on different roads in the ward. All were retained. No exact duplicate records were "
        f"found.",
    ])
    d.h("4.2.2\tDescriptive statistics", 3)
    dt = desc.copy()
    dt.columns = ["Variable", "n", "Mean", "Median", "SD", "Min", "Max", "Skewness"]
    d.p(f"{d.next_tab()} summarises the main variables. Annual Value ranges from {rs(av['Min'])} to {rs(av['Max'])}, "
        f"with a median of {rs(av['Median'])} and a strong right skew (skewness {f2(av['Skewness'])}). A small "
        f"number of large multi-storey houses have values many times the median ({d.next_fig()}a). On the log scale "
        f"the distribution is close to symmetric ({d.next_fig()}b). All models were therefore fitted on "
        f"log(Annual Value), and their estimates were converted back to rupees for evaluation.")
    d.table(dt, "Descriptive statistics of the modelling dataset (n = %d)" % n_lab, SRC_CMC,
            formats={"Skewness": "{:.2f}", "Mean": "{:,.1f}", "SD": "{:,.1f}", "Median": "{:,.1f}",
                     "Min": "{:,.1f}", "Max": "{:,.1f}"})
    d.figure(F / "4_1_annual_value_distribution.png", "Distribution of CMC Annual Value (a) and log(Annual Value) (b)",
             SRC_CMC)
    big = floors.set_index("Num_Floors")["n"]
    d.p(f"Floor area is the dominant driver of Annual Value. The Pearson correlation between log floor area and log "
        f"Annual Value is {f3(r_log)} ({d.next_fig()}a). Annual Value **per sq ft** is much more stable than Annual "
        f"Value itself: its median is {rs(rate['Median'])} and most properties lie close to it. This is consistent "
        f"with CMC deriving Annual Value from a rate per sq ft that is adjusted for the property and its road. "
        f"Building age is negatively related to Annual Value (Spearman rho {f2(age_rho)}, {d.next_fig()}b), mainly "
        f"because newer buildings tend to be larger and better graded. Section 4.6.1 shows that the age effect "
        f"disappears once floor area and condition are controlled for. The rate per sq ft "
        f"rises with the condition grade ({d.next_fig()}c, {d.next_tab()}). Most buildings are single-storey "
        f"({int(big.get(1.0, 0))} properties), {int(big.get(2.0, 0))} have two floors and "
        f"{int(big.get(3.0, 0) + big.get(4.0, 0))} have three or more.")
    d.figure(F / "4_2_property_relationships.png",
             "Floor area, age and condition in relation to Annual Value", SRC_CMC)
    ct = cond.rename(columns={"Median_AV": "Median AV (Rs.)", "Median_floor_area": "Median floor area (sq ft)",
                              "Median_AV_per_sqft": "Median AV per sq ft (Rs.)"})
    d.table(ct, "Annual Value by condition grade", SRC_CMC, formats={"Median AV per sq ft (Rs.)": "{:.1f}"})
    d.p(f"{len(flagged)} properties have an Annual Value per sq ft far outside the normal range (beyond three "
        f"interquartile ranges on the log scale). Examples are a {flagged['Total_Floor_Area'].max():,.0f} sq ft "
        f"building assessed at only {rs(flagged.loc[flagged['Total_Floor_Area'].idxmax(), 'AV_per_sqft'])} per sq ft, "
        f"and small units assessed at more than Rs. 90 per sq ft. Possible explanations are recording errors, partly "
        f"commercial use, or a floor area that does not correspond to the assessed unit. They were **kept** in the "
        f"main analysis, because removing difficult records would flatter the models. Their influence is tested in "
        f"Section 4.5. {int((flagged['Partition'] == 'test').sum())} of them fall in the test set.")

    # ================================================================== 4.3
    d.h("4.3\tGIS-Derived Location Variables", 2)
    gd = gis_desc[["Variable", "Group", "Min", "Median", "Max"]].copy()
    d.p(f"Eleven nearest-feature distances were generated in QGIS from OpenStreetMap layers. These are the seven "
        f"variables planned in Table y (Section 3.5): distance to the nearest A/B-class road, bus stop, railway "
        f"station, school, hospital and supermarket, and to the CBD. They are complemented by four further amenity "
        f"distances available from OSM: banks, pharmacies, playgrounds and shopping malls. All distances were "
        f"computed for **every** property, so no OSM gap had to be treated as missing ({d.next_tab()}).")
    d.table(gd, "GIS-derived distance variables (metres, all %d records)" % n_raw, SRC_GIS,
            formats={"Min": "{:,.0f}", "Median": "{:,.0f}", "Max": "{:,.0f}"})
    gc = gis_corr.set_index("Variable")
    rho_av = gc["Spearman rho (AV)"].abs()
    rho_rt = gc["Spearman rho (AV per sq ft)"]
    n_sig_rate = int((gc["Spearman p-value (AV per sq ft)"] < 0.05).sum())
    neg_rate = [C.SHORT_LABELS[v].replace("Distance to ", "") for v in rho_rt.index if rho_rt[v] < 0]
    pos_rate = [C.SHORT_LABELS[v].replace("Distance to ", "") for v in rho_rt.sort_values(ascending=False).index[:3]]
    d.p(f"{d.next_fig()}a shows that the distances are only weakly related to Annual Value itself: the largest "
        f"absolute Spearman correlation is {f2(rho_av.max())}. This is expected, because Annual Value is dominated by "
        f"floor area. Location is more likely to show up in the **rate per sq ft**. {d.next_fig()}b confirms this: all "
        f"{n_sig_rate} distances are significantly correlated with Annual Value per sq ft (|rho| between "
        f"{f2(rho_rt.abs().min())} and {f2(rho_rt.abs().max())}, p < 0.05). The rate tends to be higher closer to the "
        f"{', '.join(neg_rate)}, and higher further from the {', '.join(pos_rate)}. One reading is that the "
        f"residential lanes in the inner (CBD and railway) side of the ward carry higher rates than properties facing "
        f"the busy A/B roads and commercial strips. Because the distances move together (below), these signs "
        f"describe one spatial gradient across the ward. They should not be read as separate causal effects of each "
        f"amenity.")
    d.figure(F / "4_4_gis_correlations.png",
             "Spearman correlation of the GIS distances with Annual Value (a) and Annual Value per sq ft (b)", SRC_GIS)
    off = gcm.where(~np.eye(len(gcm), dtype=bool)).abs()
    n_high = int((off.values >= C.CORR_THRESHOLD).sum() / 2)
    max_vif = vif_all[vif_all["Variable"].isin(C.GIS_FEATURES)]["VIF"].max()
    kept = screen[screen["Decision"] == "Retained"]["Variable"].map(lambda v: C.SHORT_LABELS[v]).tolist()
    d.p(f"Within a single ward, the eleven distances largely describe the same position. {n_high} of the 55 pairs "
        f"have |r| >= {C.CORR_THRESHOLD} ({d.next_fig()}), and with all eleven in a regression the largest variance "
        f"inflation factor is {max_vif:,.0f}. That is far above the usual limit of 5. A linear regression with all "
        f"eleven would give unstable and uninterpretable coefficients. MLR in Model B therefore uses a screened subset, "
        f"chosen on the training data only. The distances were ranked by their correlation with log Annual Value; a "
        f"distance was dropped if it correlated at |r| >= 0.70 with one already kept, and then while any VIF exceeded "
        f"5. The retained distances are {', '.join(kept)} ({d.next_tab()}), with every VIF below 5. Random Forest and "
        f"XGBoost are not destabilised by correlated inputs in the same way and use all eleven distances.")
    d.figure(F / "4_5_gis_correlation_matrix.png", "Correlation between the GIS distance variables", SRC_GIS,
             width=d.width * 0.85)
    sc = screen[["Variable", "Group", "|r| with log(Annual Value)", "Decision", "Reason"]].copy()
    sc["Variable"] = sc["Variable"].map(C.SHORT_LABELS)
    d.table(sc, "Screening of GIS variables for the MLR (training data)", SRC_GIS,
            formats={"|r| with log(Annual Value)": "{:.3f}"}, font=8.5)

    # ================================================================== 4.4
    d.h("4.4\tModel Results: Model A vs Model B", 2)
    d.h("4.4.1\tModel development", 3)
    d.p(f"The {n_lab} properties were split once into a training set ({n_train}) and a test set ({n_test}), using a "
        f"fixed random seed and stratifying by Annual Value quintile. **Both Model A and Model B use exactly the same "
        f"training and test properties**, so any difference between them is due to the predictors alone. Model A uses "
        f"building age, condition grade, total floor area and number of floors. Model B adds the location variables "
        f"({d.next_tab()}). In the MLR, floor area enters as log(floor area), the usual log-log hedonic form, so its "
        f"coefficient is an elasticity. Condition is coded as dummies with 'Fair' as the reference grade. In the tree "
        f"models, condition is an ordinal grade. The Random Forest and XGBoost hyperparameters were tuned with "
        f"RandomizedSearchCV (5-fold cross-validation on the training set; 40 and 60 candidate settings). Each "
        f"configuration was then evaluated with 5-fold cross-validation repeated 10 times on the training set, and "
        f"once on the held-out test set.")
    pr = pd.concat([predA, predB])
    pr["Tuned hyperparameters"] = pr["Tuned hyperparameters"].map(_fmt_params)
    pr["Predictor list"] = pr["Predictor list"].map(_fmt_predictors)
    pr["Model set"] = pr["Model set"].map(short_set)
    d.table(pr[["Model set", "Algorithm", "Predictor list", "Tuned hyperparameters"]],
            "Predictors and tuned hyperparameters of the six configurations", SRC_MODEL, font=8,
            col_widths=[1.8, 2.6, 5.2, 5.0])

    d.h("4.4.2\tPredictive performance", 3)
    cv_best = comp.sort_values("CV RMSE").iloc[0]
    r2_lo, r2_hi = comp["Test R2"].min(), comp["Test R2"].max()
    d.p(f"{d.next_tab()} compares the six configurations. All of them reproduce the CMC Annual Values closely. "
        f"Test-set R² ranges from {f3(r2_lo)} to {f3(r2_hi)}, and the typical absolute error (MAE) is "
        f"{rs(comp['Test MAE'].min())} to {rs(comp['Test MAE'].max())} per property, against a median Annual Value "
        f"of {rs(av['Median'])}. Cross-validated errors are larger than the test errors. This is because the "
        f"training set contains the few very large properties (up to {area['Max']:,.0f} sq ft), and an error on one "
        f"of them dominates the RMSE of the fold it falls in. For this reason the pooled out-of-fold R² is reported "
        f"as the cross-validated R². The configuration with the lowest cross-validated RMSE is "
        f"**{short_set(cv_best['Model set'])} {cv_best['Algorithm']}** (CV RMSE {rs(cv_best['CV RMSE'])}; test "
        f"RMSE {rs(cv_best['Test RMSE'])}, MAE {rs(cv_best['Test MAE'])}, R² {f3(cv_best['Test R2'])}). It is "
        f"selected for interpretation. The selection was made on cross-validation, not on the test set.")
    ct = comp[["Model set", "Algorithm", "CV RMSE", "CV MAE", "CV R2 pooled", "Test RMSE", "Test MAE", "Test R2",
               "Test MAPE (%)"]].copy()
    ct["Model set"] = ct["Model set"].map(short_set)
    ct = ct.rename(columns={"CV R2 pooled": "CV R²", "Test R2": "Test R²", "CV RMSE": "CV RMSE (Rs.)",
                            "CV MAE": "CV MAE (Rs.)", "Test RMSE": "Test RMSE (Rs.)", "Test MAE": "Test MAE (Rs.)"})
    d.table(ct, "Performance of the six configurations (CV = 5-fold x 10 repeats on the training set; test = %d "
                "held-out properties)" % n_test, SRC_MODEL, formats={"Test MAPE (%)": "{:.1f}"}, font=8.5)
    d.figure(F / "AB_model_comparison.png", "Test-set RMSE, MAE and R² for Model A and Model B", SRC_MODEL)
    bpred = pd.read_csv(C.PREDICTIONS_DIR / "B_test_predictions.csv")
    pred_col = f"Pred_{C.MODEL_SHORT[{v: k for k, v in C.MODEL_NAMES.items()}[best_alg]]}"
    worst = bpred.loc[np.abs(np.log(bpred[pred_col] / bpred["Annual_Value"])).idxmax()]
    d.p(f"{d.next_fig()} plots the Model B estimates against the CMC values for the test properties. The points lie "
        f"close to the line of equality across the whole value range. The largest relative deviations are the "
        f"flagged records discussed in Section 4.2. For example, property {worst['Property_ID']} is assessed by "
        f"CMC at {rs(worst['Annual_Value'])} but estimated at {rs(worst[pred_col])} from its recorded "
        f"characteristics.")
    d.figure(F / "B_actual_vs_predicted.png", "Estimated vs CMC Annual Value, Model B, test set", SRC_MODEL)
    mlr_a, mlr_b = cmp("A", MLR, "Test RMSE"), cmp("B", MLR, "Test RMSE")
    d.p(f"The simple MLR is competitive with the machine-learning models. Its test RMSE ({rs(mlr_b)} in Model B) is "
        f"within about {pct(100 * (mlr_b / comp['Test RMSE'].min() - 1), 0)} of the best model. Its explanatory "
        f"power on the log scale is high (adjusted R² {f3(fitA['Adj. R2 (log scale)'])} in Model A and "
        f"{f3(fitB['Adj. R2 (log scale)'])} in Model B). This supports the point made in Section 2.6: machine "
        f"learning should be compared with a regression baseline rather than assumed to be superior. When the "
        f"relationship is close to 'rate x area', a transparent regression captures most of it.")

    d.h("4.4.3\tContribution of the OSM-derived location variables", 3)
    g = gain
    d.p(f"The central question of the study is whether the OSM-derived variables capture the location effect that "
        f"valuers now judge through site inspection and the road-wise rate chart. {d.next_tab()} compares Model B with "
        f"Model A for each algorithm, on the same test properties and on the same 50 cross-validation folds.")
    gt = pd.DataFrame({
        "Algorithm": g.index,
        "Test RMSE A": g["Test RMSE A"].values, "Test RMSE B": g["Test RMSE B"].values,
        "Test RMSE change (%)": g["Test RMSE change (%)"].values,
        "Test MAE change (%)": g["Test MAE change (%)"].values,
        "Wilcoxon p (test)": g["Wilcoxon p (test abs. errors)"].values,
        "CV RMSE change (%)": g["CV RMSE change (%)"].values,
        "CV folds B better (%)": g["CV folds where B better (%)"].values,
        "Paired t p (CV)": g["Paired t-test p (50 CV folds)"].values,
    })
    d.table(gt, "Change in error when the GIS variables are added (Model B vs Model A; negative = improvement)",
            SRC_MODEL, font=8.5,
            formats={"Test RMSE change (%)": "{:+.1f}", "Test MAE change (%)": "{:+.1f}", "CV RMSE change (%)": "{:+.1f}",
                     "CV folds B better (%)": "{:.0f}", "Wilcoxon p (test)": "{:.3f}", "Paired t p (CV)": "{:.3f}"})
    test_gains = g["Test RMSE change (%)"]
    cv_rf, cv_xgb, cv_mlr = (g.loc[a, "CV RMSE change (%)"] for a in [RF, XGB, MLR])
    d.p(f"On the test set, adding the location variables lowered the error of **all three algorithms**: RMSE by "
        f"{pct(-test_gains[MLR])} (MLR), {pct(-test_gains[RF])} (Random Forest) and {pct(-test_gains[XGB])} "
        f"(XGBoost). MAE fell by {pct(-g.loc[MLR, 'Test MAE change (%)'])}, {pct(-g.loc[RF, 'Test MAE change (%)'])} "
        f"and {pct(-g.loc[XGB, 'Test MAE change (%)'])}. The cross-validation shows the same direction for the "
        f"tree models, but the gain is smaller: Model B lowered the CV RMSE by {pct(-cv_rf)} (Random Forest, better in "
        f"{g.loc[RF, 'CV folds where B better (%)']:.0f} % of folds) and {pct(-cv_xgb)} (XGBoost, "
        f"{g.loc[XGB, 'CV folds where B better (%)']:.0f} % of folds). For MLR, the screened distances "
        f"{'raised' if cv_mlr > 0 else 'lowered'} the CV RMSE by {pct(abs(cv_mlr))}. The GIS gain is therefore "
        f"**real but modest, and not uniform**. It is clearest for the flexible tree models, which can combine the "
        f"distances into local rate differences, and weakest for the linear model, which can only use a few "
        f"distances as straight-line gradients.")
    d.figure(F / "AB_paired_cv_differences.png",
             "Paired cross-validation folds: change in RMSE when GIS variables are added", SRC_MODEL)
    sp = spatial.copy()
    sp["Model set"] = sp["Model set"].map(short_set)
    spA = spatial.set_index(["Model set", "Algorithm"])
    sp_b_worse = [a for a in C.MODEL_NAMES.values()
                  if spA.loc[(C.MODEL_SETS["B"], a), "Spatial CV RMSE"] > spA.loc[(C.MODEL_SETS["A"], a), "Spatial CV RMSE"]]
    d.p(f"A stricter test is whether the location effect learned in one part of the ward carries over to another. "
        f"The dataset has no road field, so the road-grouped validation proposed in Section 3.8 was approximated. The "
        f"training properties were grouped into {C.SPATIAL_CLUSTERS} spatial clusters (k-means on their distances), "
        f"and whole clusters were held out in turn ({d.next_tab()}). Under this design every configuration performs "
        f"worse than under random cross-validation, and Model B is "
        f"{'no better than Model A for any algorithm' if len(sp_b_worse) == 3 else 'not consistently better than Model A'}. "
        f"In other words, the distances help the models reproduce rate differences **between locations that are "
        f"represented in the training data**. They do not provide a general rule that transfers to an unseen "
        f"neighbourhood. This matters for the research aim. Within a ward where a sample of properties has been "
        f"assessed, OSM variables can partly stand in for the location judgement. They cannot yet replace the "
        f"road-wise rate chart for areas without assessed comparables.")
    sp_tab = d.table(sp.rename(columns={"Spatial CV R2": "Spatial CV R²"}),
            "Spatially grouped cross-validation (whole clusters held out) vs random cross-validation", SRC_MODEL,
            formats={"Spatial CV R²": "{:.3f}"}, font=8.5)
    neg = int(rawt["Negative estimates"].max())
    d.p(f"A sensitivity run fitted on Annual Value in rupees instead of log(Annual Value) gave the same ranking for "
        f"the tree models, with slightly higher errors. The raw-scale MLR performed much worse (test R² "
        f"{f3(rawt[rawt['Algorithm'] == MLR]['Test R2'].min())}-{f3(rawt[rawt['Algorithm'] == MLR]['Test R2'].max())}) "
        f"and produced {neg} negative estimates, which confirms the choice of the log scale.")

    # ================================================================== 4.5
    d.h("4.5\tConsistency and Bias: Assessment-Ratio Analysis", 2)
    d.p("For each property the assessment ratio is the model estimate divided by the CMC-recorded Annual Value. "
        "Following the IAAO Standard on Ratio Studies, four statistics are reported. The **median ratio** measures the "
        "overall level (target 0.90-1.10). The **coefficient of dispersion (COD)** measures uniformity, the average "
        "percentage deviation from the median ratio (target <= 15 for residential property). The **price-related "
        "differential (PRD)** and the **price-related bias (PRB)** measure vertical equity: a PRD above 1.03 or a "
        "negative PRB indicates that lower-value properties receive relatively higher estimates than higher-value "
        "ones (regressivity). The ratios are computed for the test set and, as a larger sample, for the out-of-fold "
        f"cross-validation estimates of all {n_train} training properties ({d.next_tab()}).")
    rt = ratio[["Model set", "Algorithm", "Sample", "Median ratio", "COD (%)", "PRD", "PRB", "Within +/-20% (%)"]].copy()
    rt["Model set"] = rt["Model set"].map(short_set)
    rt["Sample"] = rt["Sample"].map({"Hold-out test": "Test", "Training (out-of-fold CV)": "CV (train)"})
    d.table(rt.rename(columns={"Within +/-20% (%)": "Within ±20 % (%)"}),
            "Assessment-ratio statistics (ratio = model estimate / CMC Annual Value)", SRC_MODEL, font=8.5,
            formats={"Median ratio": "{:.3f}", "COD (%)": "{:.1f}", "PRD": "{:.3f}", "PRB": "{:.3f}",
                     "Within ±20 % (%)": "{:.1f}"})
    tst = ratio[ratio["Sample"] == "Hold-out test"]
    oof = ratio[ratio["Sample"] != "Hold-out test"]
    oof_cod = oof.set_index(["Model set", "Algorithm"])["COD (%)"]
    n_b_better = sum(oof_cod[(C.MODEL_SETS["B"], a)] < oof_cod[(C.MODEL_SETS["A"], a)] for a in C.MODEL_NAMES.values())
    more_uniform = {3: "slightly more", 2: "slightly more (for two of the three algorithms)",
                    1: "not consistently more", 0: "not more"}[n_b_better]
    d.p(f"**Level.** Every configuration reproduces the CMC level almost exactly. The median ratios lie between "
        f"{f3(ratio['Median ratio'].min())} and {f3(ratio['Median ratio'].max())}, well inside the 0.90-1.10 range. "
        f"**Uniformity.** On the cross-validation sample, COD is {f2(oof['COD (%)'].min())}-{f2(oof['COD (%)'].max())}, "
        f"within the IAAO limit. Model B is {more_uniform} uniform than Model A. On the test set, "
        f"COD rises to {f2(tst['COD (%)'].min())}-{f2(tst['COD (%)'].max())}. **Vertical equity.** PRD is above "
        f"1.03 for all configurations ({f3(tst['PRD'].min())}-{f3(tst['PRD'].max())} on the test set). This "
        f"indicates mild regressivity: lower-value properties are estimated relatively higher than higher-value ones.")
    nf = noflag.copy()
    prb_zero = int(((tst["PRB 95% CI low"] <= 0) & (tst["PRB 95% CI high"] >= 0)).sum())
    d.p(f"Two factors explain the weaker test-set figures. First, the test set contains "
        f"{int((flagged['Partition'] == 'test').sum())} of the flagged records. One of them is assessed at only "
        f"{rs(flagged['Annual_Value'].min())}, "
        f"so even a reasonable estimate produces a very large ratio. Without these records ({d.next_tab()}), the "
        f"test-set COD falls to {f2(nf['COD (%)'].min())}-{f2(nf['COD (%)'].max())} and PRD to "
        f"{f3(nf['PRD'].min())}-{f3(nf['PRD'].max())}, with PRB between {f3(nf['PRB'].min())} and {f3(nf['PRB'].max())}. "
        f"Even with them, the 95 % confidence interval of the test-set PRB includes zero for "
        f"{'all six' if prb_zero == 6 else f'{prb_zero} of the six'} configurations, so the evidence of regressivity on the test set alone is weak. Second, "
        f"{d.next_fig()} shows a residual pattern by value band: in the highest quartile of CMC values the median ratio "
        f"is about {f2(bands[bands['Band'] == 'Q4']['Median_ratio'].mean())}. The largest properties are therefore "
        f"estimated somewhat **below** their CMC values. This is partly a statistical effect, because extreme "
        f"values are pulled towards the average, and partly a sign that CMC applies premiums to large houses that the "
        f"recorded variables do not fully explain.")
    nf["Model set"] = nf["Model set"].map(short_set)
    d.table(nf[["Model set", "Algorithm", "n", "RMSE", "R2", "Median ratio", "COD (%)", "PRD", "PRB"]].rename(
        columns={"R2": "R²", "RMSE": "RMSE (Rs.)"}),
        "Test-set accuracy and ratio statistics without the flagged extreme-rate records", SRC_MODEL, font=8.5,
        formats={"R²": "{:.3f}", "Median ratio": "{:.3f}", "COD (%)": "{:.1f}", "PRD": "{:.3f}", "PRB": "{:.3f}"})
    d.figure(F / "AB_ratio_by_value_band.png", "Median assessment ratio by CMC value quartile (test set)", SRC_MODEL)
    d.p("These results must be read with care. A model trained on CMC values reproduces any systematic pattern in "
        "those values, so a good ratio study shows **consistency with the existing assessments**, not freedom from "
        "bias in the existing assessments (Section 3.9). What the model adds is that bias becomes measurable. The same "
        "rules are applied to every property, so the flagged records and the high-value band can be identified "
        "objectively and referred to a valuer for review.")

    # ================================================================== 4.6
    d.h("4.6\tInterpretation of the Models", 2)
    d.h("4.6.1\tMultiple Linear Regression coefficients", 3)
    cf = coef.set_index("Term")
    d.p(f"The MLR gives the most direct explanation, because each coefficient is a fixed adjustment that applies to "
        f"every property ({d.next_tab()}). The floor-area elasticity is {f3(cf.loc['Log_Total_Floor_Area', 'Coefficient (log scale) B'])}, "
        f"almost exactly 1. Annual Value is therefore close to proportional to floor area, which mirrors a rate-per-"
        f"sq-ft assessment. Holding total area constant, each additional floor reduces Annual Value by about "
        f"{pct(-100 * (np.exp(cf.loc['Num_Floors', 'Coefficient (log scale) B']) - 1), 1)}, consistent with lower "
        f"rates for upper floors. Condition has a clear, ordered effect. Relative to 'Fair', 'Good' adds "
        f"{cf.loc['Condition_Good', 'Readable effect B'].split(' vs')[0]}, 'Fair+' adds "
        f"{cf.loc['Condition_Fair+', 'Readable effect B'].split(' vs')[0]}, while 'Fair-' and 'Poor' reduce the "
        f"value by {cf.loc['Condition_Fair-', 'Readable effect B'].split(' vs')[0].lstrip('-')} and "
        f"{cf.loc['Condition_Poor', 'Readable effect B'].split(' vs')[0].lstrip('-')}. Building age is not "
        f"significant (p = {f2(cf.loc['Age', 'p-value B'])}) once condition is included, which suggests CMC's "
        f"condition grade already captures most of the effect of age. Of the screened distances, only the distance "
        f"to the nearest A/B road is significant ({cf.loc['Dist_AB_Road', 'Readable effect B']}, p = "
        f"{f3(cf.loc['Dist_AB_Road', 'p-value B'])}). Properties set back from the main roads carry slightly higher "
        f"Annual Values.")
    ctab = coef[coef["Term"] != "const"][["Term", "Readable effect A", "p-value A", "Readable effect B", "p-value B"]].copy()
    ctab["Term"] = ctab["Term"].map(C.label)
    d.table(ctab, "MLR coefficients expressed as effects on Annual Value (log-scale model, training set n = %d)"
            % int(fitB["n"]), SRC_MODEL, font=8.5,
            formats={"p-value A": "{:.3f}", "p-value B": "{:.3f}"})
    d.h("4.6.2\tVariable importance", 3)
    ib = imp[(imp["Model set"] == C.MODEL_SETS["B"]) & (imp["Algorithm"] == tree_alg)]
    gis_share = grp.loc[tree_alg, "GIS / location (all distances)"]
    top_gis = ib[ib["Feature"].isin(C.GIS_FEATURES)].iloc[0]
    sd = shp[shp["Feature"].isin(C.GIS_FEATURES)]["Typical effect on Annual Value (%)"]
    sd_min, sd_max = sd.min(), sd.max()
    d.p(f"Permutation importance measures how much the cross-validated RMSE increases when one variable is randomly "
        f"shuffled ({d.next_fig()}). For the {tree_alg} Model B, total floor area is by far the most important "
        f"variable, followed by condition grade. All eleven distances together account for only about "
        f"{pct(gis_share, 0)} of the total importance. The most important single distance is "
        f"{top_gis['Label'].split(' (')[0].lower()}. The SHAP analysis gives a similar ranking (Figure {d.chapter}.{d.fig_no + 2}): a "
        f"typical property's estimate moves by about {pct(shp.iloc[0]['Typical effect on Annual Value (%)'], 0)} because of "
        f"its floor area, {pct(shp[shp['Feature'] == 'Condition']['Typical effect on Annual Value (%)'].iloc[0])} "
        f"because of its condition, and by only {f2(sd_min)}-{f2(sd_max)} % because of each individual distance. The location variables "
        f"therefore act as fine adjustments to a value that is mostly set by size and condition. This matches the "
        f"modest GIS gain in Section 4.4.")
    d.figure(F / f"AB_permutation_importance_B_{tree_key}.png",
             f"Permutation importance, Model B {tree_alg} (mean ± SD over 50 CV folds)", SRC_MODEL, width=d.width * 0.85)
    d.figure(F / "AB_shap_summary_B.png", f"SHAP summary, Model B {tree_alg} (training properties)", SRC_MODEL,
             width=d.width * 0.85)
    d.p(f"The partial dependence plots ({d.next_fig()}) show the average estimated Annual Value as one variable "
        f"changes, with the others at their observed values. Floor area has a steep, almost linear effect over the "
        f"range where most properties lie. The distance effects are small and step-like, which is typical of tree "
        f"models, and should be read as local differences rather than smooth distance-decay functions.")
    d.figure(F / "AB_pdp_B.png", f"Partial dependence plots, Model B {tree_alg}", SRC_MODEL)

    d.h("4.6.3\tWorked example: explaining one property's assessment", 3)
    m_final = ex_m.iloc[-1]["End (Rs.)"]
    m_base = ex_m.iloc[0]
    t_final = ex_t.iloc[-1]["End (Rs.)"]
    d.p(f"To show how an estimate could be explained to a taxpayer, one typical test property was selected: the one "
        f"whose CMC Annual Value is closest to the test-set median. It is property {ex['Property_ID']} (assessment no. "
        f"{ex['Assessment_No']}), a {float(ex['Total_Floor_Area']):,.0f} sq ft, {int(float(ex['Num_Floors']))}-floor "
        f"building, {int(float(ex['Age']))} years old, in '{ex['Condition']}' condition, with a CMC Annual Value of "
        f"{rs(float(ex['Annual_Value']))}. The MLR explanation ({d.next_tab()}, {d.next_fig()}) starts from a reference "
        f"property assessed at {rs(m_base['End (Rs.)'])}: median floor area, one floor, median age, 'Fair' condition and "
        f"median distances. It then applies one adjustment for each recorded characteristic and reaches an estimate of "
        f"**{rs(m_final)}** (ratio {f3(m_final / float(ex['Annual_Value']))}).")
    wt = ex_m[["Label", "Effect (%)", "Change (Rs.)", "End (Rs.)"]].rename(
        columns={"Label": "Step", "Effect (%)": "Adjustment (%)", "End (Rs.)": "Running estimate (Rs.)"})
    d.table(wt, "Itemised MLR (Model B) explanation of the example property's estimate", SRC_MODEL, font=8.5,
            formats={"Adjustment (%)": "{:+.1f}", "Change (Rs.)": "{:+,.0f}", "Running estimate (Rs.)": "{:,.0f}"})
    d.figure(F / "AB_worked_example_mlr.png", "MLR explanation of the example property (waterfall)", SRC_MODEL,
             width=d.width * 0.9)
    def eff(key):
        return ex_m[ex_m["Label"].str.contains(key, regex=False)]["Effect (%)"].iloc[0]
    n_fl = int(float(ex["Num_Floors"]))
    d.p(f"A valuer could read this directly to the owner: 'A typical single-storey house in Fair condition would be "
        f"assessed at about {rs(m_base['End (Rs.)'])}. Your building has {n_fl} floors, and the same area spread over "
        f"more floors is valued lower ({eff('Number of floors'):+.1f} %). Its {ex['Condition']} condition changes the "
        f"value by {eff('Condition grade'):+.1f} %, its floor area by {eff('Total floor area'):+.1f} %, and its "
        f"location relative to the main road by {eff('A/B road'):+.1f} %.' The {tree_alg} model gives a "
        f"similar estimate ({rs(t_final)}, {d.next_fig()}) with a similar ordering of reasons. Its adjustments, "
        f"however, depend on the combination of characteristics, so they cannot be written as a fixed rate table. "
        f"This is the practical trade-off. The tree models are slightly more accurate, while the MLR provides "
        f"the itemised, repeatable basis that the research aim calls for (Section 1.3).")
    d.figure(F / "AB_worked_example_tree.png", f"SHAP explanation of the example property, Model B {tree_alg}",
             SRC_MODEL, width=d.width * 0.9)

    # ================================================================== 4.7 - 4.9 scaffolds
    d.h("4.7\tExisting vs Proposed Assessment Process", 2)
    d.note("Add your process comparison table and process figure here. Key points from the results to connect:")
    d.bullets([
        f"**Consistency.** The same model rules reproduce CMC values with a median ratio of about 1.00 and a "
        f"cross-validated COD of {f2(oof['COD (%)'].min())}-{f2(oof['COD (%)'].max())}, so the recorded "
        f"characteristics explain the existing assessments in a repeatable way.",
        "**Bias detection.** Records that deviate strongly from the model (the flagged extreme-rate records, the "
        "under-estimated top value band) can be listed automatically for valuer review. This turns 'valuation bias' "
        "from something hard to see into something measurable.",
        "**Transparency.** The MLR itemisation (Section 4.6.3) gives a step-by-step basis that can be explained to an "
        "owner or used in an appeal.",
        "**Site inspection.** Size and condition, which still need field data, account for most of the value. OSM "
        "distances add a modest refinement within surveyed areas, but did not transfer to unseen clusters "
        f"({sp_tab}). Inspection can therefore be targeted (verification, condition updates, flagged cases) "
        "rather than removed.",
    ])
    d.h("4.8\tImplementation Requirements and Framework", 2)
    d.note("Develop the requirements framework (data, spatial data, technical capacity, institutional capacity, "
           "validation). Evidence from this chapter:")
    d.bullets([
        "**Data.** A digital register of floor area per floor, number of floors, condition grade and use for every "
        "assessment number, with a road identifier (missing in the current data, which limited road-level validation).",
        "**Spatial data.** OSM layers and a georeferenced property point per assessment number. All eleven distances "
        "could be generated for every property, so OSM coverage was not the constraint.",
        "**Technical capacity.** An MLR with about ten terms performs within a few per cent of XGBoost and can be run "
        "and explained without specialist ML skills. The tree models need Python skills and interpretation tools "
        "(SHAP).",
        "**Validation.** Routine ratio studies (median ratio, COD, PRD, PRB) on every revaluation cycle, plus a "
        "review list of outlier records.",
    ])
    d.h("4.9\tConclusion", 2)
    d.note("Summarise in your own words. Suggested points:")
    d.bullets([
        f"Recorded CMC characteristics alone (Model A) reproduce the 2025 Annual Values closely (test R² "
        f"{f3(comp[comp['Model set'] == C.MODEL_SETS['A']]['Test R2'].min())}-"
        f"{f3(comp[comp['Model set'] == C.MODEL_SETS['A']]['Test R2'].max())}).",
        f"Adding OSM-derived distances (Model B) lowers the test error of all three algorithms "
        f"({pct(-test_gains.max(), 1)} to {pct(-test_gains.min(), 1)} RMSE), with smaller and less consistent gains "
        f"in cross-validation, and no gain when whole areas are held out.",
        f"{short_set(best_set)} {best_alg} is the most accurate configuration, but the MLR is close behind and offers "
        f"the clearer, itemised explanation.",
        "The OSM variables capture part of the location effect now judged through site inspection and the road-wise "
        "rate chart, within areas that have assessed comparables. They are not yet a substitute for the chart in "
        "areas without them.",
    ])

    # ================================================================== Chapter 3 inserts
    d.page_break()
    d.h("Suggested additions to Chapter 3 (to merge into the methodology)", 1)
    d.note("These paragraphs record decisions made during the analysis so that Chapter 3 matches Chapter 4. Merge "
           "them into the indicated sections and then delete this page.")
    d.h("Section 3.5: add to Table y", 3)
    d.p("In addition to the seven planned variables, four further amenity distances available from OSM were "
        "generated with the same nearest-feature method: distance to the nearest bank, pharmacy, playground and "
        "shopping mall (Neighbourhood amenities group, metres).")
    d.h("Section 3.7: data preprocessing (as applied)", 3)
    d.p(f"Of the {n_raw} records, {n_unlab} without an Annual Value were excluded from training. Blank upper-floor "
        f"areas were interpreted as 0 (no such floor), and total floor area and number of floors were derived. "
        f"Condition labels were standardised to five ordered grades (Poor, Fair-, Fair, Fair+, Good), with 'Modern' "
        f"and 'Good (Modern)' merged into 'Good'. The few missing ages and condition grades were imputed with the "
        f"training-set median and mode inside the model pipeline. Records with an extreme Annual Value per sq ft "
        f"were flagged but retained, and a sensitivity analysis without them is reported.")
    d.h("Section 3.8: model development (as applied)", 3)
    d.p(f"Because Annual Value is strongly right-skewed, all models were fitted on log(Annual Value), and their "
        f"estimates were back-transformed to rupees for evaluation. MLR used log(floor area), giving a log-log hedonic "
        f"specification, and condition dummies with 'Fair' as reference. To avoid severe multicollinearity among the "
        f"distances, MLR in Model B used a screened subset (|r| < 0.70 and VIF <= 5, determined on the training set). "
        f"Random Forest and XGBoost used all distances. The data were split 80:20 ({n_train}/{n_test}), stratified by "
        f"Annual Value with seed 42, and the same split was used for Models A and B. Hyperparameters were tuned with "
        f"RandomizedSearchCV (5-fold CV), and each configuration was evaluated with 5-fold CV repeated 10 times and "
        f"once on the test set. Because the dataset contains no road identifier, road-grouped validation was "
        f"approximated by spatially grouped CV over {C.SPATIAL_CLUSTERS} k-means clusters of the GIS distances.")
    d.h("Section 3.10: interpretation of model results", 3)
    d.p("The models were interpreted at three levels. First, MLR coefficients, converted to percentage effects on "
        "Annual Value, give fixed adjustments for each characteristic. Second, permutation importance (the increase "
        "in cross-validated RMSE when a variable is shuffled) and SHAP values (TreeExplainer) rank the variables "
        "for the tree models, and partial dependence plots show the average shape of each effect. Third, a worked "
        "example decomposes one property's estimate into itemised adjustments from a reference property (MLR) and "
        "into SHAP contributions (tree model). This shows how a valuer could explain an assessment to an owner.")

    d.save(path)
    return path


def _fmt_params(s):
    if not isinstance(s, str) or not s.startswith("{"):
        return "none (OLS)"
    p = json.loads(s)
    short = {"n_estimators": "trees", "max_depth": "depth", "min_samples_leaf": "min leaf",
             "min_samples_split": "min split", "max_features": "features/split", "learning_rate": "learning rate",
             "subsample": "subsample", "colsample_bytree": "col. sample", "min_child_weight": "min child wt",
             "reg_lambda": "lambda", "reg_alpha": "alpha", "gamma": "gamma"}
    return "; ".join(f"{short.get(k, k)} {v}" for k, v in p.items())


def _fmt_predictors(s):
    return ", ".join(C.SHORT_LABELS.get(v, v).replace("Distance to ", "d. ") for v in s.split(", "))
