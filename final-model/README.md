# Final analysis: ML-based residential property tax assessment, Dematagoda Ward (CMC)

The final thesis analysis, run on the full Dematagoda dataset: 566 properties, of which 502 have a CMC Annual Value.

It compares two predictor sets, as set out in methodology section 3.8:

| Set | Predictors | Notebook |
|---|---|---|
| **Model A**, property only | building age, condition grade, total floor area, number of floors | `notebooks/01_Model_A_property_only.ipynb` |
| **Model B**, property + GIS | Model A + 11 OSM/QGIS distances (AB road, bus stop, railway, CBD, school, hospital, supermarket, bank, pharmacy, playground, shopping mall) | `notebooks/02_Model_B_property_plus_GIS.ipynb` |

Each set is run with three algorithms: Multiple Linear Regression, Random Forest and XGBoost. That gives six configurations, which are compared in `notebooks/03_Model_A_vs_B_comparison_and_interpretation.ipynb`. Notebook 03 also covers:
- the ratio study (median ratio, COD, PRD, PRB);
- interpretation (coefficients, permutation importance, SHAP, PDP, a worked example);
- supplementary checks;
- writing the Chapter 4 draft, `report/Chapter4_draft.docx`.

The target is the **CMC-recorded Annual Value**, not market value.

## Key design decisions
- **Same test set for A and B.** There is one fixed 80:20 split (seed 42, stratified by Annual Value), saved in `data/processed/split_ids.csv`. Tuning (RandomizedSearchCV, 5-fold) and the 5-fold CV repeated 10 times use the training set only. The test set is used once.
- **Log target.** Every model is fitted on log(Annual Value). Predictions are converted back to rupees, and all metrics are in rupees. MLR uses log(floor area), the log-log hedonic form; its coefficient is an elasticity. A raw-target sensitivity run is in notebook 03.
- **MLR in Model B** uses a VIF/correlation-screened subset of the distances, because the 11 distances are strongly inter-correlated. RF and XGBoost use all 11.
- **Cleaning rules:**
  - A blank upper-floor area counts as 0 (no such floor).
  - `Fair +` is read as `Fair+`.
  - `Modern` and `Good (Modern)` are merged into `Good`.
  - Records without an Annual Value are excluded from training.
  - Missing age or condition values are imputed inside each pipeline, using training data only.
  - Records with extreme Annual Value per sq ft are flagged and kept, with a sensitivity check that excludes them.

## Setup and run
```bash
cd final-model
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # macOS: brew install libomp (for XGBoost)
python -m ipykernel install --user --name cmc-final --display-name "Python (CMC final model)"

python scripts/build_notebooks.py        # (re)generate the notebooks
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_Model_A_property_only.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_Model_B_property_plus_GIS.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_Model_A_vs_B_comparison_and_interpretation.ipynb
cd .. && python scripts/check_report.py  # verifies the numbers in the Word draft against the output tables
```
The notebooks must run in order, because notebook 03 loads the saved results of 01 and 02. The full run takes about 15 minutes.

## Folders
```
data/raw/            original CSV (never modified)
data/processed/      cleaned_properties.csv, split_ids.csv, removed_records.csv, unlabelled_properties.csv
src/                 shared code (config, data, modelling, runner, interpret, plots, report)
notebooks/           01 Model A, 02 Model B, 03 comparison + interpretation
outputs/tables/      A_*, B_*, AB_* result tables (CSV)
outputs/figures/     thesis figures (PNG, 200 dpi)
outputs/models/      fitted models (joblib)
outputs/predictions/ test-set predictions per model set
report/              Chapter4_draft.docx
```
