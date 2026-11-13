"""Central configuration for the final Dematagoda Ward analysis.

Every definition of *what* is modelled lives here, so the three notebooks (Model A, Model B and
the comparison) use exactly the same column names, feature sets and settings.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "Dematagoda property details.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORT_DIR = PROJECT_ROOT / "report"
SPLIT_FILE = PROCESSED_DIR / "split_ids.csv"

STUDY_WARD = "Dematagoda Ward"
STUDY_AUTHORITY = "Colombo Municipal Council (CMC)"

# ----------------------------------------------------------------------------
# Reproducibility and validation design (research document, section 3.8)
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5            # five-fold CV inside the training set
CV_REPEATS = 10          # repeated 10 times for stable estimates (same folds for every model)
RF_SEARCH_ITER = 40
XGB_SEARCH_ITER = 60
PERMUTATION_REPEATS = 10
SPATIAL_CLUSTERS = 10    # supplementary spatially grouped CV (no road field in the dataset)

# ----------------------------------------------------------------------------
# Column mapping: original CSV header -> (standard name, meaning, group, unit)
# ----------------------------------------------------------------------------
COLUMN_MAP = {
    "fid": ("FID", "GIS layer feature identifier (QGIS export)", "Identifier", ""),
    "field_id": ("Field_ID", "Sequential record identifier", "Identifier", ""),
    "Assessment No": ("Assessment_No", "CMC assessment number", "Identifier", ""),
    "Unique Property ID": ("Property_ID", "Unique property identifier (DEM_xxxxx)", "Identifier", ""),
    "Building Age": ("Age", "Building age recorded by CMC", "Property", "years"),
    "Condition_": ("Condition", "Building condition grade recorded by CMC", "Property", "grade"),
    "Ground Floor Area": ("Ground_Floor_Area", "Ground floor area", "Property", "sq ft"),
    "First Floor Area": ("First_Floor_Area", "First floor area", "Property", "sq ft"),
    "Second Floor Area": ("Second_Floor_Area", "Second floor area", "Property", "sq ft"),
    "Third Floor Area": ("Third_Floor_Area", "Third floor area", "Property", "sq ft"),
    "Annual val": ("Annual_Value", "CMC-recorded Annual Value (2025 assessment basis, not market value)", "Target", "Rs."),
    "Dis_AB Road": ("Dist_AB_Road", "Distance to nearest A/B-class road", "Road accessibility", "m"),
    "Dis_Busstops": ("Dist_Bus_Stop", "Distance to nearest bus stop", "Public transport", "m"),
    "Dis_Railwaystations": ("Dist_Railway", "Distance to nearest railway station", "Public transport", "m"),
    "Dis_CBD": ("Dist_CBD", "Distance to the Colombo CBD", "Centrality", "m"),
    "Dis_Schools": ("Dist_School", "Distance to nearest school", "Neighbourhood amenities", "m"),
    "Dis_Hospitals": ("Dist_Hospital", "Distance to nearest hospital", "Neighbourhood amenities", "m"),
    "Dis_Supermarket": ("Dist_Supermarket", "Distance to nearest supermarket", "Neighbourhood amenities", "m"),
    "Dis_Banks": ("Dist_Bank", "Distance to nearest bank", "Neighbourhood amenities", "m"),
    "Dis_Pharmacy": ("Dist_Pharmacy", "Distance to nearest pharmacy", "Neighbourhood amenities", "m"),
    "Dis_Playground": ("Dist_Playground", "Distance to nearest playground", "Neighbourhood amenities", "m"),
    "Dis_Shoppingmalls": ("Dist_Shopping_Mall", "Distance to nearest shopping mall", "Neighbourhood amenities", "m"),
}
RENAME = {orig: v[0] for orig, v in COLUMN_MAP.items()}

TARGET = "Annual_Value"
FLOOR_COLUMNS = ["Ground_Floor_Area", "First_Floor_Area", "Second_Floor_Area", "Third_Floor_Area"]
UPPER_FLOOR_COLUMNS = FLOOR_COLUMNS[1:]

# The 7 variables of Table y (section 3.5) + 4 additional OSM amenity distances in the final dataset
GIS_FEATURES = ["Dist_AB_Road", "Dist_Bus_Stop", "Dist_Railway", "Dist_CBD",
                "Dist_School", "Dist_Hospital", "Dist_Supermarket",
                "Dist_Bank", "Dist_Pharmacy", "Dist_Playground", "Dist_Shopping_Mall"]
GIS_IN_METHODOLOGY = GIS_FEATURES[:7]
GIS_GROUPS = {f: COLUMN_MAP[k][2] for k, (f, *_rest) in COLUMN_MAP.items() if f in GIS_FEATURES}

# ----------------------------------------------------------------------------
# Predictor sets (section 3.8)
# ----------------------------------------------------------------------------
PROPERTY_FEATURES = ["Age", "Condition", "Total_Floor_Area", "Num_Floors"]
# MLR uses log(floor area): with a log target this is the standard log-log hedonic form
# (Annual Value ~ rate x area), and the coefficient is an elasticity. Tree models are unaffected
# by monotone transforms, so they use floor area in sq ft.
PROPERTY_FEATURES_MLR = ["Age", "Condition", "Log_Total_Floor_Area", "Num_Floors"]
CATEGORICAL_FEATURES = ["Condition"]
MODEL_SETS = {
    "A": "Model A - property only",
    "B": "Model B - property + GIS",
}
MODEL_NAMES = {"mlr": "Multiple Linear Regression", "rf": "Random Forest", "xgb": "XGBoost"}
MODEL_SHORT = {"mlr": "MLR", "rf": "RF", "xgb": "XGBoost"}
ALGORITHMS = ["mlr", "rf", "xgb"]

# ----------------------------------------------------------------------------
# Condition grades
# ----------------------------------------------------------------------------
# keys produced by data.condition_key(): lower case, spaces removed
CONDITION_MAP = {
    "poor": "Poor",
    "fair-": "Fair-",
    "fair": "Fair",
    "fair+": "Fair+",
    "good": "Good",
    "good(modern)": "Good",   # n = 2; merged into Good (decision 11 Oct 2026, kept for final model)
    "modern": "Good",         # n = 2; same grade label without the 'Good' prefix
}
CONDITION_ORDER = ["Poor", "Fair-", "Fair", "Fair+", "Good"]
CONDITION_REFERENCE = "Fair"   # MLR reference category (common, mid-scale grade)

# ----------------------------------------------------------------------------
# Thresholds
# ----------------------------------------------------------------------------
CORR_THRESHOLD = 0.70
VIF_THRESHOLD = 5.0
MISSING_EXCLUSION_THRESHOLD = 0.30
# IAAO Standard on Ratio Studies reference ranges (residential)
IAAO_MEDIAN_RATIO = (0.90, 1.10)
IAAO_COD_MAX = 15.0
IAAO_PRD = (0.98, 1.03)
IAAO_PRB = (-0.05, 0.05)

FEATURE_LABELS = {
    "Age": "Building age (years)",
    "Condition": "Condition grade",
    "Total_Floor_Area": "Total floor area (sq ft)",
    "Num_Floors": "Number of floors",
    "Log_Total_Floor_Area": "log(Total floor area)",
    "Ground_Floor_Area": "Ground floor area (sq ft)",
    "First_Floor_Area": "First floor area (sq ft)",
    "Second_Floor_Area": "Second floor area (sq ft)",
    "Third_Floor_Area": "Third floor area (sq ft)",
    "Dist_AB_Road": "Distance to A/B road (m)",
    "Dist_Bus_Stop": "Distance to bus stop (m)",
    "Dist_Railway": "Distance to railway station (m)",
    "Dist_CBD": "Distance to CBD (m)",
    "Dist_School": "Distance to school (m)",
    "Dist_Hospital": "Distance to hospital (m)",
    "Dist_Supermarket": "Distance to supermarket (m)",
    "Dist_Bank": "Distance to bank (m)",
    "Dist_Pharmacy": "Distance to pharmacy (m)",
    "Dist_Playground": "Distance to playground (m)",
    "Dist_Shopping_Mall": "Distance to shopping mall (m)",
    "Annual_Value": "CMC Annual Value (Rs.)",
    "AV_per_sqft": "Annual Value per sq ft (Rs.)",
}
SHORT_LABELS = {k: v.split(" (")[0] for k, v in FEATURE_LABELS.items()}

TARGET_DISCLAIMER = ("The models estimate the CMC-recorded Annual Value (the existing assessment basis). "
                     "They do not estimate independent market value.")


def label(name: str) -> str:
    """Readable label, also for one-hot columns such as 'Condition_Good'."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name.startswith("Condition_"):
        return f"Condition: {name.split('_', 1)[1]} (vs {CONDITION_REFERENCE})"
    return name


def ensure_dirs():
    for d in [PROCESSED_DIR, TABLES_DIR, FIGURES_DIR, PREDICTIONS_DIR, MODELS_DIR, REPORT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
