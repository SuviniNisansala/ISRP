"""Loading, documented cleaning, feature engineering and the fixed train/test split.

Principles (research document, sections 3.6-3.7)
-------------------------------------------------
* The raw CSV is never modified; every change is logged.
* Records without a CMC Annual Value are excluded from training (target unavailable) but kept
  in a separate unlabelled table.
* Missing predictors of labelled records are NOT imputed here; median / mode imputation happens
  inside each model pipeline, fitted on training rows only (no leakage).
* Blank upper-floor areas are read as "floor does not exist" (= 0), the rule adopted for the
  interim prototype (assumption to be confirmed with CMC).
* The 80:20 split is created once, saved to data/processed/split_ids.csv and re-used by Model A
  and Model B, so both are tested on exactly the same properties.
"""
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C

MISSING_TOKENS = {"", "NA", "N/A", "NAN", "NULL", "NONE", "-"}


def condition_key(s: pd.Series) -> pd.Series:
    return s.str.lower().str.replace(r"\s+", "", regex=True)


def load_raw(path=C.RAW_CSV) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def variable_table() -> pd.DataFrame:
    return pd.DataFrame([{"Original column": k, "Variable": v[0], "Meaning": v[1], "Group": v[2], "Unit": v[3]}
                         for k, v in C.COLUMN_MAP.items()])


def raw_missing_table(raw: pd.DataFrame) -> pd.DataFrame:
    std = raw.rename(columns=C.RENAME)
    rows = []
    for col in std.columns:
        s = std[col].astype(str).str.strip().str.upper()
        miss = s.isin(MISSING_TOKENS)
        rows.append({"Variable": col, "Records": len(s), "Missing": int(miss.sum()),
                     "Missing (%)": round(100 * miss.mean(), 1),
                     "Literal 'NA' entries": int((s == "NA").sum())})
    return pd.DataFrame(rows)


def clean(raw: pd.DataFrame) -> dict:
    df = raw.rename(columns=C.RENAME).copy()
    log = [f"Raw file: {len(df)} records x {df.shape[1]} columns."]

    # 1. whitespace + missing tokens --------------------------------------
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    n_na = {c: int((df[c].str.upper() == "NA").sum()) for c in df.columns}
    for c, n in n_na.items():
        if n:
            log.append(f"'{c}': {n} literal 'NA' entr{'y' if n == 1 else 'ies'} treated as missing.")
    df = df.mask(df.apply(lambda s: s.str.upper().isin(MISSING_TOKENS)))

    # 2. numeric coercion ---------------------------------------------------
    numeric = ["FID", "Field_ID", "Age", C.TARGET] + C.FLOOR_COLUMNS + C.GIS_FEATURES
    malformed = []
    for col in numeric:
        present = df[col].notna()
        num = pd.to_numeric(df[col].str.replace(",", "", regex=False), errors="coerce")
        for idx in df.index[present & num.isna()]:
            malformed.append({"Property_ID": df.at[idx, "Property_ID"], "Column": col, "Raw value": df.at[idx, col]})
        df[col] = num
    log.append(f"Numeric coercion: {len(malformed)} malformed numeric value(s) set to missing.")

    # 3. condition ----------------------------------------------------------
    df["Condition_Raw"] = df["Condition"]
    key = condition_key(df["Condition"])
    mapped = key.map(C.CONDITION_MAP)
    unmapped = key.notna() & mapped.isna()
    if unmapped.any():
        raise ValueError(f"Unmapped condition values: {sorted(df.loc[unmapped, 'Condition'].unique())}")
    df["Condition"] = mapped
    cond_table = (df.assign(Raw=df["Condition_Raw"].fillna("(blank)"), Clean=df["Condition"].fillna("(missing)"))
                  .groupby(["Raw", "Clean"]).size().reset_index(name="Records")
                  .rename(columns={"Raw": "Raw value", "Clean": "Cleaned grade"}))
    log.append("Condition: 'Fair +' -> 'Fair+'; 'Modern' and 'Good (Modern)' (n=4) merged into 'Good'; "
               "grade order Poor < Fair- < Fair < Fair+ < Good.")

    # 4. floors -------------------------------------------------------------
    has_gf = df["Ground_Floor_Area"].notna()
    for col in C.UPPER_FLOOR_COLUMNS:
        fill = has_gf & df[col].isna()
        df.loc[fill, col] = 0.0
        log.append(f"'{col}': {int(fill.sum())} blank value(s) set to 0 (no such floor).")
    df["Total_Floor_Area"] = df[C.FLOOR_COLUMNS].sum(axis=1, min_count=1).where(has_gf)
    df["Num_Floors"] = (df[C.FLOOR_COLUMNS].fillna(0) > 0).sum(axis=1).where(has_gf)
    df["Log_Total_Floor_Area"] = np.log(df["Total_Floor_Area"].where(df["Total_Floor_Area"] > 0))

    # 5. invalid values -----------------------------------------------------
    invalid = []
    rules = {"Age": lambda s: (s < 0) | (s > 150), C.TARGET: lambda s: s <= 0,
             "Total_Floor_Area": lambda s: s <= 0, **{g: (lambda s: s < 0) for g in C.GIS_FEATURES}}
    for col, rule in rules.items():
        bad = df[col].notna() & rule(df[col])
        for idx in df.index[bad]:
            invalid.append({"Property_ID": df.at[idx, "Property_ID"], "Column": col, "Value": df.at[idx, col]})
            df.at[idx, col] = np.nan
    log.append(f"Invalid values (outside valid range): {len(invalid)} set to missing.")

    # 6. duplicates -----------------------------------------------------------
    dup_review, drop_ids = review_duplicates(df)
    df["Exact_Duplicate"] = df["Property_ID"].isin(drop_ids).astype(int)
    log.append(f"Duplicates: {df['Assessment_No'].duplicated(keep=False).sum()} records share an assessment number "
               f"with another record; {len(drop_ids)} exact duplicate(s) (all attributes and distances identical) removed.")

    # 7. target / usable rows -----------------------------------------------
    removed = []
    keep = df["Exact_Duplicate"] == 0
    for idx in df.index[~keep]:
        removed.append((idx, "Exact duplicate of another record"))
    no_target = keep & df[C.TARGET].isna()
    for idx in df.index[no_target]:
        removed.append((idx, "No Annual Value (target unavailable)"))
    no_area = keep & df[C.TARGET].notna() & df["Total_Floor_Area"].isna()
    for idx in df.index[no_area]:
        removed.append((idx, "Annual Value present but no floor area recorded (core size variable missing)"))
    rem = df.loc[[i for i, _ in removed]].copy()
    rem.insert(0, "Removal_Reason", [r for _, r in removed])

    labelled = df[keep & df[C.TARGET].notna() & df["Total_Floor_Area"].notna()].copy()
    unlabelled = df[keep & df[C.TARGET].isna()].copy()
    labelled["AV_per_sqft"] = labelled[C.TARGET] / labelled["Total_Floor_Area"]
    log.append(f"Records without Annual Value: {int(no_target.sum())} (kept separately as unlabelled).")
    log.append(f"Labelled records without floor area: {int(no_area.sum())} (excluded).")
    log.append(f"Modelling dataset: {len(labelled)} labelled records.")

    accounting = pd.DataFrame([
        ("Records in raw file", len(df)),
        ("Exact duplicates removed", int((~keep).sum())),
        ("No Annual Value (unlabelled)", int(no_target.sum())),
        ("Annual Value but no floor area", int(no_area.sum())),
        ("Records used for modelling", len(labelled)),
    ], columns=["Step", "Records"])
    assert accounting["Records"].iloc[0] - accounting["Records"].iloc[1:4].sum() == len(labelled)

    return {"all": df, "labelled": labelled, "unlabelled": unlabelled, "removed": rem, "log": log,
            "accounting": accounting, "condition_table": cond_table, "duplicate_review": dup_review,
            "malformed": pd.DataFrame(malformed), "invalid": pd.DataFrame(invalid)}


def review_duplicates(df: pd.DataFrame):
    """Classify repeated assessment numbers. Only exact copies (every attribute and every distance
    identical) are removed; the rest are distinct units / records and are retained."""
    attr = ["Assessment_No", "Age", "Condition", C.TARGET] + C.FLOOR_COLUMNS + C.GIS_FEATURES
    exact_later = df.duplicated(subset=attr, keep="first")
    same_no = df["Assessment_No"].duplicated(keep=False)
    rows = []
    for idx in df.index[same_no]:
        r = df.loc[idx]
        grp = df[df["Assessment_No"] == r["Assessment_No"]]
        identical_loc = grp[C.GIS_FEATURES].round(3).drop_duplicates().shape[0] == 1
        if exact_later[idx]:
            cls, dec = "Exact duplicate record", "Removed (later copy)"
        elif df.duplicated(subset=attr, keep=False)[idx]:
            cls, dec = "Exact duplicate record", "Retained (first occurrence)"
        else:
            if identical_loc:
                cls, dec = "Same number, same location, different attributes", "Retained (separate units; verify with CMC)"
            else:
                cls = "Same number at a different location (number re-used on another road)"
                dec = "Retained (distinct property)"
        rows.append({"Property_ID": r["Property_ID"], "Assessment_No": r["Assessment_No"],
                     "Records with this number": len(grp), "Annual_Value": r[C.TARGET],
                     "Total_Floor_Area": r.get("Total_Floor_Area", np.nan),
                     "Classification": cls, "Decision": dec})
    return pd.DataFrame(rows), list(df.loc[exact_later, "Property_ID"])


def base_number(no) -> str:
    m = re.match(r"^\s*(\d+)", str(no))
    return m.group(1) if m else str(no)


def flag_unusual(labelled: pd.DataFrame) -> pd.DataFrame:
    """Flag (never delete) extreme Annual Value per sq ft (outside Q1-3IQR / Q3+3IQR on the log scale)."""
    d = labelled.copy()
    lr = np.log(d["AV_per_sqft"])
    q1, q3 = lr.quantile([0.25, 0.75])
    iqr = q3 - q1
    d["Flag_Extreme_AV_per_sqft"] = ((lr < q1 - 3 * iqr) | (lr > q3 + 3 * iqr)).astype(int)
    return d


# ----------------------------------------------------------------------------
# Fixed train / test split shared by Model A and Model B
# ----------------------------------------------------------------------------
def make_or_load_split(labelled: pd.DataFrame, force=False) -> pd.DataFrame:
    """80:20 split stratified by Annual Value quintile (seed 42), saved by Property_ID."""
    if C.SPLIT_FILE.exists() and not force:
        split = pd.read_csv(C.SPLIT_FILE)
        if set(split["Property_ID"]) == set(labelled["Property_ID"]):
            return split
    strata = pd.qcut(labelled[C.TARGET].rank(method="first"), 5, labels=False)
    tr, te = train_test_split(labelled["Property_ID"], test_size=C.TEST_SIZE,
                              random_state=C.RANDOM_STATE, stratify=strata)
    split = pd.concat([pd.DataFrame({"Property_ID": tr, "Partition": "train"}),
                       pd.DataFrame({"Property_ID": te, "Partition": "test"})]).sort_values("Property_ID")
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    split.to_csv(C.SPLIT_FILE, index=False)
    return split


def load_and_prepare(verbose=True) -> dict:
    """One call used by every notebook: clean data + the fixed train / test partition."""
    C.ensure_dirs()
    raw = load_raw()
    res = clean(raw)
    labelled = flag_unusual(res["labelled"])
    split = make_or_load_split(labelled)
    part = labelled["Property_ID"].map(split.set_index("Property_ID")["Partition"])
    labelled["Partition"] = part.values
    train = labelled[labelled["Partition"] == "train"].reset_index(drop=True)
    test = labelled[labelled["Partition"] == "test"].reset_index(drop=True)
    assert set(train["Property_ID"]).isdisjoint(test["Property_ID"])

    keep_cols = (["Property_ID", "FID", "Assessment_No", "Age", "Condition", "Condition_Raw"] + C.FLOOR_COLUMNS +
                 ["Total_Floor_Area", "Log_Total_Floor_Area", "Num_Floors"] + C.GIS_FEATURES + [C.TARGET, "AV_per_sqft",
                 "Flag_Extreme_AV_per_sqft", "Partition"])
    labelled[keep_cols].to_csv(C.PROCESSED_DIR / "cleaned_properties.csv", index=False)
    res["removed"].to_csv(C.PROCESSED_DIR / "removed_records.csv", index=False)
    res["unlabelled"].to_csv(C.PROCESSED_DIR / "unlabelled_properties.csv", index=False)
    if verbose:
        for line in res["log"]:
            print(" -", line)
        print(f" - Train / test: {len(train)} / {len(test)} (fixed split saved to {C.SPLIT_FILE.name})")
    return {**res, "raw": raw, "labelled": labelled, "train": train, "test": test, "split": split}
