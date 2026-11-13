"""Checks that every number in the prose of report/Chapter4_draft.docx can be traced to the output
tables (outputs/tables/*.csv, data/processed/*.csv), at the precision shown in the text.

Numbers that are design constants (thresholds, seeds, split sizes, section numbers) are whitelisted.
Run from the final-model folder:  python scripts/check_report.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "report" / "Chapter4_draft.docx"
CONSTANTS = {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 15, 20, 30, 40, 42, 50, 55, 60, 80, 90, 100, 2025,
             0.1, 0.7, 0.70, 0.9, 0.90, 1.0, 1.03, 1.10, 0.98, 0.05, 1.00}

# --- every numeric value available in the tables -----------------------------------------
values = []
for f in list((ROOT / "outputs" / "tables").glob("*.csv")) + list((ROOT / "outputs" / "predictions").glob("*.csv")) + \
        [ROOT / "data" / "processed" / "cleaned_properties.csv"]:
    df = pd.read_csv(f)
    for c in df.columns:
        v = pd.to_numeric(df[c], errors="coerce").dropna().values
        values.append(v)
    values.append(np.array([len(df)], float))
vals = np.concatenate(values)
with np.errstate(all="ignore"):
    vals = np.concatenate([vals, -vals, 100 * vals, -100 * vals, 100 * (np.exp(vals) - 1), 100 * (1 - vals),
                           100 * (vals - 1), 100 * (1.1 ** vals - 1), vals.astype(float)])
vals = vals[np.isfinite(vals)]


def traceable(token: str) -> bool:
    x = float(token.replace(",", ""))
    dec = len(token.split(".")[1]) if "." in token else 0
    if x in CONSTANTS:
        return True
    tol = 0.5 * 10 ** (-dec) + 1e-9
    return bool(np.any(np.abs(vals - x) <= tol) or np.any(np.abs(np.abs(vals) - abs(x)) <= tol))


doc = Document(DOCX)
prose = "\n".join(p.text for p in doc.paragraphs
                  if not p.text.startswith(("Figure", "Table", "Source", "[Author note")) and not p.style.name.startswith("Heading"))
prose = re.sub(r"(Section|Figure|Table|Chapter)\s+\d+(\.\d+)*[a-c]?", "", prose)   # cross-references
prose = re.sub(r"DEM_\d+|no\. [\d/A-Za-z]+", "", prose)                            # identifiers
tokens = re.findall(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?", prose)
missing = [tk for tk in tokens if not traceable(tk.lstrip("+"))]
print(f"Numbers checked in prose: {len(tokens)}; not traceable to an output table: {len(missing)}")
for tk in sorted(set(missing)):
    ctx = re.search(r".{0,50}" + re.escape(tk) + r".{0,30}", prose)
    print(f"  {tk!r:>12}  ...{ctx.group(0) if ctx else ''}...")
n_tables = len(doc.tables)
n_figs = sum(1 for p in doc.paragraphs if p.text.startswith("Figure 4."))
print(f"Tables in document: {n_tables}; figure captions: {n_figs}")
sys.exit(1 if missing else 0)
