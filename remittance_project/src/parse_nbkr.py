"""Parser for NBKR 'cross-border money transfers of individuals' tables (RU/EN/KG page versions)."""
import os
import re
import shutil
import tempfile

import openpyxl
import pandas as pd
import xlrd

ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
         "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12}
MONTH_RE = re.compile(r"^\s*([IVX]+)\s*['’`]\s*(\d{2})\s*(\*?)\s*$")

LABELS = {
    "Сальдо": "balance", "Net inflow": "balance",
    "Приток": "inflow", "Inflow": "inflow",
    "Отток": "outflow", "Outflow": "outflow",
    "Страны СНГ": "cis", "CIS countries": "cis",
    "Страны дальнего зарубежья": "far", "Foreign countries": "far",
    "Россия": "russia", "Russia": "russia",
}
STOP_PREFIXES = ("Примечан", "Источник", "Note", "Source")


def _load_rows(path):
    try:
        book = xlrd.open_workbook(path)
        sheet = book.sheet_by_index(0)
        return [[sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)]
    except xlrd.XLRDError:
        # some vintages are .xlsx saved with an .xls extension
        tmp = os.path.join(tempfile.gettempdir(), "_nbkr_parse_tmp.xlsx")
        shutil.copyfile(path, tmp)
        book = openpyxl.load_workbook(tmp, data_only=True)
        return [list(row) for row in book.worksheets[0].iter_rows(values_only=True)]


def _num(v):
    return float(v) if isinstance(v, (int, float)) and v == v else None


def parse_vintage(path):
    """Return (DataFrame indexed by month Timestamp with columns balance/inflow/russia/outflow, meta dict)."""
    rows = _load_rows(path)
    header_idx = next(i for i, r in enumerate(rows[:12])
                      if sum(1 for c in r if isinstance(c, str) and MONTH_RE.match(c)) >= 6)
    # Month columns must run consecutively (annual totals sit in between as numbers).
    # NBKR headers occasionally contain typos (e.g. Sep-2020 labelled "IX'19"), so a label
    # that breaks the sequence is replaced by the expected month and reported.
    cols, preliminary, header_fixes = {}, [], []
    prev = None
    for j, c in enumerate(rows[header_idx]):
        m = MONTH_RE.match(c) if isinstance(c, str) else None
        if not (m and m.group(1) in ROMAN):
            continue
        month = pd.Timestamp(2000 + int(m.group(2)), ROMAN[m.group(1)], 1)
        if prev is not None and month != prev + pd.DateOffset(months=1):
            expected = prev + pd.DateOffset(months=1)
            header_fixes.append(f"col {j}: '{c}' -> {expected:%Y-%m}")
            month = expected
        cols[j] = month
        prev = month
        if m.group(3):
            preliminary.append(month)

    block, series = None, {}
    body = rows[header_idx + 1:]
    for r in body:
        label = r[0] if isinstance(r[0], str) and r[0].strip() else (r[1] if len(r) > 1 and isinstance(r[1], str) else None)
        if not label:
            continue
        label = label.strip()
        if label.startswith(STOP_PREFIXES):
            break
        key = LABELS.get(label)
        if key in ("balance", "inflow", "outflow"):
            block = key
            series[key] = {d: _num(r[j]) for j, d in cols.items() if j < len(r)}
        elif key in ("cis", "far", "russia") and block in ("inflow", "outflow"):
            series[f"{block}_{key}"] = {d: _num(r[j]) for j, d in cols.items() if j < len(r)}

    parse_mode = "labels"
    if "inflow" not in series:
        # Kyrgyz-language vintages: fixed row order balance, inflow, CIS, Kazakhstan, Russia
        numeric = [r for r in body if any(_num(v) is not None for v in r[1:6])]
        for key, pos in (("balance", 0), ("inflow", 1), ("inflow_cis", 2), ("inflow_russia", 4)):
            series[key] = {d: _num(numeric[pos][j]) for j, d in cols.items() if j < len(numeric[pos])}
        parse_mode = "positional"

    df = pd.DataFrame(series).sort_index()
    df = df.rename(columns={"inflow_russia": "russia"})
    df = df[[c for c in ("balance", "inflow", "russia", "outflow", "inflow_cis", "inflow_far") if c in df.columns]]
    df = df.dropna(subset=["inflow"])
    title = " ".join(str(c) for c in rows[0] if isinstance(c, str)).strip()
    meta = {"title": title, "parse_mode": parse_mode, "preliminary_months": preliminary,
            "header_fixes": header_fixes}
    return df, meta


def validate(df):
    """Structural checks; returns list of problems (empty if clean)."""
    problems = []
    expected = pd.date_range(df.index.min(), df.index.max(), freq="MS")
    missing = expected.difference(df.index)
    if len(missing):
        problems.append(f"{len(missing)} missing months, e.g. {missing[0].date()}")
    if {"inflow_cis", "inflow_far"} <= set(df.columns):
        gap = (df["inflow"] - df["inflow_cis"] - df["inflow_far"]).abs()
        if (gap > 0.05).any():
            problems.append(f"inflow != CIS + far abroad (max gap {gap.max():.3f})")
    if (df["inflow"] < 0).any():
        problems.append("negative inflow")
    return problems
