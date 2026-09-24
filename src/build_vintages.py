"""Parse every archived NBKR vintage into one long table plus a vintage index."""
import glob
import os

import pandas as pd

from parse_nbkr import parse_vintage, validate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "nbkr_vintages")
OUT = os.path.join(ROOT, "data", "processed")
LANG_PRIORITY = {"ru": 0, "en": 1, "kg": 2}
# June 2022 is 317.2 in money-transfer-system vintages and 349.4 after the 2024 redefinition
REGIME_THRESHOLD_JUN22 = 333.0


def regime_of(df):
    jun22 = df["inflow"].get(pd.Timestamp("2022-06-01"))
    if jun22 is None:
        return "mts"
    return "cross_border" if jun22 > REGIME_THRESHOLD_JUN22 else "mts"


def main():
    parsed = []
    for path in sorted(glob.glob(os.path.join(RAW, "*.xls"))):
        date_str, doc_id, lang = os.path.basename(path)[:-4].split("_")
        df, meta = parse_vintage(path)
        problems = validate(df)
        parsed.append(dict(upload=pd.Timestamp(date_str), doc_id=doc_id, lang=lang, df=df, meta=meta, problems=problems))

    # same upload date published in several languages: check they agree, keep the RU one
    by_date = {}
    for p in parsed:
        by_date.setdefault(p["upload"], []).append(p)
    conflicts = []
    chosen = []
    for date, group in sorted(by_date.items()):
        group.sort(key=lambda p: LANG_PRIORITY[p["lang"]])
        base = group[0]
        for other in group[1:]:
            common = base["df"].index.intersection(other["df"].index)
            diff = (base["df"].loc[common, "inflow"] - other["df"].loc[common, "inflow"]).abs().max()
            if diff > 1e-6:
                conflicts.append((date.date(), base["lang"], other["lang"], round(diff, 4)))
        chosen.append(base)

    long_rows, index_rows = [], []
    for p in chosen:
        df = p["df"]
        last = df.index.max()
        index_rows.append(dict(
            vintage_date=p["upload"].date(), doc_id=p["doc_id"], lang=p["lang"],
            url=f"https://www.nbkr.kg/DOC/{p['upload'].strftime('%d%m%Y')}/{p['doc_id']}.xls",
            first_month=df.index.min().strftime("%Y-%m"), last_month=last.strftime("%Y-%m"),
            n_months=len(df), lag_days=(p["upload"] - (last + pd.offsets.MonthEnd(0))).days,
            regime=regime_of(df), parse_mode=p["meta"]["parse_mode"],
            preliminary_months=";".join(m.strftime("%Y-%m") for m in p["meta"]["preliminary_months"]),
            header_fixes="; ".join(p["meta"]["header_fixes"]),
            problems="; ".join(p["problems"]), title=p["meta"]["title"][:200]))
        for month, row in df.iterrows():
            long_rows.append(dict(vintage_date=p["upload"].date(), ref_month=month.strftime("%Y-%m"),
                                  inflow=row["inflow"], russia=row.get("russia"),
                                  outflow=row.get("outflow"), balance=row.get("balance")))

    index = pd.DataFrame(index_rows)
    long = pd.DataFrame(long_rows)
    os.makedirs(OUT, exist_ok=True)
    index.to_csv(os.path.join(OUT, "vintage_index.csv"), index=False)
    long.to_csv(os.path.join(OUT, "vintages_long.csv"), index=False)

    print(f"files parsed: {len(parsed)}, distinct vintages: {len(chosen)}")
    print(f"language duplicates that disagree: {conflicts or 'none'}")
    bad = index[index.problems != ""]
    print(f"vintages with structural problems: {len(bad)}")
    if len(bad):
        print(bad[["vintage_date", "lang", "problems"]].to_string(index=False))
    fixed = [(p["upload"].date(), p["lang"], p["meta"]["header_fixes"]) for p in parsed if p["meta"]["header_fixes"]]
    print(f"files with header typos corrected: {len(fixed)}")
    for f in fixed:
        print("   ", f)
    print("parse modes:", index.parse_mode.value_counts().to_dict())
    print("regimes:", index.regime.value_counts().to_dict(),
          "| first cross_border vintage:", index.loc[index.regime == "cross_border", "vintage_date"].min())
    print("preliminary-marked months found in", (index.preliminary_months != "").sum(), "vintages")
    print(f"lag days: min {index.lag_days.min()}, median {index.lag_days.median()}, max {index.lag_days.max()}")


if __name__ == "__main__":
    main()
