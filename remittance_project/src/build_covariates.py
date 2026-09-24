"""Monthly covariates: CBR USD/RUB, EIA Brent, MOEX IMOEX. Only complete months are kept."""
import os
import xml.etree.ElementTree as ET

import pandas as pd
import xlrd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "covariates")
OUT = os.path.join(ROOT, "data", "processed", "covariates_monthly.csv")
DOWNLOAD_DATE = pd.Timestamp("2026-09-21")  # raw files were fetched on this date


def usd_rub():
    recs = []
    for rec in ET.parse(os.path.join(RAW, "cbr_usd_rub_daily.xml")).getroot().findall("Record"):
        value = float(rec.find("Value").text.replace(",", ".")) / float(rec.find("Nominal").text)
        recs.append((pd.to_datetime(rec.get("Date"), format="%d.%m.%Y"), value))
    daily = pd.Series(dict(recs)).sort_index()
    grouped = daily.groupby(daily.index.to_period("M"))
    return pd.DataFrame({"rub_per_usd_eom": grouped.last(), "rub_per_usd_avg": grouped.mean()})


def brent():
    book = xlrd.open_workbook(os.path.join(RAW, "eia_brent_monthly.xls"))
    sheet = book.sheet_by_name("Data 1")
    vals = {}
    for r in range(3, sheet.nrows):
        d, v = sheet.cell_value(r, 0), sheet.cell_value(r, 1)
        if isinstance(d, float) and isinstance(v, float):
            vals[xlrd.xldate_as_datetime(d, book.datemode)] = v
    s = pd.Series(vals)
    s.index = s.index.to_period("M")
    return s.rename("brent_avg")


def imoex():
    rows = {}
    with open(os.path.join(RAW, "moex_imoex_monthly.csv"), encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            p = line.strip().split(";")
            if len(p) >= 8 and p[6][:4].isdigit():
                rows[pd.Period(p[6][:7], freq="M")] = float(p[1])
    return pd.Series(rows, name="imoex_close")


def main():
    cov = pd.concat([usd_rub(), brent(), imoex()], axis=1, sort=True)
    last_complete = (DOWNLOAD_DATE.to_period("M") - 1)
    cov = cov[(cov.index >= pd.Period("2005-01", "M")) & (cov.index <= last_complete)]
    cov.index.name = "month"
    cov.to_csv(OUT)
    print(cov.shape, cov.index.min(), "->", cov.index.max())
    print("missing values per column:", cov.isna().sum().to_dict())
    print(cov.loc["2022-01":"2022-06"].round(2).to_string())


if __name__ == "__main__":
    main()
