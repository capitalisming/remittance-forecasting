"""Stage 3 (Armenia panel): parse the 'MONEY TRANSFERS OF INDIVIDUALS/NATURAL PERSONS ...
thous(and) US dollars' table out of pdftotext -layout extractions of CBA (Central Bank of
Armenia) Statistical Overview bulletins, pulling the single-month 'Total' inflow row (current
month, current-year block) as the monthly series value.

Source bulletins: data/raw/armenia_cba/bulletin_text/*.txt (pdftotext -layout output; raw PDFs
not committed -- CBA's own copyright notice on each bulletin, same reasoning as the Springer
llncs class files). Not all months are recoverable: some downloads came back with a corrupted
PDF xref table (only page numbers/chart titles survived pdftotext's error-recovery), tracked in
armenia_corrupted_months.txt.
"""
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TXT_DIR = os.path.join(ROOT, "data", "raw", "armenia_cba", "bulletin_text")
OUT = os.path.join(ROOT, "data", "processed", "armenia_total_transfers_monthly.csv")


def parse_file(path, year, month):
    text = open(path, encoding="utf-8", errors="ignore").read()
    lines = text.splitlines()

    # Find all "TOTAL MONEY TRANSFERS..." table starts (there may be a USD one and a dram one;
    # we want thousand US dollars).
    starts = [i for i, l in enumerate(lines)
              if re.search(r"MONEY TRANSFERS OF (INDIVIDUALS|NATURAL PERSONS)", l, re.I)]
    if not starts:
        return None

    for start in starts:
        # confirm this block is denominated in thousand US dollars, not million drams
        window = "\n".join(lines[start:start + 6])
        if "thous" not in window.lower() or "us dollar" not in window.lower():
            continue

        # within ~40 lines after the header, find the year block matching this file's year,
        # then the Total row right after it (before the next Country/blank-heavy break)
        block = lines[start:start + 60]
        year_line_idx = None
        for j, l in enumerate(block):
            if re.fullmatch(r"\s*" + str(year) + r"\s*", l):
                year_line_idx = j
                break
        if year_line_idx is None:
            # some issues put the year on the same line as other text; try loose search
            for j, l in enumerate(block):
                if str(year) in l and len(l.strip()) < 8:
                    year_line_idx = j
                    break
        if year_line_idx is None:
            continue

        for l in block[year_line_idx:year_line_idx + 20]:
            m = re.match(r"\s*Total\s+([\-\d,\.]+)\s+([\-\d,\.]+)\s+([\-\d,\.]+)", l)
            if m:
                inflow = float(m.group(1).replace(",", ""))
                return inflow
    return None


def main():
    rows = []
    files = sorted(glob.glob(os.path.join(TXT_DIR, "*.txt")))
    for path in files:
        ym = os.path.basename(path).replace(".txt", "")
        year, month = ym.split("-")
        val = parse_file(path, year, int(month))
        rows.append((ym, val))

    ok = [r for r in rows if r[1] is not None]
    bad = [r for r in rows if r[1] is None]
    print(f"parsed: {len(ok)}   failed: {len(bad)}")
    if bad:
        print("corrupted/unparseable months:", [r[0] for r in bad])

    with open(OUT, "w") as f:
        f.write("month,total_inflow_usd_thousand\n")
        for ym, val in ok:
            f.write(f"{ym},{val}\n")
    print(f"written {OUT}")

    corrupted_path = os.path.join(os.path.dirname(TXT_DIR), "armenia_corrupted_months.txt")
    with open(corrupted_path, "w") as f:
        f.write("\n".join(r[0] for r in bad) + "\n")
    print(f"written {corrupted_path}")


if __name__ == "__main__":
    main()
