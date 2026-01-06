# check_label_ranges.py
import argparse
from ast import literal_eval
from collections import Counter

import pandas as pd


def is_normal_list(lst) -> bool:
    return isinstance(lst, (list, tuple)) and any(isinstance(x, str) and x.lower() == "normal" for x in lst)


def extract_triplet_values(lst):
    """
    lst format:
      ['PATIENT', 'VESSEL', s_mm, e_mm, val, s_mm, e_mm, val, ...]
    returns: list of vals
    """
    if not isinstance(lst, (list, tuple)) or len(lst) < 3:
        return []
    if is_normal_list(lst):
        return []
    if len(lst) < 5:
        return []

    vals = []
    n = (len(lst) - 2) // 3
    for k in range(n):
        v = lst[2 + 3 * k + 2]
        # keep ints if possible
        try:
            v = int(v)
        except Exception:
            pass
        vals.append(v)
    return vals


def scan_csv(csv_path: str):
    df = pd.read_csv(csv_path, header=None)

    plaque_col = df.columns[-2]
    sten_col = df.columns[-1]

    plaque_vals = []
    sten_vals = []
    normal_rows = 0
    lesion_rows = 0

    for p_str, s_str in zip(df[plaque_col], df[sten_col]):
        try:
            p = literal_eval(p_str)
            s = literal_eval(s_str)
        except Exception:
            # skip malformed rows
            continue

        is_normal = is_normal_list(p) or is_normal_list(s)
        if is_normal:
            normal_rows += 1
            # treat normal as class 0 (no plaque / no stenosis)
            plaque_vals.append(0)
            sten_vals.append(0)
            continue

        lesion_rows += 1
        plaque_vals.extend(extract_triplet_values(p))
        sten_vals.extend(extract_triplet_values(s))

    plaque_ctr = Counter(plaque_vals)
    sten_ctr = Counter(sten_vals)

    plaque_unique = sorted(plaque_ctr.keys())
    sten_unique = sorted(sten_ctr.keys())

    print(f"\n=== {csv_path} ===")
    print(f"rows: {len(df)} | normal rows: {normal_rows} | lesion rows: {lesion_rows}")
    print(f"Plaque (2nd last col) unique (raw, with 0=normal): {plaque_unique}")
    if plaque_unique:
        print(f"Plaque min/max: {min(plaque_unique)} / {max(plaque_unique)}")
    print(f"Plaque counts: {plaque_ctr}")

    print(f"\nStenosis (last col) unique (raw, with 0=normal): {sten_unique}")
    if sten_unique:
        print(f"Stenosis min/max: {min(sten_unique)} / {max(sten_unique)}")
    print(f"Stenosis counts: {sten_ctr}")

    # Convenience checks
    ok_sten = set(sten_unique).issubset(set(range(0, 6)))
    print(f"\nCheck: stenosis within 0..5 ? {ok_sten}")

    # Plaque might be encoded as 1..3 (then zero-based becomes 0..2)
    nonzero_plaque = sorted([v for v in plaque_unique if v != 0])
    if nonzero_plaque:
        minp, maxp = min(nonzero_plaque), max(nonzero_plaque)
        if minp == 1 and maxp == 3:
            zb = sorted(set([v - 1 for v in nonzero_plaque]))
            print(f"Plaque looks like 1..3 encoding. Zero-based would be: {zb} (expected 0..2).")
        else:
            print(f"Plaque non-zero range is {minp}..{maxp} (not 1..3).")
    else:
        print("No non-zero plaque values found (all normal).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="One or more CSV files")
    args = ap.parse_args()

    for p in args.csv:
        scan_csv(p)


if __name__ == "__main__":
    main()
