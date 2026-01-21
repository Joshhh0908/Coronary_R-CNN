# collect_sweep_results.py
import os
import re
import argparse
import pandas as pd

def read_one_csv(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] failed reading {path}: {e}")
        return None

def parse_thresh(folder_name: str):
    # expects t_0, t_0.005, t_0.1, etc.
    m = re.match(r"^t_(.+)$", folder_name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None

def pick_row(df, group_name):
    if df is None or df.empty:
        return None
    sub = df[df["group"] == group_name]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="sweep root, contains t_*/ subfolders")
    ap.add_argument("--out", default="sweep_aggregate.csv", help="output csv path")
    ap.add_argument("--pattern", default=r"^t_", help="subfolder regex to include (default: ^t_)")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    folder_re = re.compile(args.pattern)

    rows = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        if not folder_re.search(name):
            continue

        thr = parse_thresh(name)
        if thr is None:
            continue

        sum_df = read_one_csv(os.path.join(full, "summary.csv"))
        sep_sum_df = read_one_csv(os.path.join(full, "score_separation_summary.csv"))
        sep_df = read_one_csv(os.path.join(full, "score_separation.csv"))

        if sum_df is None or sum_df.empty:
            print(f"[WARN] missing/empty summary.csv in {full}")
            continue

        # summary.csv is 1 row
        s = sum_df.iloc[0].to_dict()

        # separation summary is 1 row
        sep_s = {}
        if sep_sum_df is not None and not sep_sum_df.empty:
            sep_s = sep_sum_df.iloc[0].to_dict()

        # score_separation.csv has 3 rows: TP_matched, FP_unmatched, TP_IoU
        tp = pick_row(sep_df, "TP_matched") or {}
        fp = pick_row(sep_df, "FP_unmatched") or {}

        row = {
            "thresh": thr,

            # detection
            "det_recall": s.get("det_recall", None),
            "total_fp": s.get("total_fp", None),
            "fp_per_sample": s.get("fp_per_sample", None),
            "mean_iou_matched": s.get("mean_iou_matched", None),

            # useful score percentiles
            "tp_p10": tp.get("p10", None),
            "tp_p50": tp.get("p50", None),
            "tp_p90": tp.get("p90", None),
            "fp_p50": fp.get("p50", None),
            "fp_p90": fp.get("p90", None),
            "fp_p95": fp.get("p95", None),
            "fp_p99": fp.get("p99", None),
            "fp_p100": fp.get("p100", None),

            # how much junk remains above common cutoffs
            "fp_frac_gt_0.01": fp.get("frac_gt_0.01", None),
            "fp_frac_gt_0.05": fp.get("frac_gt_0.05", None),
            "fp_frac_gt_0.1": fp.get("frac_gt_0.1", None),

            # your scalar separations (if present)
            "tp_p50_minus_fp_p95": sep_s.get("tp_p50_minus_fp_p95", None),
            "tp_p10_minus_fp_p90": sep_s.get("tp_p10_minus_fp_p90", None),
            "tp_mean_minus_fp_mean": sep_s.get("tp_mean_minus_fp_mean", None),

            # provenance
            "folder": name,
        }

        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values("thresh").reset_index(drop=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved aggregate: {args.out}\n")

    # Quick “what should I look at” view:
    cols = ["thresh", "det_recall", "fp_per_sample", "total_fp",
            "fp_p95", "fp_p99", "tp_p50", "tp_p10_minus_fp_p90", "tp_p50_minus_fp_p95"]
    present = [c for c in cols if c in out_df.columns]
    print(out_df[present].to_string(index=False))

if __name__ == "__main__":
    main()
