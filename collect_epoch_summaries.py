import os
import glob
import csv
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="e.g. .../fe_rpn_roi_YYYYMMDD_HHMMSS")
    ap.add_argument("--pattern", default="test/epoch_*/summary.csv",
                    help="glob pattern relative to run_dir")
    ap.add_argument("--out_csv", default="", help="output csv path (default: run_dir/epoch_trend.csv)")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    files = sorted(glob.glob(os.path.join(run_dir, args.pattern)))

    if not files:
        raise FileNotFoundError(f"No files matched: {os.path.join(run_dir, args.pattern)}")

    rows = []
    for f in files:
        # epoch number from .../epoch_140/summary.csv
        epoch_dir = os.path.basename(os.path.dirname(f))
        try:
            epoch = int(epoch_dir.split("_")[-1])
        except Exception:
            epoch = None

        df = pd.read_csv(f)
        if df.shape[0] != 1:
            # your summary.csv should be single-row; take first row anyway
            rec = df.iloc[0].to_dict()
        else:
            rec = df.iloc[0].to_dict()

        rec["epoch"] = epoch
        rec["summary_path"] = f
        rows.append(rec)

    out = pd.DataFrame(rows)

    # sort by epoch if available
    if out["epoch"].notna().any():
        out = out.sort_values(["epoch"], na_position="last")

    out_csv = args.out_csv.strip() or os.path.join(run_dir, "epoch_trend.csv")
    out.to_csv(out_csv, index=False)

    print(f"Found {len(files)} summaries")
    print(f"Saved: {out_csv}")

    # Also print a quick “headline” view
    cols = [c for c in [
        "epoch",
        "det_recall",
        "fp_per_sample",
        "mean_iou_matched",
        "sten_f1_macro",
        "sten_recall_macro",
        "plaq_f1_macro",
        "plaq_recall_macro",
    ] if c in out.columns]
    if cols:
        print("\nPreview:")
        print(out[cols].tail(10).to_string(index=False))

if __name__ == "__main__":
    main()
