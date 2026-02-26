# aggregate_eval_epochs.py
# Collate eval outputs (summary.csv + score_separation_summary.csv + selected quantiles)
# across epoch folders into one CSV.

import os
import re
import glob
import csv
import argparse

def read_single_row_csv(path):
    """Reads a CSV with header + single row. Returns dict or None."""
    if not os.path.isfile(path):
        return None
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            return row
    return None

def read_score_separation_quantiles(path):
    """
    Reads score_separation.csv which has rows like TP_matched / FP_unmatched / TP_IoU
    Returns a flat dict with useful fields (fp_p95, fp_p99, tp_p50, etc).
    """
    if not os.path.isfile(path):
        return {}
    out = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            g = row.get("group", "")
            if g == "TP_matched":
                out["tp_p10"] = row.get("p10")
                out["tp_p50"] = row.get("p50")
                out["tp_p90"] = row.get("p90")
                out["tp_p95"] = row.get("p95")
                out["tp_p99"] = row.get("p99")
                out["tp_mean"] = row.get("mean")
            elif g == "FP_unmatched":
                out["fp_p10"] = row.get("p10")
                out["fp_p50"] = row.get("p50")
                out["fp_p90"] = row.get("p90")
                out["fp_p95"] = row.get("p95")
                out["fp_p99"] = row.get("p99")
                out["fp_mean"] = row.get("mean")
            elif g == "TP_IoU":
                out["tp_iou_p10"] = row.get("p10")
                out["tp_iou_p50"] = row.get("p50")
                out["tp_iou_p90"] = row.get("p90")
                out["tp_iou_mean"] = row.get("mean")
    return out

def epoch_from_folder(name):
    # expects epoch_010 or epoch_010.pt etc
    m = re.search(r"epoch_(\d+)", name)
    return int(m.group(1)) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Folder that contains eval outputs per epoch (e.g. RUN_DIR/eval_val)")
    ap.add_argument("--out", required=True,
                    help="Output CSV path (e.g. RUN_DIR/eval_val/aggregate.csv)")
    ap.add_argument("--pattern", default="epoch_*",
                    help="Subfolder glob under --root (default: epoch_*)")
    args = ap.parse_args()

    root = args.root
    folders = sorted(glob.glob(os.path.join(root, args.pattern)))

    rows = []
    for d in folders:
        if not os.path.isdir(d):
            continue
        ep = epoch_from_folder(os.path.basename(d))
        if ep is None:
            continue

        summary = read_single_row_csv(os.path.join(d, "summary.csv")) or {}
        sep_sum = read_single_row_csv(os.path.join(d, "score_separation_summary.csv")) or {}
        sep_q = read_score_separation_quantiles(os.path.join(d, "score_separation.csv"))

        row = {"epoch": ep, "eval_dir": d}
        # merge (later keys overwrite earlier keys if duplicated)
        row.update(summary)
        row.update(sep_sum)
        row.update(sep_q)

        rows.append(row)

    if not rows:
        raise SystemExit(f"No epoch folders found under: {root}")

    # sort by epoch
    rows.sort(key=lambda x: x["epoch"])

    # union of all keys for header
    all_keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        w.writerows(rows)

    print("Saved aggregate:", args.out)
    print("Rows:", len(rows))

if __name__ == "__main__":
    main()

