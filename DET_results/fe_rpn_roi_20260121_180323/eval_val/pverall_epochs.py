import pandas as pd

df = pd.read_csv("aggregate_epochs.csv")

# Sort helpers
def show(title, sub):
    cols = [
        "epoch", "det_recall", "fp_per_sample", "total_fp", "mean_iou_matched",
        "tp_p50", "fp_p95", "fp_p99", "tp_p50_minus_fp_p95"
    ]
    cols = [c for c in cols if c in sub.columns]
    print("\n===", title, "===")
    print(sub[cols].to_string(index=False))

# 1) Best (lowest FP/sample)
show("Lowest FP/sample overall", df.sort_values("fp_per_sample").head(10))

# 2) Best recall overall
show("Highest recall overall", df.sort_values("det_recall", ascending=False).head(10))

# 3) Best epochs for recall >= thresholds
for r in [0.85, 0.82, 0.80, 0.75]:
    sub = df[df["det_recall"] >= r].sort_values("fp_per_sample").head(10)
    show(f"Lowest FP/sample with recall >= {r}", sub)

# 4) Best epochs under FP budgets
for budget in [10, 15, 20, 30]:
    sub = df[df["fp_per_sample"] <= budget].sort_values("det_recall", ascending=False).head(10)
    show(f"Highest recall with FP/sample <= {budget}", sub)
