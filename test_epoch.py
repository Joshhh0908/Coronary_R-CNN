import os
import glob
import csv
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from FE import FeatureExtractorFE
from RPN import RPN1D, interval_iou_1d
from roi_pool import roi_pool_1d
from heads import RoIHeads

from cached_ds import CachedWindowDataset, collate_fn


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def parse_int_list(s: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())

def clamp_boxes_1d(boxes: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    b = boxes.clone()
    b[:, 0] = b[:, 0].clamp(lo, hi)
    b[:, 1] = b[:, 1].clamp(lo, hi)
    b[:, 1] = torch.maximum(b[:, 1], b[:, 0] + 1e-3)
    return b

def iou_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # expects [N,2], [M,2] in same coord space
    return interval_iou_1d(a, b)

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)

def plaque_bits_to_class(bits01: torch.Tensor) -> torch.Tensor:
    """
    bits01: [N,2] booleans/0-1 for (calc_bit, noncalc_bit)
      (0,0)->0 background
      (0,1)->1 non-calc
      (1,1)->2 mixed
      (1,0)->3 calcified
    """
    c = bits01[:, 0].long()
    n = bits01[:, 1].long()
    out = torch.zeros((bits01.shape[0],), dtype=torch.long, device=bits01.device)
    out[(c == 0) & (n == 1)] = 1
    out[(c == 1) & (n == 1)] = 2
    out[(c == 1) & (n == 0)] = 3
    return out
def save_cm_png(cm: np.ndarray, labels: List[str], out_png: str, title: str, normalize: bool = False):
    cm_show = cm.astype(np.float32)

    if normalize:
        row_sum = cm_show.sum(axis=1, keepdims=True)
        cm_show = np.divide(cm_show, np.maximum(row_sum, 1.0))  # per-GT-row normalize

    plt.figure(figsize=(6, 5))
    plt.imshow(cm_show, cmap="Blues", interpolation="nearest")  # <-- BLUE COLORMAP
    plt.title(title)
    plt.xlabel("Pred")
    plt.ylabel("GT")
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels)), labels)

    # numbers
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm_show[i, j]
            txt = f"{v:.2f}" if normalize else str(int(cm[i, j]))
            plt.text(j, i, txt, ha="center", va="center", fontsize=9,
                     color="white" if v > (cm_show.max() * 0.5) else "black")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def save_cm_csv(cm: np.ndarray, labels: List[str], out_csv: str):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gt\\pred"] + labels)
        for i, lab in enumerate(labels):
            w.writerow([lab] + list(cm[i].astype(int)))


def norm_img_percentile(x: np.ndarray, lo=1, hi=99) -> np.ndarray:
    a = np.percentile(x, lo)
    b = np.percentile(x, hi)
    if b <= a:
        return np.zeros_like(x)
    y = (x - a) / (b - a)
    return np.clip(y, 0, 1)


@dataclass
class EvalAgg:
    gt_total: int = 0
    rpn_hit03: int = 0
    rpn_hit05: int = 0
    rpn_hit07: int = 0

    matched_gt_for_head: int = 0
    refined_iou_sum: float = 0.0

    sten_correct: int = 0
    plaque_exact_correct: int = 0
    bit0_correct: int = 0
    bit1_correct: int = 0

    def rpn_recall(self):
        d = max(1, self.gt_total)
        return (self.rpn_hit03 / d, self.rpn_hit05 / d, self.rpn_hit07 / d)


# -----------------------------
# Main
# -----------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)

    # must match model shape
    ap.add_argument("--stenosis_classes", type=int, required=True)
    ap.add_argument("--anchor_lengths", type=str, required=True)

    # proposal filters (deployment-style)
    ap.add_argument("--score_thresh", type=float, default=0.05)
    ap.add_argument("--pre_nms_topk", type=int, default=600)
    ap.add_argument("--post_nms_topk", type=int, default=100)
    ap.add_argument("--nms_iou", type=float, default=0.5)

    # recall + matching
    ap.add_argument("--topk_for_recall", type=int, default=50)
    ap.add_argument("--roi_len", type=int, default=16)
    ap.add_argument("--match_iou_for_head", type=float, default=0.5)

    # viz
    ap.add_argument("--max_viz", type=int, default=80)
    ap.add_argument("--viz_w_slice", type=str, default="mid", choices=["mid", "mip"])
    ap.add_argument("--viz_lo", type=float, default=1.0)
    ap.add_argument("--viz_hi", type=float, default=99.0)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(args.out_dir)
    viz_dir = ensure_dir(os.path.join(out_dir, "viz"))

    # load ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")
    stride = float(ckpt.get("stride", 16.0))

    anchor_lengths = parse_int_list(args.anchor_lengths)

    # build models
    fe = FeatureExtractorFE(in_ch=1, norm="gn").to(device)
    rpn = RPN1D(in_c=512, anchor_lengths=anchor_lengths).to(device)
    roi_heads = RoIHeads(stenosis_num_classes=args.stenosis_classes, in_c=512, hidden=256, dropout=0.1).to(device)

    fe.load_state_dict(ckpt["fe"], strict=True)
    rpn.load_state_dict(ckpt["rpn"], strict=True)
    roi_heads.load_state_dict(ckpt["roi_heads"], strict=True)

    fe.eval(); rpn.eval(); roi_heads.eval()

    ds = CachedWindowDataset(args.data, args.split)
    loader = DataLoader(
        ds, batch_size=args.bs, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn
    )

    agg = EvalAgg()

    # confusion matrices (only for matched GTs)
    K = args.stenosis_classes
    plaque_labels = ["bg0", "noncalc1", "mixed2", "calc3"]
    sten_labels = [str(i) for i in range(K)]

    cm_sten = np.zeros((K, K), dtype=np.int64)
    cm_plaque = np.zeros((4, 4), dtype=np.int64)

    viz_done = 0

    for (x, targets) in loader:
        x = x.to(device, non_blocking=True)   # [B,1,D,H,W] (your pipeline)
        B = x.shape[0]

        feat = fe(x)                          # [B,512,Lf]
        B, C, Lf = feat.shape

        obj_logits, deltas, anchors = rpn(feat)
        proposals_feat, scores_list = rpn.propose(
            obj_logits, deltas, anchors, Lf=Lf,
            score_thresh=args.score_thresh,
            pre_nms_topk=args.pre_nms_topk,
            post_nms_topk=args.post_nms_topk,
            iou_thresh=args.nms_iou,
            allow_fallback=True,
        )

        pooled, batch_ix, rois_cat = roi_pool_1d(feat, proposals_feat, roi_len=args.roi_len)

        if pooled.numel() > 0:
            out = roi_heads(pooled, rois_cat, Lf=Lf)
            plaque_logits = out["plaque_logits"]      # [R,2]
            sten_logits = out["stenosis_logits"]      # [R,K]
            refined_feat = out["roi_refined"]         # [R,2] (feature coords)

            # decode plaque + sten per roi
            plaque_bits = (sigmoid(plaque_logits) >= 0.5).long()
            plaque_pred = plaque_bits_to_class(plaque_bits)         # [R] in 0..3
            sten_pred = torch.argmax(sten_logits, dim=1).long()     # [R] in 0..K-1
        else:
            refined_feat = rois_cat
            plaque_pred = torch.empty((0,), device=device, dtype=torch.long)
            sten_pred = torch.empty((0,), device=device, dtype=torch.long)

        # per-sample eval
        for b in range(B):
            gt_boxes_s = targets[b]["boxes"].to(device).float()     # [G,2] slice coords
            gt_plq = targets[b]["plaque"].to(device).long()         # [G]
            gt_ste = targets[b]["stenosis"].to(device).long()       # [G]

            G = gt_boxes_s.shape[0]
            agg.gt_total += int(G)
            if G == 0:
                continue

            # proposals for recall (slice coords)
            prop_f = proposals_feat[b]
            prop_s = (prop_f * stride).to(torch.float32) if prop_f is not None and prop_f.numel() else torch.empty((0,2), device=device)
            prop_s = clamp_boxes_1d(prop_s, 0.0, float(x.shape[2] - 1))

            # apply topk_for_recall
            if prop_s.numel() > 0 and scores_list is not None:
                sc = scores_list[b]
                if sc is not None and sc.numel() == prop_s.shape[0]:
                    k = min(args.topk_for_recall, sc.numel())
                    topk = torch.topk(sc, k=k, largest=True).indices
                    prop_s_k = prop_s[topk]
                else:
                    prop_s_k = prop_s
            else:
                prop_s_k = prop_s

            # rpn recall hits
            if prop_s_k.numel() > 0:
                ious = iou_1d(gt_boxes_s, prop_s_k)  # [G,P]
                best = ious.max(dim=1).values
                agg.rpn_hit03 += int((best >= 0.3).sum().item())
                agg.rpn_hit05 += int((best >= 0.5).sum().item())
                agg.rpn_hit07 += int((best >= 0.7).sum().item())

            # head matching uses REFINED rois (more meaningful)
            ridx = torch.where(batch_ix == b)[0]
            if ridx.numel() == 0:
                continue

            ref_s = refined_feat[ridx] * stride
            ref_s = clamp_boxes_1d(ref_s, 0.0, float(x.shape[2] - 1))

            ious_h = iou_1d(gt_boxes_s, ref_s)  # [G,Rb]
            best_iou, best_j = ious_h.max(dim=1)

            matched = (best_iou >= args.match_iou_for_head)
            m_idx = torch.where(matched)[0]
            if m_idx.numel() == 0:
                continue

            agg.matched_gt_for_head += int(m_idx.numel())
            agg.refined_iou_sum += float(best_iou[m_idx].sum().item())

            # predictions for matched GTs
            roi_pick = ridx[best_j[m_idx]]  # indices into pooled outputs
            pred_ste = sten_pred[roi_pick]
            pred_plq = plaque_pred[roi_pick]

            gt_ste_m = gt_ste[m_idx].clamp(0, K - 1)
            gt_plq_m = gt_plq[m_idx].clamp(0, 3)

            agg.sten_correct += int((pred_ste == gt_ste_m).sum().item())
            agg.plaque_exact_correct += int((pred_plq == gt_plq_m).sum().item())

            # plaque bits acc
            # gt bits from label mapping: 0 bg, 1 noncalc, 2 mixed, 3 calc
            gt_bits = torch.zeros((gt_plq_m.numel(), 2), device=device, dtype=torch.long)
            gt_bits[:, 0] = ((gt_plq_m == 3) | (gt_plq_m == 2)).long()  # calc present
            gt_bits[:, 1] = ((gt_plq_m == 1) | (gt_plq_m == 2)).long()  # noncalc present

            pred_bits = torch.zeros_like(gt_bits)
            pred_bits[:, 0] = ((pred_plq == 3) | (pred_plq == 2)).long()
            pred_bits[:, 1] = ((pred_plq == 1) | (pred_plq == 2)).long()

            agg.bit0_correct += int((pred_bits[:, 0] == gt_bits[:, 0]).sum().item())
            agg.bit1_correct += int((pred_bits[:, 1] == gt_bits[:, 1]).sum().item())

            # confusion matrices (matched only)
            for i in range(m_idx.numel()):
                g_s = int(gt_ste_m[i].item())
                p_s = int(pred_ste[i].item())
                cm_sten[g_s, p_s] += 1

                g_p = int(gt_plq_m[i].item())
                p_p = int(pred_plq[i].item())
                cm_plaque[g_p, p_p] += 1

            # --- Visualization ---
            if viz_done < args.max_viz:
                x_cpu = x[b, 0].detach().float().cpu().numpy()  # [D,H,W]
                D, H, W = x_cpu.shape

                # ✅ Pick a REAL slice (crisp), not a projection
                # Option A (recommended): mid-W slice => [D,H]
                img_DH = x_cpu[:, :, W // 2]

                # If you insist on something "more global", use MIP over W instead of mean over H:
                # img_DH = x_cpu.max(axis=2)  # [D,H]  (still less blurry than mean over H)

                # Put depth on x-axis: show as [H,D]
                img_show = img_DH.T  # [H,D]

                # ✅ Robust windowing (percentiles) on raw values

                vmin, vmax = -3.0, 3.0

                if vmax <= vmin:
                    vmax = vmin + 1e-6

                plt.figure(figsize=(14, 4))
                plt.imshow(
                    img_show,
                    cmap="gray",
                    origin="lower",
                    aspect="auto",
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="none",   # or "nearest"
                )

                # draw GT boxes (green) across full height
                for gi in range(G):
                    s, e = gt_boxes_s[gi].detach().cpu().numpy().tolist()
                    rect = Rectangle((s, 0), max(1.0, e - s), H, fill=False, linewidth=2, edgecolor="lime")
                    plt.gca().add_patch(rect)

                # draw matched predicted refined boxes (red) + text
                for i in range(m_idx.numel()):
                    pj = int(best_j[m_idx[i]].item())
                    s, e = ref_s[pj].detach().cpu().numpy().tolist()

                    rect = Rectangle((s, 0), max(1.0, e - s), H, fill=False, linewidth=2, edgecolor="red")
                    plt.gca().add_patch(rect)

                    ps = int(pred_ste[i].item())
                    pp = int(pred_plq[i].item())
                    plt.text(
                        s, H - 8 - 12 * (i % 6),   # put labels near top
                        f"pred sten={ps}, plq={pp}",
                        fontsize=8,
                        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
                    )

                plt.title(f"{args.split} sample b={b} | GT(green) vs Pred(refined red)")
                plt.xlabel("Depth (slice index)")
                plt.ylabel("Height (H)")

                out_png = os.path.join(viz_dir, f"{args.split}_idx_{viz_done:04d}.png")
                plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0)
                plt.close()
                viz_done += 1
                arr = x[b,0].detach().float().cpu().numpy()
                print("min/mean/std/max:", arr.min(), arr.mean(), arr.std(), arr.max())



    # final metrics
    r03, r05, r07 = agg.rpn_recall()
    denom_m = max(1, agg.matched_gt_for_head)

    metrics = {
        "split": args.split,
        "ckpt": args.ckpt,
        "stride": stride,
        "gt_total": agg.gt_total,
        "rpn_recall@0.3": r03,
        "rpn_recall@0.5": r05,
        "rpn_recall@0.7": r07,
        "matched_gt_for_head": agg.matched_gt_for_head,
        "match_iou_for_head": args.match_iou_for_head,
        "stenosis_acc_on_matched": agg.sten_correct / denom_m,
        "plaque_exact_acc_on_matched": agg.plaque_exact_correct / denom_m,
        "plaque_bit0_acc": agg.bit0_correct / denom_m,
        "plaque_bit1_acc": agg.bit1_correct / denom_m,
        "refined_iou_mean_on_matched": agg.refined_iou_sum / denom_m,
        "num_viz_saved": viz_done,
    }

    # write metrics csv
    out_metrics_csv = os.path.join(out_dir, f"{args.split}_metrics.csv")
    with open(out_metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        for k, v in metrics.items():
            w.writerow([k, v])

    # write confusion matrices
    save_cm_csv(cm_sten, sten_labels, os.path.join(out_dir, f"{args.split}_cm_stenosis.csv"))
    save_cm_csv(cm_plaque, plaque_labels, os.path.join(out_dir, f"{args.split}_cm_plaque.csv"))

    # --- stenosis PNGs ---
    save_cm_png(cm_sten, sten_labels,
                os.path.join(out_dir, f"{args.split}_cm_stenosis.png"),
                "Stenosis Confusion Matrix (matched GT)",
                normalize=False)

    save_cm_png(cm_sten, sten_labels,
                os.path.join(out_dir, f"{args.split}_cm_stenosis_norm.png"),
                "Stenosis Confusion Matrix (row-normalized)",
                normalize=True)

    # plaque PNGs
    save_cm_png(cm_plaque, plaque_labels,
                os.path.join(out_dir, f"{args.split}_cm_plaque.png"),
                "Plaque Confusion Matrix (matched GT)",
                normalize=False)

    save_cm_png(cm_plaque, plaque_labels,
                os.path.join(out_dir, f"{args.split}_cm_plaque_norm.png"),
                "Plaque Confusion Matrix (row-normalized)",
                normalize=True)


    print("DONE. Outputs in:", out_dir)
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
