# eval_test_full.py
import os
import csv
import glob
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
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


def nms_1d_eval(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh=0.5):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    idx = scores.argsort(descending=True)
    keep = []
    while idx.numel() > 0:
        i = idx[0]
        keep.append(i)
        if idx.numel() == 1:
            break
        ious = interval_iou_1d(boxes[i:i+1], boxes[idx[1:]]).squeeze(0)
        idx = idx[1:][ious <= iou_thresh]
    return torch.stack(keep, dim=0)

def pct(x, ps=(0, 10, 50, 90, 95, 99, 100)):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return {p: None for p in ps}
    return {p: float(np.percentile(x, p)) for p in ps}

def fmt_pct(d):
    return " ".join([f"p{p}={d[p]:.4g}" for p in d.keys() if d[p] is not None])
# -------------------------
# Helpers: plaque bits <-> class
# -------------------------
def plaque_logits_to_class(plaque_logits: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """
    plaque_logits: [N,2] logits for (calc_bit, noncalc_bit)
    returns class [N] in {0,1,2,3}:
      0 none/bg
      1 non-calc
      2 mixed
      3 calcified
    """
    probs = torch.sigmoid(plaque_logits)
    calc = (probs[:, 0] >= thr).long()
    nonc = (probs[:, 1] >= thr).long()

    # mapping bits -> class
    # 00 -> 0
    # 01 -> 1
    # 11 -> 2
    # 10 -> 3
    out = torch.zeros_like(calc)
    out[(calc == 0) & (nonc == 1)] = 1
    out[(calc == 1) & (nonc == 1)] = 2
    out[(calc == 1) & (nonc == 0)] = 3
    return out


# -------------------------
# 1D greedy matching (score-sorted)
# -------------------------
def greedy_match_1d(pred_boxes: torch.Tensor, pred_scores: torch.Tensor,
                    gt_boxes: torch.Tensor, iou_thr: float):
    """
    pred_boxes: [N,2] (slice coords)
    pred_scores: [N]
    gt_boxes: [M,2] (slice coords)

    returns:
      pred2gt: LongTensor [N] with gt idx or -1
      gt2pred: LongTensor [M] with pred idx or -1
      matched_ious: FloatTensor [N] IoU of matched pred (0 if unmatched)
    """
    device = pred_boxes.device
    N = pred_boxes.shape[0]
    M = gt_boxes.shape[0]

    pred2gt = torch.full((N,), -1, dtype=torch.long, device=device)
    gt2pred = torch.full((M,), -1, dtype=torch.long, device=device)
    matched_ious = torch.zeros((N,), dtype=torch.float32, device=device)

    if N == 0 or M == 0:
        return pred2gt, gt2pred, matched_ious

    ious = interval_iou_1d(pred_boxes, gt_boxes)  # [N,M]
    order = torch.argsort(pred_scores, descending=True)
    gt_used = torch.zeros((M,), dtype=torch.bool, device=device)

    for pi in order:
        row = ious[pi]  # [M]
        gi = torch.argmax(row).item()
        best = row[gi].item()
        if best >= iou_thr and not gt_used[gi]:
            pred2gt[pi] = gi
            gt2pred[gi] = pi
            matched_ious[pi] = best
            gt_used[gi] = True

    return pred2gt, gt2pred, matched_ious


def update_cm_with_bg(cm: np.ndarray,
                      gt_labels: np.ndarray, pred_labels: np.ndarray,
                      pred2gt: torch.Tensor, gt2pred: torch.Tensor):
    """
    cm: (C+1, C+1) last index is BG
    gt_labels: [M] in 0..C-1
    pred_labels: [N] in 0..C-1
    pred2gt: [N] (-1 or gt idx)
    gt2pred: [M] (-1 or pred idx)
    """
    C = cm.shape[0] - 1
    BG = C

    pred2gt_np = pred2gt.detach().cpu().numpy()
    gt2pred_np = gt2pred.detach().cpu().numpy()

    # matched: GT row, Pred col
    for pi, gi in enumerate(pred2gt_np):
        if gi != -1:
            cm[int(gt_labels[gi]), int(pred_labels[pi])] += 1

    # FN: GT unmatched -> Pred = BG
    for gi, pi in enumerate(gt2pred_np):
        if pi == -1:
            cm[int(gt_labels[gi]), BG] += 1

    # FP: pred unmatched -> GT = BG
    for pi, gi in enumerate(pred2gt_np):
        if gi == -1:
            cm[BG, int(pred_labels[pi])] += 1


def cm_stats(cm: np.ndarray):
    """
    Returns per-class precision/recall/F1 (excluding BG), plus macro averages.
    """
    C = cm.shape[0] - 1
    BG = C
    eps = 1e-9

    # exclude BG row/col for "class metrics"
    tp = np.diag(cm)[:C].astype(np.float64)
    col_sum = cm[:, :C].sum(axis=0).astype(np.float64)  # predicted as class c
    row_sum = cm[:C, :].sum(axis=1).astype(np.float64)  # gt class c

    prec = tp / (col_sum + eps)
    rec = tp / (row_sum + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)

    macro = {
        "precision_macro": float(np.mean(prec)),
        "recall_macro": float(np.mean(rec)),
        "f1_macro": float(np.mean(f1)),
    }
    return prec, rec, f1, macro


# -------------------------
# Visualization
# -------------------------
def _robust_minmax(img, lo=1, hi=99):
    vmin, vmax = np.percentile(img, [lo, hi])
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def save_viz_grayproj(
    out_path: str,
    x_3d: torch.Tensor,          # [D,H,W] CPU torch
    gt_boxes: torch.Tensor,      # [G,2] CPU torch (slice coords)
    pred_boxes: torch.Tensor,    # [P,2] CPU torch (slice coords)
    pred_scores: torch.Tensor,   # [P] CPU torch
    pred_sten: torch.Tensor,     # [P] CPU torch
    pred_plaq: torch.Tensor,     # [P] CPU torch
    title: str = ""
):
    """
    Two-panel plot:
      (top) 1D mean-intensity signal along D, with GT(red) and Pred(blue) spans
      (bottom) grayscale mean projection (mean over H) => [D,W], with spans
    """
    x_3d = x_3d.float()
    D, H, W = x_3d.shape

    signal = x_3d.mean(dim=(1, 2)).cpu().numpy()    # [D]
    proj_DW = x_3d.mean(dim=1).cpu().numpy()        # [D,W]
    vmin, vmax = _robust_minmax(proj_DW, 1, 99)

    fig = plt.figure(figsize=(16, 6))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(np.arange(D), signal)

    # GT spans (red)
    for (g0, g1) in gt_boxes.cpu().numpy():
        ax1.axvspan(g0, g1, color="red", alpha=0.18, linewidth=0)

    # Pred spans (blue, alpha by score)
    P = int(pred_boxes.shape[0])
    for i in range(P):
        p0, p1 = pred_boxes[i].cpu().numpy()
        sc = float(pred_scores[i].item()) if pred_scores.numel() else 0.0
        ax1.axvspan(p0, p1, color="blue", alpha=min(0.35, 0.05 + 0.35 * sc), linewidth=0)

    # annotate top-k scores + classes
    if pred_scores.numel() and P > 0:
        topk = min(8, P)
        idx = torch.topk(pred_scores, topk).indices
        y_top = ax1.get_ylim()[1]
        for j in idx.tolist():
            p0, p1 = pred_boxes[j].cpu().numpy()
            sc = float(pred_scores[j].item())
            s = int(pred_sten[j].item())
            q = int(pred_plaq[j].item())
            ax1.text((p0 + p1) / 2.0, y_top * 0.95, f"{sc:.2f} s{s} p{q}",
                     ha="center", va="top", fontsize=9)

    ax1.set_xlim(0, D - 1)
    ax1.set_title(title)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.imshow(
        proj_DW.T,
        aspect="auto",
        origin="lower",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax2.set_ylabel("W (mean proj over H)")
    ax2.set_xlabel("D (slice index)")

    for (g0, g1) in gt_boxes.cpu().numpy():
        ax2.axvspan(g0, g1, color="red", alpha=0.18, linewidth=0)
    for i in range(P):
        p0, p1 = pred_boxes[i].cpu().numpy()
        sc = float(pred_scores[i].item()) if pred_scores.numel() else 0.0
        ax2.axvspan(p0, p1, color="blue", alpha=min(0.25, 0.05 + 0.25 * sc), linewidth=0)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# -------------------------
# Main eval
# -------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="root folder with train/val/test .pt")
    ap.add_argument("--ckpt", type=str, required=True, help="path to epoch_XXX.pt checkpoint")
    ap.add_argument("--out_dir", type=str, default="", help="output directory; default beside checkpoint")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--plaque_score_thresh", type=float, default=0.3,
                help="filter detections by plaque lesionness score")
    ap.add_argument("--plaque_bit_thr", type=float, default=0.5,
                    help="threshold for converting plaque bits to class")
    ap.add_argument("--use_combined_score", action="store_true",
                    help="if set, final_score = rpn_score * plaque_lesionness else plaque_lesionness only")
    ap.add_argument("--final_nms", action="store_true",
                    help="apply a final NMS after roi_refined")
    ap.add_argument("--final_nms_iou", type=float, default=0.5)


    # Must match training
    ap.add_argument("--stenosis_classes", type=int, required=True)  # K=5
    ap.add_argument("--anchor_lengths", type=str, default="1,2,3,5,7")
    ap.add_argument("--roi_len", type=int, default=16)
    ap.add_argument("--score_thresh", type=float, default=0.05)
    ap.add_argument("--pre_nms_topk", type=int, default=600)
    ap.add_argument("--post_nms_topk", type=int, default=100)
    ap.add_argument("--nms_iou", type=float, default=0.5)

    # Matching thresholds
    ap.add_argument("--match_iou", type=float, default=0.5)

    # How many PNGs
    ap.add_argument("--save_png", action="store_true")
    ap.add_argument("--max_png", type=int, default=50)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    stride = float(ckpt.get("stride", 16.0))  # slice-per-feature
    anchor_lengths = tuple(int(x) for x in args.anchor_lengths.split(",") if x.strip())

    # output folder
    if args.out_dir.strip():
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(args.ckpt)), "test_eval_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    png_dir = os.path.join(out_dir, "png")
    if args.save_png:
        os.makedirs(png_dir, exist_ok=True)

    print("Checkpoint:", args.ckpt)
    print("Out dir:", out_dir)
    print("stride:", stride, "anchors:", anchor_lengths)

    # Data
    ds = CachedWindowDataset(args.data, args.split)
    loader = DataLoader(
        ds, batch_size=args.bs, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn
    )
    print(f"{args.split} samples:", len(ds))

    # Model
    fe = FeatureExtractorFE(in_ch=1, norm="gn").to(device)
    rpn = RPN1D(in_c=512, anchor_lengths=anchor_lengths).to(device)
    roi_heads = RoIHeads(stenosis_num_classes=args.stenosis_classes, in_c=512, hidden=256, dropout=0.1).to(device)

    fe.load_state_dict(ckpt["fe"])
    rpn.load_state_dict(ckpt["rpn"])
    roi_heads.load_state_dict(ckpt["roi_heads"])

    fe.eval(); rpn.eval(); roi_heads.eval()

    # Confusion matrices (with BG)
    K = args.stenosis_classes   # stenosis classes: 0..K-1
    P = 4                       # plaque: 0..3 (0 means none/bg)
    cm_sten = np.zeros((K + 1, K + 1), dtype=np.int64)
    cm_plaq = np.zeros((P + 1, P + 1), dtype=np.int64)

    # Metrics accumulators
    total_gt = 0
    total_matched = 0
    total_fp = 0
    sum_iou = 0.0
    n_iou = 0
    # Score separation accumulators (proposal scores)
    tp_scores_all = []   # scores for matched preds (TP)
    fp_scores_all = []   # scores for unmatched preds (FP)
    tp_ious_all   = []   # IoU of matched preds (for sanity)


    # CSV rows per sample (optional helpful)
    per_sample_rows = []
    png_saved = 0

    for batch_i, (x, targets) in enumerate(loader):
        x = x.to(device, non_blocking=True)  # [B,1,D,H,W]
        B = x.shape[0]
        D = x.shape[2]

        feat = fe(x)  # [B,512,Lf]
        _, _, Lf = feat.shape

        obj_logits, deltas, anchors = rpn(feat)

        proposals_feat, scores_list = rpn.propose(
            obj_logits, deltas, anchors, Lf=Lf,
            score_thresh=args.score_thresh,
            pre_nms_topk=args.pre_nms_topk,
            post_nms_topk=args.post_nms_topk,
            iou_thresh=args.nms_iou,
            allow_fallback=False,
        )

        # -------------------------
        # DEBUG: GT widths + anchor pos/neg + score stats
        # -------------------------
        if not hasattr(main, "_dbg_printed"):
            main._dbg_printed = 0

        if main._dbg_printed < 10:  # print first 10 batches only
            stride = float(ckpt.get("stride", 16.0))  # already in your file, but safe

            # 1) GT width distribution (slice + feature units)
            gt_w_slice_all = []
            gt_w_feat_all = []
            for b in range(B):
                gt = targets[b]["boxes"].to(device).float()  # slice coords
                if gt.numel() == 0:
                    continue
                w_slice = (gt[:, 1] - gt[:, 0]).clamp(min=0)
                w_feat = w_slice / stride
                gt_w_slice_all.append(w_slice.detach().cpu().numpy())
                gt_w_feat_all.append(w_feat.detach().cpu().numpy())

            if len(gt_w_slice_all):
                gt_w_slice_all = np.concatenate(gt_w_slice_all)
                gt_w_feat_all  = np.concatenate(gt_w_feat_all)
                print("[GT width slice]", fmt_pct(pct(gt_w_slice_all)))
                print("[GT width feat ]", fmt_pct(pct(gt_w_feat_all)))

            # 2) Anchor labels + pos/neg score separation
            # We need the same target-assign function used in training:
            # from RPN import assign_rpn_targets_1d
            from RPN import assign_rpn_targets_1d

            scores = torch.sigmoid(obj_logits)  # [B,N]
            pos_counts = []
            neg_counts = []
            ign_counts = []
            pos_scores_all = []
            neg_scores_all = []

            for b in range(B):
                gt = targets[b]["boxes"].to(device).float() / stride  # feature coords
                labels, _ = assign_rpn_targets_1d(
                    anchors, gt,
                    pos_iou_thresh=0.5,
                    neg_iou_thresh=0.1,
                )
                pos = (labels == 1)
                neg = (labels == 0)
                ign = (labels == -1)

                pos_counts.append(int(pos.sum().item()))
                neg_counts.append(int(neg.sum().item()))
                ign_counts.append(int(ign.sum().item()))

                if pos.any():
                    pos_scores_all.append(scores[b][pos].detach().cpu().numpy())
                if neg.any():
                    neg_scores_all.append(scores[b][neg].detach().cpu().numpy())

            if len(pos_scores_all):
                pos_scores_all = np.concatenate(pos_scores_all)
            else:
                pos_scores_all = np.array([], dtype=np.float32)

            if len(neg_scores_all):
                neg_scores_all = np.concatenate(neg_scores_all)
            else:
                neg_scores_all = np.array([], dtype=np.float32)

            print(f"[ANCHORS] pos avg={np.mean(pos_counts):.2f} neg avg={np.mean(neg_counts):.2f} ign avg={np.mean(ign_counts):.2f} (N={anchors.shape[0]})")
            print("[SCORES pos]", fmt_pct(pct(pos_scores_all, ps=(0,10,50,90,95,99,100))))
            print("[SCORES neg]", fmt_pct(pct(neg_scores_all, ps=(0,10,50,90,95,99,100))))
            if neg_scores_all.size:
                print("[NEG frac >0.05]", float((neg_scores_all > 0.05).mean()))
                print("[NEG frac >0.5 ]", float((neg_scores_all > 0.5).mean()))

            main._dbg_printed += 1

        pooled, batch_ix, rois_cat = roi_pool_1d(feat, proposals_feat, roi_len=args.roi_len)

        # Build a scores tensor aligned with rois_cat (same order as roi_pool concatenation)
        scores_cat = []
        for b in range(B):
            sb = scores_list[b]
            if sb is None or len(sb) == 0:
                continue
            sb = torch.as_tensor(sb, device=device, dtype=torch.float32)
            scores_cat.append(sb)
        if len(scores_cat) > 0:
            scores_cat = torch.cat(scores_cat, dim=0)
        else:
            scores_cat = torch.empty((0,), device=device, dtype=torch.float32)

        
        # scores_cat should align 1-to-1 with roi_pool concatenation
        if scores_cat.numel() != rois_cat.shape[0]:
            raise RuntimeError(
                f"scores_cat mismatch: scores_cat={scores_cat.numel()} vs rois_cat={rois_cat.shape[0]}"
            )

        # If no proposals at all: everything is FN for non-empty GT; nothing to classify
        if pooled.numel() == 0:
            for b in range(B):
                gt_boxes = targets[b]["boxes"].float()
                M = int(gt_boxes.shape[0])
                total_gt += M
                if M > 0:
                    # all FN: GT -> BG
                    gt_sten = targets[b]["stenosis"].long().to(device).clamp(0, K - 1)
                    gt_plaq = targets[b]["plaque"].long().clamp(0, 3)              # 1..3
                    update_cm_with_bg(cm_sten, gt_sten.cpu().numpy(), np.zeros((0,), dtype=np.int64),
                                      torch.empty((0,), dtype=torch.long), torch.full((M,), -1, dtype=torch.long))
                    update_cm_with_bg(cm_plaq, gt_plaq.cpu().numpy(), np.zeros((0,), dtype=np.int64),
                                      torch.empty((0,), dtype=torch.long), torch.full((M,), -1, dtype=torch.long))
            continue 

        out = roi_heads(pooled, rois_cat, Lf=Lf)
        
        plq_probs = torch.sigmoid(out["plaque_logits"])  # [R,2]
        p_calc = plq_probs[:, 0]
        p_nonc = plq_probs[:, 1]

        # probability of "any plaque" (better than max): 1 - P(none) = 1 - (1-pc)(1-pn)
        p_lesion = 1.0 - (1.0 - p_calc) * (1.0 - p_nonc)  # [R]



        # Pred labels
        pred_sten = torch.argmax(out["stenosis_logits"], dim=1).long()  # [R] in 0..K-1
        pred_plaq = plaque_logits_to_class(out["plaque_logits"], thr=args.plaque_bit_thr).long()

        # Choose boxes to match: refined boxes (better final behavior)
        pred_boxes_feat = out["roi_refined"].float()      # [R,2] in feature coords
        pred_boxes_s = pred_boxes_feat * float(stride)    # slice coords

        # Split by batch
        for b in range(B):
            ridx = torch.where(batch_ix == b)[0]
            if ridx.numel() == 0:
                # No predictions for this sample -> all FN if GT exists
                gt_boxes = targets[b]["boxes"].float()
                M = int(gt_boxes.shape[0])
                total_gt += M
                if M > 0:
                    gt_sten = targets[b]["stenosis"].long().to(device).clamp(0, K - 1)
                    gt_plaq = targets[b]["plaque"].long().clamp(0, 3)
                    # all FN
                    pred2gt = torch.empty((0,), dtype=torch.long)
                    gt2pred = torch.full((M,), -1, dtype=torch.long)
                    update_cm_with_bg(cm_sten, gt_sten.cpu().numpy(), np.zeros((0,), dtype=np.int64), pred2gt, gt2pred)
                    update_cm_with_bg(cm_plaq, gt_plaq.cpu().numpy(), np.zeros((0,), dtype=np.int64), pred2gt, gt2pred)
                continue

            pb_all = pred_boxes_s[ridx].clamp(0, float(D)).contiguous()
            rpn_s  = scores_cat[ridx]
            lesion_s = p_lesion[ridx]

            # final score
            if args.use_combined_score:
                ps_all = rpn_s * lesion_s
            else:
                ps_all = lesion_s

            # FILTER: drop background-ish RoIs
            keep = torch.where(ps_all >= args.plaque_score_thresh)[0]

            pb = pb_all[keep]
            ps = ps_all[keep]
            y_sten = pred_sten[ridx][keep]
            y_plaq = pred_plaq[ridx][keep]

            # optional: final NMS after refinement (recommended)
            if args.final_nms and pb.numel() > 0:
                keep_nms = nms_1d_eval(pb, ps, iou_thresh=args.final_nms_iou)
                pb = pb[keep_nms]
                ps = ps[keep_nms]
                y_sten = y_sten[keep_nms]
                y_plaq = y_plaq[keep_nms]


            gt_boxes = targets[b]["boxes"].float().to(device)  # slice coords already
            M = int(gt_boxes.shape[0])
            total_gt += M

            # Remap GT stenosis from 1..5 -> 0..4 (your dataset has 1..5 when non-empty)
            if M > 0:
                gt_sten = targets[b]["stenosis"].long().to(device).clamp(0, K - 1)
                gt_plaq = targets[b]["plaque"].long().to(device).clamp(0, 3)
            else:
                gt_sten = torch.empty((0,), dtype=torch.long, device=device)
                gt_plaq = torch.empty((0,), dtype=torch.long, device=device)

            # Match preds to GT for lesion-level evaluation
            pred2gt, gt2pred, matched_ious = greedy_match_1d(pb, ps, gt_boxes, iou_thr=args.match_iou)

                        # ---- score separation logging ----
            if ps.numel() > 0:
                tp_mask = (pred2gt != -1)
                fp_mask = ~tp_mask

                if tp_mask.any():
                    tp_scores_all.append(ps[tp_mask].detach().cpu().numpy())
                    tp_ious_all.append(matched_ious[tp_mask].detach().cpu().numpy())

                if fp_mask.any():
                    fp_scores_all.append(ps[fp_mask].detach().cpu().numpy())


            # Update confusion matrices (+BG)
            update_cm_with_bg(
                cm_sten,
                gt_sten.detach().cpu().numpy(),
                y_sten.detach().cpu().numpy(),
                pred2gt, gt2pred
            )
            update_cm_with_bg(
                cm_plaq,
                gt_plaq.detach().cpu().numpy(),
                y_plaq.detach().cpu().numpy(),
                pred2gt, gt2pred
            )

            # Aggregate metrics
            num_matched = int((pred2gt != -1).sum().item())
            num_fp = int((pred2gt == -1).sum().item())
            total_matched += num_matched
            total_fp += num_fp

            # per-sample row
            meta = targets[b].get("patient_id", "") + "_" + targets[b].get("vessel", "")
            miou_mean = 0.0
            if num_matched > 0:
                miou = matched_ious[pred2gt != -1]
                miou_mean = float(miou.mean().item())
                sum_iou += float(miou.sum().item())
                n_iou += int(miou.numel())

            per_sample_rows.append({
                "sample": meta,
                "gt_count": M,
                "pred_count": int(pb.shape[0]),
                "matched": num_matched,
                "fp": num_fp,
                "fn": int(M - int((gt2pred != -1).sum().item())),
                "mean_iou_matched": miou_mean,
            })


            # PNG
            if args.save_png and png_saved < args.max_png:
                title = f"{args.split} {meta} | GT={M} Pred={int(pb.shape[0])} Match={num_matched} FP={num_fp}"
                out_png = os.path.join(png_dir, f"{batch_i:04d}_{b:02d}_{meta}.png")

                x_cpu = x[b, 0].detach().cpu()  # [D,H,W]
                save_viz_grayproj(
                    out_png,
                    x_cpu,
                    gt_boxes.detach().cpu(),
                    pb.detach().cpu(),
                    ps.detach().cpu(),
                    y_sten.detach().cpu(),
                    y_plaq.detach().cpu(),
                    title=title
                )
                png_saved += 1

    # Summary metrics
    det_recall = (total_matched / max(1, total_gt))
    fp_per_sample = (total_fp / max(1, len(ds)))
    mean_iou = (sum_iou / max(1, n_iou))

    sten_prec, sten_rec, sten_f1, sten_macro = cm_stats(cm_sten)
    plaq_prec, plaq_rec, plaq_f1, plaq_macro = cm_stats(cm_plaq)

    # -------------------------
    # Score separation summary (TP vs FP)
    # -------------------------
    def score_row(name: str, arr: np.ndarray):
        d = pct(arr, ps=(0, 10, 50, 90, 95, 99, 100))
        row = {
            "group": name,
            "n": int(arr.size),
            "mean": float(arr.mean()) if arr.size else 0.0,
            "std": float(arr.std()) if arr.size else 0.0,
            "p0": d[0], "p10": d[10], "p50": d[50], "p90": d[90], "p95": d[95], "p99": d[99], "p100": d[100],
            "frac_gt_0.001": float((arr > 0.001).mean()) if arr.size else 0.0,
            "frac_gt_0.01":  float((arr > 0.01).mean()) if arr.size else 0.0,
            "frac_gt_0.05":  float((arr > 0.05).mean()) if arr.size else 0.0,
            "frac_gt_0.1":   float((arr > 0.1).mean()) if arr.size else 0.0,
        }
        return row

    tp_scores = np.concatenate(tp_scores_all) if len(tp_scores_all) else np.array([], dtype=np.float32)
    fp_scores = np.concatenate(fp_scores_all) if len(fp_scores_all) else np.array([], dtype=np.float32)
    tp_ious   = np.concatenate(tp_ious_all)   if len(tp_ious_all)   else np.array([], dtype=np.float32)

    rows = []
    rows.append(score_row("TP_matched", tp_scores))
    rows.append(score_row("FP_unmatched", fp_scores))
    rows.append(score_row("TP_IoU", tp_ious))  # just for sanity; should be >= match_iou mostly

    # Simple “separation” scalars you can track
    sep = {
        "tp_p50_minus_fp_p95": float(np.percentile(tp_scores, 50) - np.percentile(fp_scores, 95)) if (tp_scores.size and fp_scores.size) else 0.0,
        "tp_p10_minus_fp_p90": float(np.percentile(tp_scores, 10) - np.percentile(fp_scores, 90)) if (tp_scores.size and fp_scores.size) else 0.0,
        "tp_mean_minus_fp_mean": float(tp_scores.mean() - fp_scores.mean()) if (tp_scores.size and fp_scores.size) else 0.0,
    }

    score_csv = os.path.join(out_dir, "score_separation.csv")
    with open(score_csv, "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    sep_csv = os.path.join(out_dir, "score_separation_summary.csv")
    with open(sep_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sep.keys()))
        w.writeheader()
        w.writerow(sep)

    print("\n[Score separation]")
    print("  TP scores:", fmt_pct(pct(tp_scores)))
    print("  FP scores:", fmt_pct(pct(fp_scores)))
    print("  Saved:", score_csv)
    print("  Saved:", sep_csv)


    # Save outputs
    def save_cm_csv(path, cm, labels):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["GT\\Pred"] + labels)
            for i, row in enumerate(cm):
                w.writerow([labels[i]] + list(row))

    sten_labels = [f"sten_{i}" for i in range(K)] + ["BG"]
    plaq_labels = ["plq_0none", "plq_1noncalc", "plq_2mixed", "plq_3calc"] + ["BG"]

    save_cm_csv(os.path.join(out_dir, "cm_stenosis.csv"), cm_sten, sten_labels)
    save_cm_csv(os.path.join(out_dir, "cm_plaque.csv"), cm_plaq, plaq_labels)

    # Save per-sample CSV
    ps_csv = os.path.join(out_dir, "per_sample.csv")
    if len(per_sample_rows) > 0:
        keys = list(per_sample_rows[0].keys())
        with open(ps_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(per_sample_rows)

    # Save summary
    summary = {
        "ckpt": args.ckpt,
        "split": args.split,
        "match_iou": args.match_iou,
        "total_gt": total_gt,
        "total_matched": total_matched,
        "det_recall": det_recall,
        "total_fp": total_fp,
        "fp_per_sample": fp_per_sample,
        "mean_iou_matched": mean_iou,
        **{f"sten_precision_c{i}": float(sten_prec[i]) for i in range(K)},
        **{f"sten_recall_c{i}": float(sten_rec[i]) for i in range(K)},
        **{f"sten_f1_c{i}": float(sten_f1[i]) for i in range(K)},
        **sten_macro,
        **{f"plaq_precision_c{i}": float(plaq_prec[i]) for i in range(4)},
        **{f"plaq_recall_c{i}": float(plaq_rec[i]) for i in range(4)},
        **{f"plaq_f1_c{i}": float(plaq_f1[i]) for i in range(4)},
        **plaq_macro,
    }

    sum_csv = os.path.join(out_dir, "summary.csv")
    with open(sum_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    print("\n=== Lesion-level Test Summary ===")
    print("GT lesions:", total_gt)
    print("Matched (TP):", total_matched)
    print("Detection recall (TP/GT):", f"{det_recall:.4f}")
    print("Total FP preds:", total_fp)
    print("FP per sample:", f"{fp_per_sample:.4f}")
    print("Mean IoU (matched):", f"{mean_iou:.4f}")
    print("\nStenosis macro:", sten_macro)
    print("Plaque macro:", plaq_macro)
    print("\nSaved:")
    print("  ", sum_csv)
    print("  ", os.path.join(out_dir, "cm_stenosis.csv"))
    print("  ", os.path.join(out_dir, "cm_plaque.csv"))
    if args.save_png:
        print("  ", png_dir)


if __name__ == "__main__":
    main()



