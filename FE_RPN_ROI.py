import os
import csv
import random
import argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from FE import FeatureExtractorFE
from RPN import RPN1D, rpn_loss_1d, interval_iou_1d, assign_rpn_targets_1d
from roi_pool import roi_pool_1d
from heads import RoIHeads

from cached_ds import CachedWindowDataset, collate_fn


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def append_metrics_csv(csv_path: str, header: list, row: dict):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def _pct_np(arr, ps=(50, 90, 99)):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return {p: np.nan for p in ps}
    return {p: float(np.percentile(arr, p)) for p in ps}


class DebugEpochAcc:
    """
    Accumulates training-time debug stats and writes 1 row per epoch.
    Keep arrays small by only storing scalars (counts, sampled scores, etc).
    """
    def __init__(self, score_thresh: float):
        self.score_thresh = float(score_thresh)
        self.reset()

    def reset(self):
        # anchor label counts
        self.pos_counts = []
        self.neg_counts = []
        self.ign_counts = []

        # score distributions (pos/neg anchors)
        self.pos_scores = []
        self.neg_scores = []

        # delta magnitudes (RPN)
        self.tc_abs = []
        self.tw = []

        # proposal stage counts
        self.keep_after_thresh = []
        self.keep_after_nms = []
        self.keep_after_topk = []

        # proposal quality vs GT
        self.prop_max_iou = []          # max IoU per proposal (after NMS)
        self.prop_frac_iou_ge = []      # per-image fraction of proposals with IoU >= match_iou

        self.gt_max_iou = []   # max IoU per GT (after NMS)
        self.gt_recall = []    # per-image recall at match_iou
                
        # roi counts
        self.rois_total = 0
        self.pos_rois_total = 0
        self.num_batches = 0
        self.num_windows = 0


    def add_proposal_stats(
        self,
        scores_b: torch.Tensor,          # [N] sigmoid scores for this image (ALL anchors)
        pre_nms_topk: int,
        proposals_after_nms: torch.Tensor,  # [P,2] FEATURE coords
        gt_boxes_feat: torch.Tensor,        # [G,2] FEATURE coords
        match_iou: float,
    ):
        """
        Logs proposal pipeline counts consistent with: topk -> threshold -> nms.
        Also logs IoU quality of final proposals vs GT.
        """
        scores_b = scores_b.detach()

        # (1) after topk
        k = min(int(pre_nms_topk), scores_b.numel())
        topk_scores, _ = torch.topk(scores_b, k)
        self.keep_after_topk.append(int(k))

        # (2) after threshold (applied AFTER topk)
        n_thr = int((topk_scores >= self.score_thresh).sum().item())
        self.keep_after_thresh.append(n_thr)

        # (3) after nms (already done by propose)
        P = int(proposals_after_nms.shape[0]) if proposals_after_nms is not None else 0
        self.keep_after_nms.append(P)

        # ---- IoU quality: proposals(after nms) vs GT ----
        if P == 0:
            self.prop_frac_iou_ge.append(np.nan)
            if gt_boxes_feat.numel() > 0:
                self.gt_recall.append(0.0)
            return
        if gt_boxes_feat.numel() == 0:
            self.prop_frac_iou_ge.append(np.nan)
            return

        ious = interval_iou_1d(proposals_after_nms, gt_boxes_feat)  # [P,G]
        max_iou = ious.max(dim=1).values                             # [P]
        max_iou_per_gt = ious.max(dim=0).values  # [G]
        self.gt_max_iou.append(max_iou_per_gt.detach().cpu().numpy())
        self.gt_recall.append(float((max_iou_per_gt >= float(match_iou)).float().mean().item()))
        self.prop_max_iou.append(max_iou.detach().cpu().numpy())

        frac_ge = float((max_iou >= float(match_iou)).float().mean().item())
        self.prop_frac_iou_ge.append(frac_ge)

    def add_anchor_stats(self, labels, obj_logits_b, deltas_b, sample_cap=4000):
        """
        labels: [N] in {-1,0,1}
        obj_logits_b: [N]
        deltas_b: [N,2]
        """
        labels = labels.detach()
        scores = torch.sigmoid(obj_logits_b.detach())

        pos = (labels == 1)
        neg = (labels == 0)
        ign = (labels == -1)

        self.pos_counts.append(int(pos.sum().item()))
        self.neg_counts.append(int(neg.sum().item()))
        self.ign_counts.append(int(ign.sum().item()))

        # sample some scores to avoid huge memory
        # (random subset from pos and neg)
        def _sample(x):
            x = x.flatten()
            if x.numel() <= sample_cap:
                return x
            idx = torch.randperm(x.numel(), device=x.device)[:sample_cap]
            return x[idx]

        if pos.any():
            self.pos_scores.append(_sample(scores[pos]).cpu().numpy())
        if neg.any():
            self.neg_scores.append(_sample(scores[neg]).cpu().numpy())

        # delta magnitudes (only for positives is also fine; but logging all is ok)
        if pos.any():
            tc = deltas_b[pos, 0].detach()
            tw = deltas_b[pos, 1].detach()
            self.tc_abs.append(_sample(tc.abs()).cpu().numpy())
            self.tw.append(_sample(tw).cpu().numpy())

    def add_roi_counts(self, num_rois, num_pos_rois):
        self.rois_total += int(num_rois)
        self.pos_rois_total += int(num_pos_rois)

    def bump_batch(self, B):
        self.num_batches += 1
        self.num_windows += int(B)

    def summarize_row(self, epoch: int, lr: float):
        # concat arrays
        pos_scores = np.concatenate(self.pos_scores) if len(self.pos_scores) else np.array([], np.float32)
        neg_scores = np.concatenate(self.neg_scores) if len(self.neg_scores) else np.array([], np.float32)
        tc_abs = np.concatenate(self.tc_abs) if len(self.tc_abs) else np.array([], np.float32)
        tw = np.concatenate(self.tw) if len(self.tw) else np.array([], np.float32)

        pos_p = _pct_np(pos_scores, (50, 90, 99))
        neg_p = _pct_np(neg_scores, (50, 90, 99))
        tc_p = _pct_np(tc_abs, (50, 90, 99))
        tw_p = _pct_np(tw, (50, 90, 99))

        neg_frac_gt_thr = float(np.mean(neg_scores >= self.score_thresh)) if neg_scores.size else np.nan
        prop_max_iou = np.concatenate(self.prop_max_iou) if len(self.prop_max_iou) else np.array([], np.float32)
        iou_p = _pct_np(prop_max_iou, (50, 90, 99))

        row = {
            "epoch": epoch,
            "lr": float(lr),

            "anchors_pos_mean": float(np.mean(self.pos_counts)) if self.pos_counts else np.nan,
            "anchors_neg_mean": float(np.mean(self.neg_counts)) if self.neg_counts else np.nan,
            "anchors_ign_mean": float(np.mean(self.ign_counts)) if self.ign_counts else np.nan,

            "pos_score_p50": pos_p[50], "pos_score_p90": pos_p[90], "pos_score_p99": pos_p[99],
            "neg_score_p50": neg_p[50], "neg_score_p90": neg_p[90], "neg_score_p99": neg_p[99],
            "neg_frac_ge_score_thresh": neg_frac_gt_thr,

            "tc_abs_p50": tc_p[50], "tc_abs_p90": tc_p[90], "tc_abs_p99": tc_p[99],
            "tw_p50": tw_p[50], "tw_p90": tw_p[90], "tw_p99": tw_p[99],

            "props_after_thresh_mean": float(np.mean(self.keep_after_thresh)) if self.keep_after_thresh else np.nan,
            "props_after_topk_mean": float(np.mean(self.keep_after_topk)) if self.keep_after_topk else np.nan,
            "props_after_nms_mean": float(np.mean(self.keep_after_nms)) if self.keep_after_nms else np.nan,
            
            "prop_max_iou_p50": iou_p[50],
            "prop_max_iou_p90": iou_p[90],
            "prop_max_iou_p99": iou_p[99],

            "rois_total": int(self.rois_total),
            "pos_rois_total": int(self.pos_rois_total),
            "pos_roi_frac": float(self.pos_rois_total / max(1, self.rois_total)),
            "num_windows": int(self.num_windows),
            "num_batches": int(self.num_batches),
        }
        vals = np.asarray(self.prop_frac_iou_ge, dtype=np.float32)
        row["prop_frac_iou_ge_mean"] = float(np.mean(vals[~np.isnan(vals)])) if np.any(~np.isnan(vals)) else np.nan
        gt_max_iou = np.concatenate(self.gt_max_iou) if len(self.gt_max_iou) else np.array([], np.float32)
        gt_p = _pct_np(gt_max_iou, (50, 90, 99))

        row["gt_max_iou_p50"] = gt_p[50]
        row["gt_max_iou_p90"] = gt_p[90]
        row["gt_max_iou_p99"] = gt_p[99]
        row["gt_recall_mean"] = float(np.mean(self.gt_recall)) if len(self.gt_recall) else np.nan
        return row
    
@torch.no_grad()
def assign_rois_to_gt_1d(rois_feat: torch.Tensor, gt_boxes_feat: torch.Tensor,
                        pos_iou_thresh=0.5, neg_iou_thresh=0.1):
    """
    rois_feat: [R,2] proposals in FEATURE coords
    gt_boxes_feat: [G,2] GT in FEATURE coords

    Returns:
      labels: [R] {1 pos, 0 neg, -1 ignore}
      matched_idx: [R] index of best GT
      matched_iou: [R]
    """
    R = rois_feat.shape[0]
    device = rois_feat.device

    labels = torch.full((R,), -1, dtype=torch.long, device=device)
    matched_idx = torch.zeros((R,), dtype=torch.long, device=device)
    matched_iou = torch.zeros((R,), dtype=torch.float32, device=device)

    if R == 0:
        return labels, matched_idx, matched_iou

    if gt_boxes_feat.numel() == 0:
        labels[:] = 0
        return labels, matched_idx, matched_iou

    iou = interval_iou_1d(rois_feat, gt_boxes_feat)  # [R,G]
    matched_iou, matched_idx = iou.max(dim=1)

    labels[matched_iou < neg_iou_thresh] = 0
    labels[matched_iou >= pos_iou_thresh] = 1
    return labels, matched_idx, matched_iou


def sample_rois(labels: torch.Tensor, batch_size=128, pos_fraction=0.5):
    device = labels.device
    pos_idx = torch.where(labels == 1)[0]
    neg_idx = torch.where(labels == 0)[0]

    num_pos = int(batch_size * pos_fraction)
    num_pos = min(num_pos, pos_idx.numel())
    num_neg = batch_size - num_pos
    num_neg = min(num_neg, neg_idx.numel())

    if num_pos > 0:
        perm = torch.randperm(pos_idx.numel(), device=device)[:num_pos]
        pos_keep = pos_idx[perm]
    else:
        pos_keep = torch.empty((0,), dtype=torch.long, device=device)

    if num_neg > 0:
        perm = torch.randperm(neg_idx.numel(), device=device)[:num_neg]
        neg_keep = neg_idx[perm]
    else:
        neg_keep = torch.empty((0,), dtype=torch.long, device=device)

    return torch.cat([pos_keep, neg_keep], dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out_root", type=str, default="DET_results")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)

    # RPN thresholds + proposal params
    ap.add_argument("--pos_iou_thresh", type=float, default=0.4)
    ap.add_argument("--neg_iou_thresh", type=float, default=0.1)
    ap.add_argument("--score_thresh", type=float, default=0.1)
    ap.add_argument("--pre_nms_topk", type=int, default=600)
    ap.add_argument("--post_nms_topk", type=int, default=100)
    ap.add_argument("--nms_iou", type=float, default=0.5)
    ap.add_argument("--anchor_lengths", type=str, default="1,2,3,5,7")

    # RoI pooling + RoI sampling
    ap.add_argument("--roi_len", type=int, default=16)
    ap.add_argument("--roi_pos_iou", type=float, default=0.5)
    ap.add_argument("--roi_neg_iou", type=float, default=0.1)
    ap.add_argument("--roi_sample_size", type=int, default=128)
    ap.add_argument("--roi_pos_fraction", type=float, default=0.5)

    # stenosis classes
    ap.add_argument("--stenosis_classes", type=int, required=True,
                    help="number of stenosis classes (K). targets[b]['stenosis'] must be in [0..K-1].")
    ap.add_argument(
        "--stenosis_class_weights",
        type=str,
        default="",
        help="Comma-separated weights for stenosis CE, length = stenosis_classes. Example: 1,1,3"
    )

    # loss weights
    ap.add_argument("--w_rpn", type=float, default=1.0)
    ap.add_argument("--w_plaque", type=float, default=1.0)
    ap.add_argument("--w_stenosis", type=float, default=1.0)
    ap.add_argument("--w_roi_reg", type=float, default=1.0)
    ap.add_argument("--plaque_on_pos_only", action="store_true",
                    help="compute plaque loss only on positive RoIs (ignore background)")

    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--no_pbar", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = (args.run_name.strip() + "_") if args.run_name.strip() else ""
    out_root = os.path.abspath(args.out_root)
    out_dir = ensure_dir(os.path.join(out_root, f"{prefix}fe_rpn_roi_{run_stamp}"))
    ckpt_dir = ensure_dir(os.path.join(out_dir, "checkpoints"))
    metrics_csv = os.path.join(out_dir, "metrics.csv")
    debug_csv = os.path.join(out_dir, "debug_train.csv")
    debug_header = [
        "epoch","lr",
        "anchors_pos_mean","anchors_neg_mean","anchors_ign_mean",
        "pos_score_p50","pos_score_p90","pos_score_p99",
        "neg_score_p50","neg_score_p90","neg_score_p99",
        "neg_frac_ge_score_thresh",
        "tc_abs_p50","tc_abs_p90","tc_abs_p99",
        "tw_p50","tw_p90","tw_p99",

        "props_after_topk_mean",
        "props_after_thresh_mean",
        "props_after_nms_mean",

        "prop_max_iou_p50","prop_max_iou_p90","prop_max_iou_p99",
        "prop_frac_iou_ge_mean",

        "rois_total","pos_rois_total","pos_roi_frac",
        "num_samples","num_batches",
        "gt_max_iou_p50","gt_max_iou_p90","gt_max_iou_p99","gt_recall_mean",
    ]

    anchor_lengths = tuple(int(x) for x in args.anchor_lengths.split(",") if x.strip())
    stride = 16.0  # must match FE downsample along D

    print("Outputs:", out_dir)
    print("Anchor lengths:", anchor_lengths, "stride:", stride, "stenosis_classes:", args.stenosis_classes)

    stenosis_ce_w = None
    if args.stenosis_class_weights.strip():
        w = [float(x) for x in args.stenosis_class_weights.split(",")]
        if len(w) != args.stenosis_classes:
            raise ValueError(f"--stenosis_class_weights must have length {args.stenosis_classes}, got {len(w)}")
        stenosis_ce_w = torch.tensor(w, dtype=torch.float32, device=device)


    ds_train = CachedWindowDataset(args.data, "train")


    train_loader = DataLoader(
        ds_train, batch_size=args.bs, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn
    )

    fe = FeatureExtractorFE(in_ch=1, norm="gn").to(device)
    rpn = RPN1D(in_c=512, anchor_lengths=anchor_lengths).to(device)
    roi_heads = RoIHeads(stenosis_num_classes=args.stenosis_classes, in_c=512, hidden=256, dropout=0.1).to(device)

    params = list(fe.parameters()) + list(rpn.parameters()) + list(roi_heads.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    header = [
        "epoch",
        "train_loss_total",
        "train_loss_rpn",
        "train_loss_roi",
        "train_loss_plaque",
        "train_loss_stenosis",
        "train_loss_roi_reg",
        "train_num_pos_rois",
        "lr",
        "ckpt_path",
    ]

    for epoch in range(1, args.epochs + 1):
        fe.train(); rpn.train(); roi_heads.train()

        dbg = DebugEpochAcc(score_thresh=args.score_thresh)

        it = train_loader
        if not args.no_pbar:
            it = tqdm(train_loader, desc=f"Train E{epoch:03d}", leave=False, dynamic_ncols=True)

        run_total = 0.0
        run_rpn = 0.0
        run_roi = 0.0
        run_plq = 0.0
        run_ste = 0.0
        run_rrg = 0.0
        run_pos_rois = 0
        nb = 0

        for (x, targets) in it:
            x = x.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                feat = fe(x)                      # [B,512,Lf]
                B, C, Lf = feat.shape

                # ---- RPN ----
                obj_logits, deltas, anchors = rpn(feat)
                # --- DEBUG: anchor label + score separation ---
                # NOTE: this is for logging only; doesn't affect training.
                _pos_rois_sampled = 0
                _rois_sampled = 0
                for b in range(B):
                    gt_boxes_slice = targets[b]["boxes"].to(device).float()
                    gt_boxes_feat = gt_boxes_slice / float(stride)
                    labels_b, _ = assign_rpn_targets_1d(
                        anchors, gt_boxes_feat,
                        pos_iou_thresh=args.pos_iou_thresh,
                        neg_iou_thresh=args.neg_iou_thresh,
                    )
                    dbg.add_anchor_stats(labels_b, obj_logits[b], deltas[b])

                loss_rpn, _rpn_stats = rpn_loss_1d(
                    obj_logits, deltas, anchors,
                    targets,
                    stride=stride,
                    pos_iou_thresh=args.pos_iou_thresh,
                    neg_iou_thresh=args.neg_iou_thresh,
                    sample_size=256,
                    pos_fraction=0.5,
                    reg_weight=1.0,
                )
                has_gt = [t["boxes"].numel() > 0 for t in targets]

                # ---- Proposals ----
                proposals_feat, scores_list = rpn.propose(
                    obj_logits, deltas, anchors, Lf=Lf,
                    score_thresh=args.score_thresh,
                    pre_nms_topk=args.pre_nms_topk,
                    post_nms_topk=args.post_nms_topk,
                    iou_thresh=args.nms_iou,
                    has_gt=has_gt,                 # <-- NEW: list[bool] length B (or None)
                    fallback_topk=100,           # <-- NEW: small fallback size for GT windows
                    allow_fallback=False,
                )

                scores_sig = torch.sigmoid(obj_logits)  # [B,N]
                for b in range(B):
                    gt_boxes_feat = targets[b]["boxes"].to(device).float() / float(stride)  # [G,2] feature coords
                    props_b = proposals_feat[b] if proposals_feat[b] is not None else torch.empty((0,2), device=device)

                    dbg.add_proposal_stats(
                        scores_b=scores_sig[b],
                        pre_nms_topk=args.pre_nms_topk,
                        proposals_after_nms=props_b,
                        gt_boxes_feat=gt_boxes_feat,
                        match_iou=args.roi_pos_iou,   # or args.pos_iou_thresh; pick ONE definition you care about
                    )

                # ---- RoI Pool ----
                pooled, batch_ix, rois_cat = roi_pool_1d(feat, proposals_feat, roi_len=args.roi_len)
                # pooled: [R,512,roi_len], rois_cat: [R,2] feat coords

                if pooled.numel() == 0:
                    loss_roi = torch.zeros((), device=device)
                    roi_stats = {"loss_plaque": 0.0, "loss_stenosis": 0.0, "loss_roi_reg": 0.0, "num_pos_rois": 0}
                    loss = args.w_rpn * loss_rpn
                    dbg.add_roi_counts(num_rois=0, num_pos_rois=0)
                    dbg.bump_batch(B)
                else:
                    # Build matched tensors for all pooled RoIs, then subset by sampled indices
                    R = rois_cat.shape[0]
                    matched_gt = torch.zeros((R, 2), dtype=torch.float32, device=device)
                    matched_plaque = torch.zeros((R,), dtype=torch.long, device=device)      # 0 for bg
                    matched_sten = torch.zeros((R,), dtype=torch.long, device=device)        # dummy for bg

                    keep_all = []

                    for b in range(B):
                        ridx = torch.where(batch_ix == b)[0]
                        if ridx.numel() == 0:
                            continue

                        gt_boxes_s = targets[b]["boxes"].to(device).float()       # [G,2] slice coords
                        gt_plq = targets[b]["plaque"].to(device).long()           # [G] {1,2,3}
                        gt_ste = targets[b]["stenosis"].to(device).long()         # [G] {0..K-1}

                        gt_boxes_f = gt_boxes_s / float(stride)

                        labels, midx, _ = assign_rois_to_gt_1d(
                            rois_cat[ridx], gt_boxes_f,
                            pos_iou_thresh=args.roi_pos_iou,
                            neg_iou_thresh=args.roi_neg_iou,
                        )

                        keep = sample_rois(labels, batch_size=args.roi_sample_size, pos_fraction=args.roi_pos_fraction)
                        _pos_rois_sampled += int((labels[keep] == 1).sum().item())
                        _rois_sampled += int(keep.numel())
                        if keep.numel() == 0:
                            continue

                        ridx_keep = ridx[keep]
                        keep_all.append(ridx_keep)

                        # fill for kept rois
                        pos_mask = (labels[keep] == 1)

                        if gt_boxes_f.numel() > 0:
                            matched_gt[ridx_keep] = gt_boxes_f[midx[keep]]

                            # positives take GT plaque/stenosis, negatives remain 0 (background/dummy)
                            matched_plaque[ridx_keep] = torch.where(pos_mask, gt_plq[midx[keep]], torch.zeros_like(gt_plq[midx[keep]]))
                            matched_sten[ridx_keep] = torch.where(pos_mask, gt_ste[midx[keep]], torch.zeros_like(gt_ste[midx[keep]]))

                    if len(keep_all) == 0:
                        loss_roi = torch.zeros((), device=device)
                        roi_stats = {"loss_plaque": 0.0, "loss_stenosis": 0.0, "loss_roi_reg": 0.0, "num_pos_rois": 0}
                        loss = args.w_rpn * loss_rpn
                    else:
                        keep_all = torch.cat(keep_all, dim=0)

                        pooled_k = pooled[keep_all]
                        rois_k = rois_cat[keep_all]
                        gt_k = matched_gt[keep_all]
                        plq_k = matched_plaque[keep_all]
                        ste_k = matched_sten[keep_all]

                        out = roi_heads(pooled_k, rois_k, Lf=Lf)
                        loss_roi, roi_stats = RoIHeads.losses(
                            out,
                            rois=rois_k,
                            matched_gt_rois=gt_k,
                            matched_plaque=plq_k,
                            matched_stenosis=ste_k,
                            plaque_w=args.w_plaque,
                            stenosis_w=args.w_stenosis,
                            roi_reg_w=args.w_roi_reg,
                            stenosis_class_weights=stenosis_ce_w,
                            plaque_on_pos_only=args.plaque_on_pos_only,
                        )
                        # num pooled RoIs (before sampling)
                        dbg.add_roi_counts(num_rois=_rois_sampled, num_pos_rois=_pos_rois_sampled)
                        dbg.bump_batch(B)


                        loss = args.w_rpn * loss_rpn + loss_roi

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            nb += 1
            run_total += float(loss.detach().item())
            run_rpn += float(loss_rpn.detach().item())
            run_roi += float(loss_roi.detach().item()) if isinstance(loss_roi, torch.Tensor) else float(loss_roi)
            run_plq += float(roi_stats.get("loss_plaque", 0.0))
            run_ste += float(roi_stats.get("loss_stenosis", 0.0))
            run_rrg += float(roi_stats.get("loss_roi_reg", 0.0))
            run_pos_rois += int(roi_stats.get("num_pos_rois", 0))

            if not args.no_pbar and hasattr(it, "set_postfix"):
                it.set_postfix(
                    total=f"{run_total/max(1,nb):.4f}",
                    rpn=f"{run_rpn/max(1,nb):.4f}",
                    roi=f"{run_roi/max(1,nb):.4f}",
                    pos_rois=int(roi_stats.get("num_pos_rois", 0)),
                )

        row = {
            "epoch": epoch,
            "train_loss_total": run_total / max(1, nb),
            "train_loss_rpn": run_rpn / max(1, nb),
            "train_loss_roi": run_roi / max(1, nb),
            "train_loss_plaque": run_plq / max(1, nb),
            "train_loss_stenosis": run_ste / max(1, nb),
            "train_loss_roi_reg": run_rrg / max(1, nb),
            "train_num_pos_rois": int(run_pos_rois),
            "lr": float(optim.param_groups[0]["lr"]),
            "ckpt_path": "",
        }

        ckpt = {
            "epoch": epoch,
            "fe": fe.state_dict(),
            "rpn": rpn.state_dict(),
            "roi_heads": roi_heads.state_dict(),
            "optimizer": optim.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "args": vars(args),
            "stride": stride,
            "anchor_lengths": anchor_lengths,
            "train_row": row,
        }

        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
        torch.save(ckpt, ckpt_path)
        row["ckpt_path"] = ckpt_path

        append_metrics_csv(metrics_csv, header, row)
        debug_row = dbg.summarize_row(epoch=epoch, lr=float(optim.param_groups[0]["lr"]))
        append_metrics_csv(debug_csv, debug_header, debug_row)

        print(
            f"[DBG] pos_p50={debug_row['pos_score_p50']:.3f} "
            f"neg_p99={debug_row['neg_score_p99']:.3f} "
            f"neg_frac>=thr={debug_row['neg_frac_ge_score_thresh']:.3f} | "
            f"props(topk/thr/nms)={debug_row['props_after_topk_mean']:.1f}/"
            f"{debug_row['props_after_thresh_mean']:.1f}/"
            f"{debug_row['props_after_nms_mean']:.1f} | "
            f"prop_IoU_p50={debug_row['prop_max_iou_p50']:.3f} "
            f"frac_IoU>=thr={debug_row['prop_frac_iou_ge_mean']:.3f} | "
            f"pos_roi_frac={debug_row['pos_roi_frac']:.3f}"
        )

        print(
            f"Epoch {epoch:03d} | "
            f"loss={row['train_loss_total']:.4f} "
            f"(rpn={row['train_loss_rpn']:.4f}, roi={row['train_loss_roi']:.4f}) "
            f"plq={row['train_loss_plaque']:.4f} "
            f"ste={row['train_loss_stenosis']:.4f} "
            f"roiReg={row['train_loss_roi_reg']:.4f} | saved {ckpt_path}"
        )


if __name__ == "__main__":
    main()
