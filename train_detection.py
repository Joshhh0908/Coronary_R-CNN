import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Optional

from dataprep import VesselWindowDataset, collate_fn
from detection_module import (
    FeatureExtractorFE, RPN1D, roi_pool_1d,
    rpn_loss_1d,
    assign_rois_to_gt,
    sample_rois,
    encode_boxes_to_deltas,
    RoIHeads,
    decode_deltas_to_boxes_1d,
    interval_iou_1d,
    nms_1d,
)
@torch.no_grad()
def infer_window_batch(
    fe, rpn, roi_heads,
    x,                       # [B,1,D,H,W]
    window_len: int,
    stride: float,
    roi_len: int,
    rpn_score_thresh: float = 0.0,
    rpn_pre_nms_topk: int = 600,
    rpn_post_nms_topk: int = 20,
    rpn_iou_thresh: float = 0.5,
    final_score_thresh: float = 0.0,
    final_nms_iou: float = 0.5,
    max_dets: int = 100,
    score_mode: str = "rpn",   # "rpn" or "rpn_plaque"
    debug: bool = False
):
    device = x.device
    fe.eval(); rpn.eval(); roi_heads.eval()

    feat = fe(x)  # [B,512,Lf]
    obj_logits, deltas, anchors = rpn(feat)
    Lf = feat.shape[-1]

    proposals_feat, proposals_score = rpn.propose(
        obj_logits, deltas, anchors, Lf=Lf,
        score_thresh=rpn_score_thresh,
        pre_nms_topk=rpn_pre_nms_topk,
        post_nms_topk=rpn_post_nms_topk,
        iou_thresh=rpn_iou_thresh,
    )

    roi_feat, roi_batch, roi_boxes_feat = roi_pool_1d(feat, proposals_feat, roi_len=roi_len, pool="max")

    # align RPN scores with roi_feat order
    rpn_scores_all = []
    for b in range(x.shape[0]):
        if proposals_feat[b] is None or proposals_feat[b].numel() == 0:
            continue
        rpn_scores_all.append(proposals_score[b])
    rpn_scores_all = torch.cat(rpn_scores_all, dim=0) if len(rpn_scores_all) else torch.zeros((0,), device=device)

    out = [{"boxes": torch.zeros((0,2), device=device),
            "scores": torch.zeros((0,), device=device),
            "calc_p": torch.zeros((0,), device=device),
            "noncalc_p": torch.zeros((0,), device=device),
            "sten_pred": torch.zeros((0,), dtype=torch.long, device=device),
            "sten_p": torch.zeros((0,), device=device)} for _ in range(x.shape[0])]

    if roi_feat.shape[0] == 0:
        return out

    calc_logit, noncalc_logit, sten_logits, roi_deltas = roi_heads(roi_feat)
    calc_p = torch.sigmoid(calc_logit)
    noncalc_p = torch.sigmoid(noncalc_logit)

    sten_prob = torch.softmax(sten_logits, dim=-1)
    sten_p, sten_pred_0to4 = torch.max(sten_prob, dim=-1)
    sten_pred = sten_pred_0to4 + 1  # 1..5 for lesions

    roi_boxes_slice = (roi_boxes_feat * stride).clamp(min=0.0, max=float(window_len))
    refined_slice = decode_deltas_to_boxes_1d(roi_boxes_slice, roi_deltas, clip_min=0.0, clip_max=float(window_len))

    plaque_score = torch.max(calc_p, noncalc_p)   # [R]

    if score_mode == "rpn":
        final_score = rpn_scores_all
    elif score_mode == "rpn_plaque":
        final_score = rpn_scores_all * plaque_score
    else:
        raise ValueError("score_mode must be 'rpn' or 'rpn_plaque'")

    for b in range(x.shape[0]):
        idx = torch.where(roi_batch == b)[0]
        if idx.numel() == 0:
            continue

        boxes_b = refined_slice[idx]
        scores_b = final_score[idx]
        calc_b = calc_p[idx]
        noncalc_b = noncalc_p[idx]
        sten_pred_b = sten_pred[idx]
        sten_p_b = sten_p[idx]


        if debug and b == 0:
            print(
                f"[DEBUG] scores_b min/mean/max = "
                f"{scores_b.min().item():.4f} / {scores_b.mean().item():.4f} / {scores_b.max().item():.4f} | "
                f"final_score_thresh={final_score_thresh}"
            )

        keep_score = torch.where(scores_b >= final_score_thresh)[0]
        boxes_b = boxes_b[keep_score]
        scores_b = scores_b[keep_score]
        calc_b = calc_b[keep_score]
        noncalc_b = noncalc_b[keep_score]
        sten_pred_b = sten_pred_b[keep_score]
        sten_p_b = sten_p_b[keep_score]

        if boxes_b.numel() == 0:
            continue

        keep = nms_1d(boxes_b, scores_b, iou_thresh=final_nms_iou, topk=max_dets)

        out[b] = {
            "boxes": boxes_b[keep],
            "scores": scores_b[keep],
            "calc_p": calc_b[keep],
            "noncalc_p": noncalc_b[keep],
            "sten_pred": sten_pred_b[keep],
            "sten_p": sten_p_b[keep],
        }

    return out


@torch.no_grad()
def match_tp_fp_fn(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, iou_thresh: float = 0.5):
    """
    pred_boxes: [P,2], gt_boxes: [G,2] in same coords
    Greedy matching by max IoU.
    Returns (tp, fp, fn).
    """
    if gt_boxes.numel() == 0 and pred_boxes.numel() == 0:
        return 0, 0, 0
    if gt_boxes.numel() == 0:
        return 0, int(pred_boxes.shape[0]), 0
    if pred_boxes.numel() == 0:
        return 0, 0, int(gt_boxes.shape[0])

    iou = interval_iou_1d(pred_boxes, gt_boxes)  # [P,G]
    tp = 0
    used_gt = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool, device=gt_boxes.device)

    # sort predictions by best IoU descending
    best_iou, best_gt = iou.max(dim=1)  # [P]
    order = best_iou.argsort(descending=True)

    for pi in order:
        if best_iou[pi] < iou_thresh:
            continue
        gi = best_gt[pi].item()
        if not used_gt[gi]:
            used_gt[gi] = True
            tp += 1

    fp = int(pred_boxes.shape[0]) - tp
    fn = int(gt_boxes.shape[0]) - int(used_gt.sum().item())
    return tp, fp, fn

def f1_score(p, r):
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)
import time

@torch.no_grad()
def eval_one_epoch_params(
    dl, fe, rpn, roi_heads, device,
    window_len, stride, roi_len,
    iou_thresh=0.5,
    rpn_score_thresh=0.10,
    final_score_thresh=0.30,
    final_nms_iou=0.5,
    max_dets=100,
    score_mode="rpn_plaque",
    limit_batches: Optional[int] = None,
    debug_first_batch: bool = False,    # <-- ADD
):
    fe.eval(); rpn.eval(); roi_heads.eval()

    TP = FP = FN = 0
    total_preds = 0
    total_gts = 0

    t0 = time.time()

    num_windows = 0
    bi = -1
    for bi, (x, targets) in enumerate(dl):
        if limit_batches is not None and bi >= limit_batches:
            break

        x = x.to(device, non_blocking=True)

        num_windows += len(targets)
        preds = infer_window_batch(
            fe, rpn, roi_heads, x,
            window_len=window_len,
            stride=stride,
            roi_len=roi_len,
            rpn_score_thresh=rpn_score_thresh,
            rpn_post_nms_topk=20,
            rpn_iou_thresh=0.5,
            final_score_thresh=final_score_thresh,
            final_nms_iou=final_nms_iou,
            max_dets=max_dets,
            score_mode=score_mode,
            debug=(debug_first_batch and bi == 0),   # <-- use your debug print
        )

        for b in range(len(targets)):
            gt_boxes = targets[b]["boxes"].to(device)
            pred_boxes = preds[b]["boxes"]

            tp, fp, fn = match_tp_fp_fn(pred_boxes, gt_boxes, iou_thresh=iou_thresh)
            TP += tp; FP += fp; FN += fn
            total_preds += int(pred_boxes.shape[0])
            total_gts += int(gt_boxes.shape[0])

    prec = TP / max(1, (TP + FP))
    rec  = TP / max(1, (TP + FN))
    avg_preds = total_preds / max(1, num_windows)
    avg_gts   = total_gts   / max(1, num_windows)

    dt = time.time() - t0
    return {
        "TP": TP, "FP": FP, "FN": FN,
        "precision": prec,
        "recall": rec,
        "f1": f1_score(prec, rec),
        "avg_preds_per_window": avg_preds,
        "avg_gts_per_window": avg_gts,
        "seconds": dt,   # <-- helpful
        "batches": (bi + 1),
    }


@torch.no_grad()
def estimate_max_score_on_val(
    dl_val, fe, rpn, roi_heads, device,
    window_len, stride, roi_len,
    score_mode="rpn",
    rpn_score_thresh=0.0,
    final_nms_iou=0.5,
    max_dets=100,
    limit_batches: Optional[int] = None
):
    fe.eval(); rpn.eval(); roi_heads.eval()
    max_score = 0.0
    
    for bi, (x, _targets) in enumerate(dl_val):
        if limit_batches is not None and bi >= limit_batches:   # <-- ADD
            break

        x = x.to(device, non_blocking=True)
        preds = infer_window_batch(
            fe, rpn, roi_heads, x,
            window_len=window_len,
            stride=stride,
            roi_len=roi_len,
            rpn_score_thresh=rpn_score_thresh,
            final_score_thresh=0.0,
            final_nms_iou=final_nms_iou,
            max_dets=max_dets,
            score_mode=score_mode,
        )
        for b in range(len(preds)):
            if preds[b]["scores"].numel() > 0:
                max_score = max(max_score, float(preds[b]["scores"].max().item()))

    return max_score


import time

@torch.no_grad()
def tune_final_threshold_only(
    dl_val, fe, rpn, roi_heads, device,
    window_len, stride, roi_len,
    score_mode="rpn",
    rpn_score_thresh=0.0,
    iou_thresh=0.5,
    pred_budget=10.0,
    num_thresh=20,
    max_dets_list=(10,),
    nms_iou_list=(0.3,),
    limit_batches: Optional[int] = None,
):
    max_score = estimate_max_score_on_val(
        dl_val, fe, rpn, roi_heads, device,
        window_len, stride, roi_len,
        score_mode=score_mode,
        rpn_score_thresh=rpn_score_thresh,
        final_nms_iou=0.5,
        max_dets=100,
        limit_batches=limit_batches,   # <-- ADD
    )

    print(f"[tune] estimated max score on val = {max_score:.4f}")

    if max_score <= 1e-6:
        print("[tune] max_score ~ 0 -> model producing near-zero scores; train more.")
        return None, None, []

    thresh_list = np.linspace(0.0, max_score, num_thresh)

    total_runs = len(max_dets_list) * len(nms_iou_list) * len(thresh_list)
    run = 0
    t0 = time.time()

    results = []
    for md in max_dets_list:
        for nms_iou in nms_iou_list:
            best_here = None
            for fin_t in thresh_list:
                run += 1
                if run % 5 == 0 or run == 1:
                    elapsed = time.time() - t0
                    print(f"[tune] run {run}/{total_runs} (elapsed {elapsed:.1f}s)")

                stats = eval_one_epoch_params(
                    dl_val, fe, rpn, roi_heads, device,
                    window_len, stride, roi_len,
                    iou_thresh=iou_thresh,
                    rpn_score_thresh=rpn_score_thresh,
                    final_score_thresh=float(fin_t),
                    final_nms_iou=float(nms_iou),
                    max_dets=int(md),
                    score_mode=score_mode,
                    limit_batches=limit_batches,       # <-- KEY
                    debug_first_batch=False,
                )

                row = {
                    "final_score_thresh": float(fin_t),
                    "final_nms_iou": float(nms_iou),
                    "max_dets": int(md),
                    **stats,
                }
                results.append(row)
                if best_here is None or row["f1"] > best_here["f1"]:
                    best_here = row

            print(
                f"[best F1] md={md:3d} nms={nms_iou:.1f} | "
                f"thr={best_here['final_score_thresh']:.4f} "
                f"F1={best_here['f1']:.4f} P={best_here['precision']:.4f} R={best_here['recall']:.4f} "
                f"(batches={best_here['batches']} time={best_here['seconds']:.1f}s)"
            )

    best_f1 = max(results, key=lambda d: d["f1"])

    under = [r for r in results if r["avg_preds_per_window"] <= pred_budget]
    best_budget = max(under, key=lambda d: d["recall"]) if len(under) else None

    print("\n==== BEST BY F1 ====")
    print(best_f1)

    print(f"\n==== BEST RECALL with avg_preds_per_window <= {pred_budget} ====")
    print(best_budget if best_budget is not None else "No setting met the budget.")

    return best_f1, best_budget, results

def main():
    train_csv = "/home/joshua/Coronary_R-CNN/train_cpr_all26_allbranch_02mm.csv"
    val_csv   = "/home/joshua/Coronary_R-CNN/val_cpr_all26_allbranch_02mm.csv"
    test_csv  = "/home/joshua/Coronary_R-CNN/test_cpr_all26_allbranch_02mm.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    window_len = 768
    roi_len = 16
    stride = 16.0  # feature->slice

    ds_train = VesselWindowDataset(csv_path=train_csv, window_len=window_len, train=True, do_augment=True)
    dl_train = DataLoader(ds_train, batch_size=2, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn)

    ds_val = VesselWindowDataset(csv_path=val_csv, window_len=window_len, train=False, do_augment=False)
    dl_val = DataLoader(ds_val, batch_size=2, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn)


    fe = FeatureExtractorFE(in_ch=1, norm="gn").to(device)
    rpn = RPN1D(in_c=512, anchor_lengths=(2,4,6,9,13,18)).to(device)
    roi_heads = RoIHeads(in_c=512, roi_len=roi_len, num_stenosis_classes=5).to(device)

    params = list(fe.parameters()) + list(rpn.parameters()) + list(roi_heads.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)

    fe.train(); rpn.train(); roi_heads.train()

    for step, (x, targets) in enumerate(dl_train):
        x = x.to(device, non_blocking=True)

        # ===== FE + RPN =====
        feat = fe(x)                                   # [B,512,Lf]
        obj_logits, deltas, anchors = rpn(feat)         # [B,N], [B,N,2], [N,2]
        Lf = feat.shape[-1]

        loss_rpn, stats_rpn = rpn_loss_1d(
            obj_logits, deltas, anchors, targets,
            stride=stride,
            pos_iou_thresh=0.5,
            neg_iou_thresh=0.1,
            sample_size=256,
            pos_fraction=0.5,
            reg_weight=1.0,
        )

        # ===== Proposals + RoI pooling =====
        with torch.no_grad():
            proposals_feat, proposals_score = rpn.propose(
                obj_logits, deltas, anchors, Lf=Lf,
                score_thresh=0.0, pre_nms_topk=600, post_nms_topk=20, iou_thresh=0.5
            )

        # Add GT boxes (converted to feature coords) so RoI head always sees positives
        for b in range(len(targets)):
            gt_boxes_slice = targets[b]["boxes"].to(device)
            if gt_boxes_slice.numel() == 0:
                continue
            gt_feat = (gt_boxes_slice / stride).clamp(0, Lf)

            proposals_feat[b] = torch.cat([proposals_feat[b], gt_feat], dim=0)

            # give GT a score of 1.0 so it doesn't get ignored later
            gt_scores = torch.ones((gt_feat.shape[0],), device=device)
            proposals_score[b] = torch.cat([proposals_score[b], gt_scores], dim=0)


        roi_feat, roi_batch, roi_boxes_feat = roi_pool_1d(feat, proposals_feat, roi_len=roi_len, pool="max")
        # roi_feat: [R,512,roi_len], roi_batch: [R], roi_boxes_feat: [R,2]

        # If no rois, just train RPN this step
        loss_roi = torch.zeros((), device=device)

        if roi_feat.shape[0] > 0:
            # Convert RoI boxes to SLICE coords for matching
            roi_boxes_slice = (roi_boxes_feat * stride).clamp(min=0.0, max=float(window_len))

            # Build per-RoI targets by matching within each batch item
            labels_all = torch.full((roi_feat.shape[0],), -1, dtype=torch.long, device=device)
            plaque_t_all = torch.zeros((roi_feat.shape[0],), dtype=torch.long, device=device)
            sten_t_all = torch.zeros((roi_feat.shape[0],), dtype=torch.long, device=device)
            matched_gt_all = torch.zeros((roi_feat.shape[0], 2), dtype=torch.float32, device=device)

            for b in range(len(targets)):
                idx = torch.where(roi_batch == b)[0]
                if idx.numel() == 0:
                    continue

                gt_boxes = targets[b]["boxes"].to(device)                 # [G,2] slice coords
                gt_plaq = targets[b]["plaque"].to(device).long()          # [G]
                gt_sten = targets[b]["stenosis"].to(device).long()        # [G] (your stenosis is classification)

                labels, pl_t, st_t, mgt = assign_rois_to_gt(
                    proposals_slice=roi_boxes_slice[idx],
                    gt_boxes_slice=gt_boxes,
                    gt_plaque=gt_plaq,
                    gt_sten=gt_sten,
                    pos_iou_thresh=0.5,
                    neg_iou_thresh=0.1,
                )

                labels_all[idx] = labels
                plaque_t_all[idx] = pl_t
                sten_t_all[idx] = st_t
                matched_gt_all[idx] = mgt
 
            # AFTER the for-loop that fills labels_all
            if step % 10 == 0:
                print()
                pos = (labels_all == 1).sum().item()
                neg = (labels_all == 0).sum().item()
                ign = (labels_all == -1).sum().item()
                print(f"roi match: pos {pos} neg {neg} ign {ign}")
                gt_total = sum(t["boxes"].shape[0] for t in targets)
                print("gt boxes in batch:", gt_total)


            # Subsample RoIs for stable training
            keep = sample_rois(labels_all, batch_size=128, pos_fraction=0.25)

            if keep.numel() > 0:
                calc_logit, noncalc_logit, sten_logits, roi_deltas = roi_heads(roi_feat[keep])

                # ---- plaque 2-bit targets from your mapping ----
                # 0=background -> (0,0)
                # 1=calcified  -> (1,0)
                # 2=noncalc    -> (0,1)
                # 3=mixed      -> (1,1)
                pl = plaque_t_all[keep]
                calc_t = ((pl == 3) | (pl == 2)).float()
                noncalc_t = ((pl == 1) | (pl == 2)).float()

                loss_calc = torch.nn.functional.binary_cross_entropy_with_logits(calc_logit, calc_t)
                loss_noncalc = torch.nn.functional.binary_cross_entropy_with_logits(noncalc_logit, noncalc_t)

                # ---- stenosis loss ONLY for positive RoIs ----
                pos_mask_in_keep = (labels_all[keep] == 1)
                loss_sten = torch.zeros((), device=device)

                if pos_mask_in_keep.any():
                    sten_pos = sten_t_all[keep][pos_mask_in_keep].long()  # should be 1..5

                    # (optional safety) drop any accidental 0s
                    valid = (sten_pos >= 1) & (sten_pos <= 5)
                    if valid.any():
                        sten_pos = sten_pos[valid] - 1  # 1..5 -> 0..4
                        sten_logits_pos = sten_logits[pos_mask_in_keep][valid]  # [P,5]
                        loss_sten = F.cross_entropy(sten_logits_pos, sten_pos)

                # ---- RoI regression (ONLY for positive RoIs) ----
                # roi_boxes_slice is [R,2] for all rois; matched_gt_all is [R,2]
                pos_keep = keep[labels_all[keep] == 1]
                loss_roi_reg = torch.zeros((), device=device)

                if pos_keep.numel() > 0:
                    # NOTE: roi_deltas is indexed by 'keep', so we need positions within 'keep'
                    # Make a mask on keep positions:
                    pos_mask_in_keep = (labels_all[keep] == 1)

                    prop_pos = roi_boxes_slice[pos_keep]         # [Ppos,2]
                    gt_pos   = matched_gt_all[pos_keep]          # [Ppos,2]

                    # Encode proposal->gt as (t_c, t_w) targets (proposal plays the role of "anchor")
                    tgt_d = encode_boxes_to_deltas(prop_pos, gt_pos)   # [Ppos,2]

                    pred_d = roi_deltas[pos_mask_in_keep]              # [Ppos,2]
                    loss_roi_reg = torch.nn.functional.smooth_l1_loss(pred_d, tgt_d)

                loss_roi = loss_calc + loss_noncalc + loss_sten + loss_roi_reg
        loss = loss_rpn + loss_roi

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        if step % 10 == 0:
            print(
                f"step {step:04d} | "
                f"loss {float(loss.detach().cpu()):.4f} "
                f"(rpn {float(loss_rpn.detach().cpu()):.4f} roi {float(loss_roi.detach().cpu()):.4f}) | "
                f"rpn_pos {stats_rpn['pos_anchors']} | rois {roi_feat.shape[0]}"
            )
            props_counts = [p.shape[0] for p in proposals_feat]
            print("props per item:", props_counts)
        
        if step % 200 == 0 and step > 0:
            stats = eval_one_epoch_params(
                dl_val, fe, rpn, roi_heads, device,
                window_len, stride, roi_len,
                iou_thresh=0.5,
                rpn_score_thresh=0.0,
                final_score_thresh=0.0,   # keep it permissive while training
                score_mode="rpn",         # important: don’t multiply by plaque_score yet
                final_nms_iou=0.5,
                max_dets=100,
            )
            print("VAL:", stats)
            fe.train(); rpn.train(); roi_heads.train()


        
        

        if step >= 500:
            break


    print("\n[SMOKE] one val batch inference sanity check...")
    fe.eval(); rpn.eval(); roi_heads.eval()

    x, targets = next(iter(dl_val))
    x = x.to(device, non_blocking=True)

    preds = infer_window_batch(
        fe, rpn, roi_heads, x,
        window_len=window_len,
        stride=stride,
        roi_len=roi_len,
        rpn_score_thresh=0.0,
        final_score_thresh=0.0,
        final_nms_iou=0.5,
        max_dets=100,
        score_mode="rpn",
        debug=True,
    )
    # --- sanity checks on predicted boxes (SMOKE) ---
    for b in range(len(preds)):
        boxes = preds[b]["boxes"]  # [P,2]
        if boxes.numel() > 0:
            assert torch.all(boxes[:, 0] <= boxes[:, 1]), "bad box: start > end"
            assert boxes.min().item() >= -1e-3, "box < 0"
            assert boxes.max().item() <= window_len + 1e-3, "box > window_len"

    for b in range(len(preds)):
        print(f"[SMOKE] item {b} preds={preds[b]['boxes'].shape[0]} gts={targets[b]['boxes'].shape[0]}")
        if preds[b]["scores"].numel() > 0:
            print(f"        score min/mean/max = {preds[b]['scores'].min().item():.4f} / "
                f"{preds[b]['scores'].mean().item():.4f} / {preds[b]['scores'].max().item():.4f}")
    

    # ---- after training loop finishes ----
    print("\n[TUNING] searching for inference thresholds on VAL...")

    print("\n[TUNING] (FAST SMOKE) searching for inference thresholds on VAL (subset)...")

    best_f1, best_budget, results = tune_final_threshold_only(
        dl_val, fe, rpn, roi_heads, device,
        window_len=window_len,
        stride=stride,
        roi_len=roi_len,
        score_mode="rpn",
        rpn_score_thresh=0.0,
        iou_thresh=0.5,
        pred_budget=10.0,
        num_thresh=20,              # FAST
        max_dets_list=(10,),        # FAST
        nms_iou_list=(0.3,),        # FAST
        limit_batches=50,           # FAST: only 10 batches
    )

    print("\n[TUNING] best_f1:", best_f1)
    print("[TUNING] best_budget:", best_budget)

    # ---- ADD THIS: full-val check at a realistic inference threshold ----
    print("\n[VAL] sanity eval with a non-zero final_score_thresh...")

    for name, chosen in [("best_f1", best_f1), ("best_budget", best_budget)]:
        if chosen is None:
            continue
        stats = eval_one_epoch_params(
            dl_val, fe, rpn, roi_heads, device,
            window_len, stride, roi_len,
            iou_thresh=0.5,
            rpn_score_thresh=0.0,
            final_score_thresh=float(chosen["final_score_thresh"]),
            final_nms_iou=float(chosen["final_nms_iou"]),
            max_dets=int(chosen["max_dets"]),
            score_mode="rpn",
            limit_batches=None,
        )
        print(f"[FULL VAL] {name} ->", stats)
if __name__ == "__main__":
    main()
