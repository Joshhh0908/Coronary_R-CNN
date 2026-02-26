"""
Diagnostic script to collect plaque lesionness scores on proposals that overlap GT lesions.
Helps determine if the plaque head is misscoring true lesions or if the threshold is just too strict.
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from FE import FeatureExtractorFE
from RPN import RPN1D, interval_iou_1d
from roi_pool import roi_pool_1d
from heads import RoIHeads
from cached_ds import CachedWindowDataset, collate_fn


def diagnose():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="data folder with train/val/test .pt")
    ap.add_argument("--ckpt", type=str, required=True, help="path to checkpoint")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_samples", type=int, default=500, help="max samples to analyze")
    
    # RPN + proposal params (should match training)
    ap.add_argument("--score_thresh", type=float, default=0.05)
    ap.add_argument("--pre_nms_topk", type=int, default=600)
    ap.add_argument("--post_nms_topk", type=int, default=100)
    ap.add_argument("--nms_iou", type=float, default=0.5)
    ap.add_argument("--anchor_lengths", type=str, default="1,2,3,5,7")
    ap.add_argument("--roi_len", type=int, default=16)
    ap.add_argument("--stenosis_classes", type=int, required=True)
    ap.add_argument("--match_iou", type=float, default=0.5, help="IoU threshold for matching proposals to GT")
    
    args = ap.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    stride = float(ckpt.get("stride", 16.0))
    anchor_lengths = tuple(int(x) for x in args.anchor_lengths.split(",") if x.strip())
    
    # Load model
    fe = FeatureExtractorFE(in_ch=1, norm="gn").to(device)
    rpn = RPN1D(in_c=512, anchor_lengths=anchor_lengths).to(device)
    roi_heads = RoIHeads(stenosis_num_classes=args.stenosis_classes, in_c=512, hidden=256, dropout=0.1).to(device)
    
    fe.load_state_dict(ckpt["fe"])
    rpn.load_state_dict(ckpt["rpn"])
    roi_heads.load_state_dict(ckpt["roi_heads"])
    
    fe.eval(); rpn.eval(); roi_heads.eval()
    
    # Load data
    ds = CachedWindowDataset(args.data, args.split)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
                       pin_memory=(device.type == "cuda"), collate_fn=collate_fn)
    
    print(f"\n=== Plaque Score Diagnostic ===")
    print(f"Split: {args.split}, Max samples: {args.max_samples}")
    print(f"Checkpoint: {args.ckpt}\n")
    
    # Accumulators
    plaque_scores_on_gt_props = []      # plaque score for proposals overlapping GT (IoU >= match_iou)
    plaque_scores_on_negative_props = []  # plaque score for proposals NOT overlapping GT
    num_samples = 0
    total_gt = 0
    total_props = 0
    total_props_matching_gt = 0
    
    with torch.no_grad():
        for batch_i, (x, targets) in enumerate(loader):
            if num_samples >= args.max_samples:
                break
            
            x = x.to(device, non_blocking=True)
            B = x.shape[0]
            
            # Forward through FE + RPN
            feat = fe(x)
            _, _, Lf = feat.shape
            
            obj_logits, deltas, anchors = rpn(feat)
            
            # Get proposals
            proposals_feat, scores_list = rpn.propose(
                obj_logits, deltas, anchors, Lf=Lf,
                score_thresh=args.score_thresh,
                pre_nms_topk=args.pre_nms_topk,
                post_nms_topk=args.post_nms_topk,
                iou_thresh=args.nms_iou,
                allow_fallback=True,
            )
            
            # RoI pooling
            pooled, batch_ix, rois_cat = roi_pool_1d(feat, proposals_feat, roi_len=args.roi_len)
            
            if pooled.numel() > 0:
                # RoI head predictions
                out = roi_heads(pooled, rois_cat, Lf=Lf)
                
                # Plaque lesionness score
                plq_probs = torch.sigmoid(out["plaque_logits"])  # [R, 2]
                p_lesion = 1.0 - (1.0 - plq_probs[:, 0]) * (1.0 - plq_probs[:, 1])  # [R]
                
                # Convert proposal coords to slice space
                pred_boxes_s = rois_cat * float(stride)  # [R, 2] slice coords
            else:
                p_lesion = torch.empty((0,), device=device)
                pred_boxes_s = torch.empty((0, 2), device=device)
            
            # Process each item in batch
            for b in range(B):
                if num_samples >= args.max_samples:
                    break
                
                num_samples += 1
                
                # Ground truth
                gt_boxes_s = targets[b]["boxes"].float().to(device)  # [G, 2] slice coords
                
                if gt_boxes_s.numel() == 0:
                    continue
                
                num_gt = gt_boxes_s.shape[0]
                total_gt += num_gt
                
                gt_boxes_f = gt_boxes_s / float(stride)
                
                # Get RoIs from this batch item
                ridx = torch.where(batch_ix == b)[0]
                total_props += len(ridx)
                
                if ridx.numel() > 0 and num_gt > 0:
                    # Compute IoU between proposals and GT
                    rois_b = rois_cat[ridx]  # [P, 2] feat coords
                    ious = interval_iou_1d(rois_b, gt_boxes_f)  # [P, G]
                    max_iou_per_prop = ious.max(dim=1).values  # [P]
                    
                    # Split by match status
                    match_mask = (max_iou_per_prop >= args.match_iou)
                    
                    plq_b = p_lesion[ridx]
                    
                    # Scores for proposals matching GT
                    if match_mask.any():
                        scores_matched = plq_b[match_mask].cpu().numpy()
                        plaque_scores_on_gt_props.extend(scores_matched.tolist())
                        total_props_matching_gt += match_mask.sum().item()
                    
                    # Scores for proposals NOT matching GT (negatives)
                    if (~match_mask).any():
                        scores_neg = plq_b[~match_mask].cpu().numpy()
                        plaque_scores_on_negative_props.extend(scores_neg.tolist())
    
    # Compute statistics
    print(f"Examined {num_samples} samples with {total_gt} GT lesions")
    print(f"Total RPN proposals: {total_props}")
    print(f"Proposals overlapping GT (IoU >= {args.match_iou}): {total_props_matching_gt}")
    print(f"Proposals with NO GT overlap: {total_props - total_props_matching_gt}\n")
    
    if len(plaque_scores_on_gt_props) > 0:
        scores_gt = np.array(plaque_scores_on_gt_props)
        print("=" * 70)
        print("PLAQUE SCORES ON TRUE LESION PROPOSALS (IoU >= {})".format(args.match_iou))
        print("=" * 70)
        print(f"Count: {len(scores_gt)}")
        print(f"Min:    {scores_gt.min():.4f}")
        print(f"P10:    {np.percentile(scores_gt, 10):.4f}")
        print(f"P25:    {np.percentile(scores_gt, 25):.4f}")
        print(f"P50:    {np.percentile(scores_gt, 50):.4f}")
        print(f"P75:    {np.percentile(scores_gt, 75):.4f}")
        print(f"P90:    {np.percentile(scores_gt, 90):.4f}")
        print(f"Max:    {scores_gt.max():.4f}")
        print(f"Mean:   {scores_gt.mean():.4f}")
        print(f"Std:    {scores_gt.std():.4f}")
        
        # Recall at different thresholds
        print(f"\nRecall at different plaque_score_thresh:")
        for thresh in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            recall = (scores_gt >= thresh).mean()
            count_above = (scores_gt >= thresh).sum()
            print(f"  thresh={thresh:.1f}: {100*recall:.1f}% ({count_above}/{len(scores_gt)} proposals)")
    
    if len(plaque_scores_on_negative_props) > 0:
        scores_neg = np.array(plaque_scores_on_negative_props)
        print("\n" + "=" * 70)
        print("PLAQUE SCORES ON FALSE POSITIVE PROPOSALS (IoU < {})".format(args.match_iou))
        print("=" * 70)
        print(f"Count: {len(scores_neg)}")
        print(f"Min:    {scores_neg.min():.4f}")
        print(f"P10:    {np.percentile(scores_neg, 10):.4f}")
        print(f"P25:    {np.percentile(scores_neg, 25):.4f}")
        print(f"P50:    {np.percentile(scores_neg, 50):.4f}")
        print(f"P75:    {np.percentile(scores_neg, 75):.4f}")
        print(f"P90:    {np.percentile(scores_neg, 90):.4f}")
        print(f"Max:    {scores_neg.max():.4f}")
        print(f"Mean:   {scores_neg.mean():.4f}")
        print(f"Std:    {scores_neg.std():.4f}")
        
        # FP rate at different thresholds
        print(f"\nFalse positive rate at different plaque_score_thresh:")
        for thresh in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            fp_rate = (scores_neg >= thresh).mean()
            count_above = (scores_neg >= thresh).sum()
            print(f"  thresh={thresh:.1f}: {100*fp_rate:.1f}% ({count_above}/{len(scores_neg)} FP proposals)")
    
    # Summary analysis
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if len(plaque_scores_on_gt_props) > 0 and len(plaque_scores_on_negative_props) > 0:
        scores_gt = np.array(plaque_scores_on_gt_props)
        scores_neg = np.array(plaque_scores_on_negative_props)
        
        # Overlap analysis
        print(f"\nScore distribution overlap:")
        print(f"  True lesion proposals p90: {np.percentile(scores_gt, 90):.4f}")
        print(f"  FP proposals p10:         {np.percentile(scores_neg, 10):.4f}")
        
        overlap = np.percentile(scores_neg, 90) - np.percentile(scores_gt, 10)
        sep = np.percentile(scores_gt, 50) - np.percentile(scores_neg, 50)
        
        if overlap > 0.2:
            print(f"\n  ⚠ High overlap in score distributions.")
            print(f"    → Plaque head cannot cleanly separate TP from FP")
            print(f"    → Need to improve plaque head training (more weight, augmentation, harder negatives)")
        else:
            print(f"\n  ✓ Good separation between TP and FP scores")
            print(f"    → Simply lower plaque_score_thresh to recover more TP")
        
        # Recommendation
        if scores_gt.mean() < 0.3:
            print(f"\n  🔴 PROBLEM: True lesions have very low plaque scores (mean={scores_gt.mean():.3f})")
            print(f"    → Plaque head is failing to recognize lesions")
            print(f"    → Retrain with: higher w_plaque, harder negatives, more augmentation")
        elif scores_gt.mean() > 0.5:
            print(f"\n  ✓ True lesions score well (mean={scores_gt.mean():.3f})")
            print(f"    → Lower plaque_score_thresh to 0.2–0.3 and reassess recall/FP tradeoff")


if __name__ == "__main__":
    diagnose()
