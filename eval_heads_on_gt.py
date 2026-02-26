"""Evaluate plaque/stenosis heads on ground-truth boxes directly.
This isolates classification ability from RPN/proposal pipeline.
"""

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from FE import FeatureExtractorFE
from roi_pool import roi_pool_1d
from heads import RoIHeads, plaque_label_to_bits
from cached_ds import CachedWindowDataset, collate_fn


def plaque_logits_to_class(plaque_logits: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """Convert two-bit logits to class 0..3 using threshold on probs."""
    probs = torch.sigmoid(plaque_logits)
    calc = (probs[:, 0] >= thr).long()
    nonc = (probs[:, 1] >= thr).long()

    out = torch.zeros_like(calc)
    out[(calc == 0) & (nonc == 1)] = 1
    out[(calc == 1) & (nonc == 1)] = 2
    out[(calc == 1) & (nonc == 0)] = 3
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--roi_len", type=int, default=16)
    ap.add_argument("--stenosis_classes", type=int, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    fe = FeatureExtractorFE(in_ch=1, norm="gn").to(device)
    roi_heads = RoIHeads(stenosis_num_classes=args.stenosis_classes, in_c=512, hidden=256, dropout=0.1).to(device)
    fe.load_state_dict(ckpt["fe"])
    roi_heads.load_state_dict(ckpt["roi_heads"])
    fe.eval(); roi_heads.eval()

    ds = CachedWindowDataset(args.data, args.split)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type=="cuda"),
                        collate_fn=collate_fn)

    total = 0
    correct_plaq = 0
    correct_sten = 0
    bit0_correct = 0
    bit1_correct = 0
    plaq_confusion = np.zeros((4,4), dtype=np.int64)
    
    # collect logits for analysis
    all_plaq_logits = []
    all_plaq_probs = []
    all_plaq_pred_bits = []
    all_plaq_gt_bits = []

    with torch.no_grad():
        for x, targets in loader:
            x = x.to(device)
            feat = fe(x)
            stride = float(ckpt.get("stride", 16.0))
            
            for b in range(x.shape[0]):
                gt_boxes = targets[b]["boxes"].to(device).float()
                if gt_boxes.numel() == 0:
                    continue
                gt_plq = targets[b]["plaque"].to(device).long()
                gt_sten = targets[b]["stenosis"].to(device).long()
                total += gt_boxes.shape[0]

                gt_boxes_f = gt_boxes / stride
                pooled, _, rois_cat = roi_pool_1d(feat[b:b+1], [gt_boxes_f], roi_len=args.roi_len)
                pooled = pooled.to(device)
                rois_cat = rois_cat.to(device)

                out = roi_heads(pooled, rois_cat, Lf=feat.shape[2])
                pred_plaq = plaque_logits_to_class(out["plaque_logits"], thr=0.5)
                pred_sten = torch.argmax(out["stenosis_logits"], dim=1)

                # bit-level correctness
                probs = torch.sigmoid(out["plaque_logits"])
                bit0 = (probs[:,0] >= 0.5).long()
                bit1 = (probs[:,1] >= 0.5).long()
                gt_bits = plaque_label_to_bits(gt_plq).long()
                bit0_correct += (bit0 == gt_bits[:,0]).sum().item()
                bit1_correct += (bit1 == gt_bits[:,1]).sum().item()

                correct_plaq += (pred_plaq == gt_plq).sum().item()
                correct_sten += (pred_sten == gt_sten).sum().item()

                # confusion matrix
                for g,p in zip(gt_plq.cpu().numpy(), pred_plaq.cpu().numpy()):
                    plaq_confusion[g,p] += 1
                
                # collect for deeper analysis
                all_plaq_logits.extend(out["plaque_logits"].cpu().numpy().tolist())
                all_plaq_probs.extend(probs.cpu().numpy().tolist())
                all_plaq_pred_bits.extend(torch.stack([bit0, bit1], dim=1).cpu().numpy().tolist())
                all_plaq_gt_bits.extend(gt_bits.cpu().numpy().tolist())

    all_plaq_logits = np.array(all_plaq_logits)
    all_plaq_probs = np.array(all_plaq_probs)
    all_plaq_pred_bits = np.array(all_plaq_pred_bits)
    all_plaq_gt_bits = np.array(all_plaq_gt_bits)

    print(f"\nTotal GT boxes: {total}")
    print(f"Plaque accuracy: {correct_plaq/total:.4f}")
    print(f"Stenosis accuracy: {correct_sten/total:.4f}")
    print(f"Plaque bit0 accuracy: {bit0_correct/total:.4f}")
    print(f"Plaque bit1 accuracy: {bit1_correct/total:.4f}")

    print("\n" + "="*70)
    print("CONFUSION MATRIX (GT rows -> Pred cols)")
    print("="*70)
    print("        pred_0  pred_1  pred_2  pred_3")
    for i in range(4):
        print(f"gt_{i}:  {plaq_confusion[i,0]:6d}  {plaq_confusion[i,1]:6d}  {plaq_confusion[i,2]:6d}  {plaq_confusion[i,3]:6d}")

    # Analyze what the model is predicting for each GT class
    print("\n" + "="*70)
    print("MODEL PREDICTION ANALYSIS")
    print("="*70)
    
    # Get predicted bits for each GT class
    gt_to_class_names = {1: "non-calc only", 2: "mixed", 3: "calc only"}
    pred_bit_names = {(0,0): "none", (0,1): "non-calc", (1,0): "calc", (1,1): "mixed"}
    
    for gt_class in [1, 2, 3]:
        mask = (all_plaq_gt_bits[:, 0] == (gt_class // 2)) & (all_plaq_gt_bits[:, 1] == (gt_class % 2))
        if mask.sum() == 0:
            continue
        
        # Remap to class
        actual_bit0 = (gt_class == 3 or gt_class == 2)
        actual_bit1 = (gt_class == 1 or gt_class == 2)
        
        pred_bits = all_plaq_pred_bits[mask]
        probs = all_plaq_probs[mask]
        
        print(f"\nGT class {gt_class} ({gt_to_class_names[gt_class]}), n={mask.sum()}:")
        print(f"  bit0 (calc)     pred==true: {100*(pred_bits[:,0]==actual_bit0).mean():.1f}%, "
              f"pred prob: p50={np.percentile(probs[:,0], 50):.3f}, p90={np.percentile(probs[:,0], 90):.3f}")
        print(f"  bit1 (non-calc) pred==true: {100*(pred_bits[:,1]==actual_bit1).mean():.1f}%, "
              f"pred prob: p50={np.percentile(probs[:,1], 50):.3f}, p90={np.percentile(probs[:,1], 90):.3f}")
        
        # distribution of predictions
        pred_classes = pred_bits[:,0]*2 + pred_bits[:,1]
        print(f"  Prediction distribution: ", end="")
        for pc in range(4):
            cnt = (pred_classes == pc).sum()
            if cnt > 0:
                print(f"{pred_bit_names[tuple([(pc//2), (pc%2)])]}: {cnt} ", end="")
        print()


if __name__ == "__main__":
    main()
