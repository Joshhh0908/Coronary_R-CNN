# rpn_recall_eval.py
import argparse
from typing import List, Dict

import torch
from torch.utils.data import DataLoader

from dataprep import VesselWindowDataset, collate_fn
from detection_module import FeatureExtractorFE, RPN1D


def iou_1d(gt: torch.Tensor, props: torch.Tensor) -> torch.Tensor:
    """
    gt:    [G,2]
    props: [P,2]
    returns IoU matrix [G,P]
    """
    if gt.numel() == 0 or props.numel() == 0:
        return torch.zeros((gt.shape[0], props.shape[0]), device=gt.device)

    inter = (torch.min(gt[:, None, 1], props[None, :, 1]) -
             torch.max(gt[:, None, 0], props[None, :, 0])).clamp(min=0.0)
    union = (torch.max(gt[:, None, 1], props[None, :, 1]) -
             torch.min(gt[:, None, 0], props[None, :, 0])).clamp(min=1e-6)
    return inter / union


@torch.no_grad()
def eval_rpn_recall(
    fe: torch.nn.Module,
    rpn: torch.nn.Module,
    loader: DataLoader,
    device: str,
    stride: float = 16.0,
    score_thresh: float = 0.1,
    pre_nms_topk: int = 600,
    post_nms_topk: int = 100,
    nms_iou: float = 0.5,
    iou_thresholds: List[float] = [0.3, 0.5],
    max_batches: int = 100,
):
    fe.eval()
    rpn.eval()

    total_gt = 0
    total_normals = 0
    total_props = 0
    best_ious_all = []

    hits = {t: 0 for t in iou_thresholds}

    for b_idx, (x, targets) in enumerate(loader):
        if b_idx >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        feat = fe(x)  # [B,512,Lf]
        obj_logits, deltas, anchors = rpn(feat)
        Lf = feat.shape[-1]

        proposals_feat = rpn.propose(
            obj_logits, deltas, anchors, Lf=Lf,
            score_thresh=score_thresh,
            pre_nms_topk=pre_nms_topk,
            post_nms_topk=post_nms_topk,
            iou_thresh=nms_iou,
        )

        # per item in batch
        for i in range(len(targets)):
            gt = targets[i]["boxes"].to(device)  # slice coords within window [0..L]
            if gt.numel() == 0:
                total_normals += 1
                # still count proposals stats
                total_props += proposals_feat[i].shape[0]
                continue

            # proposals are in FEATURE coords [0..Lf], convert to SLICE coords [0..window_len]
            props = proposals_feat[i] * float(stride)  # [P,2] slice coords
            total_props += props.shape[0]

            total_gt += gt.shape[0]

            if props.numel() == 0:
                # no proposals => all misses
                best_ious_all.extend([0.0] * gt.shape[0])
                continue

            ious = iou_1d(gt, props)      # [G,P]
            best_iou = ious.max(dim=1).values  # [G]
            best_ious_all.extend(best_iou.detach().cpu().tolist())

            for t in iou_thresholds:
                hits[t] += int((best_iou >= t).sum().item())

    recall = {t: (hits[t] / max(1, total_gt)) for t in iou_thresholds}
    avg_best_iou = (sum(best_ious_all) / max(1, len(best_ious_all))) if best_ious_all else 0.0
    avg_props_per_item = total_props / max(1, (total_normals + (total_gt > 0)))  # not perfect, but ok

    print("==== RPN Window Recall Eval ====")
    print(f"GT lesions counted: {total_gt}")
    print(f"Normal (no-GT) windows: {total_normals}")
    print(f"Avg best IoU over GT lesions: {avg_best_iou:.3f}")
    for t in iou_thresholds:
        print(f"Recall@IoU{t:.1f}: {recall[t]:.3f}")
    print(f"Total proposals produced: {total_props}")
    print("================================")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--window_len", type=int, default=768)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_batches", type=int, default=100)

    # This is important:
    # use_train_sampling=True => uses lesion-aware cropping you added (focus_prob) so GT is present more often
    # use_train_sampling=False => deterministic z0=0 windows (will miss lesions beyond first 768)
    ap.add_argument("--use_train_sampling", action="store_true")

    # proposal params
    ap.add_argument("--score_thresh", type=float, default=0.4)
    ap.add_argument("--pre_nms_topk", type=int, default=600)
    ap.add_argument("--post_nms_topk", type=int, default=100)
    ap.add_argument("--nms_iou", type=float, default=0.5)

    # optional weights
    ap.add_argument("--weights", type=str, default="")

    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ds = VesselWindowDataset(
        csv_path=args.csv,
        window_len=args.window_len,
        train=args.use_train_sampling,     # IMPORTANT: enables lesion-focused z0 if you set it up
        do_augment=False,                  # keep eval clean
        do_random_rotate=False,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
    )

    fe = FeatureExtractorFE(in_ch=1, norm="gn").to(device)
    rpn = RPN1D(in_c=512, anchor_lengths=(2, 4, 6, 9, 13, 18)).to(device)

    if args.weights:
        ckpt = torch.load(args.weights, map_location=device)
        # support either {"fe":..., "rpn":...} or direct state_dict keys
        if isinstance(ckpt, dict) and "fe" in ckpt and "rpn" in ckpt:
            fe.load_state_dict(ckpt["fe"], strict=True)
            rpn.load_state_dict(ckpt["rpn"], strict=True)
        else:
            # if you saved a single merged dict
            fe.load_state_dict(ckpt["fe_state_dict"], strict=False)
            rpn.load_state_dict(ckpt["rpn_state_dict"], strict=False)

    eval_rpn_recall(
        fe, rpn, dl, device=device,
        stride=16.0,
        score_thresh=args.score_thresh,
        pre_nms_topk=args.pre_nms_topk,
        post_nms_topk=args.post_nms_topk,
        nms_iou=args.nms_iou,
        iou_thresholds=[0.3, 0.5],
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
