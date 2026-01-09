import os
import csv
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from FE import FeatureExtractorFE
from RPN import RPN1D, interval_iou_1d
from cached_ds import CachedWindowDataset, collate_fn
from torch.utils.data import DataLoader


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


@torch.no_grad()
def run_one(fe, rpn, x, stride, score_thresh, pre_nms_topk, post_nms_topk, nms_iou):
    """
    x: [1,1,D,H,W] or [B,1,D,H,W]
    returns proposals in SLICE coords for each item in batch
    """
    feat = fe(x)                     # [B,512,Lf]
    obj_logits, deltas, anchors = rpn(feat)
    B, _, Lf = feat.shape

    proposals_feat, scores_list = rpn.propose(
        obj_logits, deltas, anchors, Lf=Lf,
        score_thresh=score_thresh,
        pre_nms_topk=pre_nms_topk,
        post_nms_topk=post_nms_topk,
        iou_thresh=nms_iou,
        allow_fallback=False,   # IMPORTANT for honest eval/visuals
    )

    proposals_slice = []
    for b in range(B):
        pf = proposals_feat[b]
        sf = scores_list[b]
        if pf is None or pf.numel() == 0:
            proposals_slice.append((torch.empty((0, 2), device=x.device),
                                    torch.empty((0,), device=x.device)))
        else:
            proposals_slice.append((pf * float(stride), sf))
    return proposals_slice



def _robust_minmax(img, lo=1, hi=99):
    vmin, vmax = np.percentile(img, [lo, hi])
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)

def plot_overlay_1d_and_proj(
    out_path,
    x_3d,                 # [D,H,W] CPU torch
    gt_boxes,             # [G,2] CPU torch
    props, scores,        # [P,2], [P] CPU torch
    title=""
):
    D, H, W = x_3d.shape

    # 1D signal
    signal = x_3d.float().mean(dim=(1, 2)).numpy()

    # projection: try mean projection (usually less “blobby” than max)
    proj_DW = x_3d.float().mean(dim=1).numpy()   # [D,W]

    # robust scaling for display
    vmin, vmax = _robust_minmax(proj_DW, 1, 99)

    fig = plt.figure(figsize=(16, 6))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(np.arange(D), signal)

    for (g0, g1) in gt_boxes.numpy():
        ax1.axvspan(g0, g1, color="red", alpha=0.18, linewidth=0)

    for i in range(props.shape[0]):
        p0, p1 = props[i].numpy()
        sc = float(scores[i].item()) if scores.numel() else 0.0
        ax1.axvspan(p0, p1, color="blue", alpha=min(0.35, 0.05 + 0.35 * sc), linewidth=0)

    if scores.numel() and props.shape[0]:
        topk = min(8, props.shape[0])
        idx = torch.topk(scores, topk).indices
        y_top = ax1.get_ylim()[1]
        for j in idx.tolist():
            p0, p1 = props[j].numpy()
            sc = float(scores[j].item())
            ax1.text((p0 + p1) / 2.0, y_top * 0.95, f"{sc:.2f}", ha="center", va="top", fontsize=9)

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
        interpolation="nearest"
    )
    ax2.set_ylabel("W (projection)")
    ax2.set_xlabel("D (slice index)")

    for (g0, g1) in gt_boxes.numpy():
        ax2.axvspan(g0, g1, color="red", alpha=0.18, linewidth=0)
    for i in range(props.shape[0]):
        p0, p1 = props[i].numpy()
        sc = float(scores[i].item()) if scores.numel() else 0.0
        ax2.axvspan(p0, p1, color="blue", alpha=min(0.25, 0.05 + 0.25 * sc), linewidth=0)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)

    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--stride", type=float, default=16.0)

    ap.add_argument("--score_thresh", type=float, default=0.1)
    ap.add_argument("--pre_nms_topk", type=int, default=600)
    ap.add_argument("--post_nms_topk", type=int, default=100)
    ap.add_argument("--nms_iou", type=float, default=0.5)

    ap.add_argument("--miss_iou_thresh", type=float, default=0.5)
    ap.add_argument("--max_vis", type=int, default=100)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    vis_dir = ensure_dir(os.path.join(out_dir, "vis"))
    miss_dir = ensure_dir(os.path.join(out_dir, "miss_vis"))
    miss_csv = os.path.join(out_dir, "missed_gt.csv")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # dataset / loader
    ds = CachedWindowDataset(args.data, "val")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # load model
    ckpt = torch.load(args.ckpt, map_location=device)
    fe = FeatureExtractorFE(in_ch=1, norm="gn").to(device)
    rpn = RPN1D(in_c=512, anchor_lengths=tuple(ckpt.get("anchor_lengths", (1, 2, 3, 5, 7)))).to(device)
    fe.load_state_dict(ckpt["fe"])
    rpn.load_state_dict(ckpt["rpn"])
    fe.eval(); rpn.eval()

    missed_rows = []
    vis_count = 0

    for idx, (x, targets) in enumerate(loader):
        x = x.to(device)  # [1,1,D,H,W]
        gt = targets[0]["boxes"].cpu()  # [G,2] slice coords
        D = x.shape[2]

        (props_s, scores) = run_one(
            fe, rpn, x,
            stride=args.stride,
            score_thresh=args.score_thresh,
            pre_nms_topk=args.pre_nms_topk,
            post_nms_topk=args.post_nms_topk,
            nms_iou=args.nms_iou
        )[0]

        props_s = props_s.cpu()
        scores = scores.cpu()

        # visualize some samples
        if vis_count < args.max_vis:
            x_cpu = x[0, 0].detach().cpu()  # [D,H,W]
            out_path = os.path.join(vis_dir, f"val{idx:04d}.png")
            plot_overlay_1d_and_proj(
                out_path,
                x_cpu,
                gt,
                props_s,
                scores,
                miss_iou_thresh=args.miss_iou_thresh,
                title=f"val idx={idx} | gt={gt.shape[0]} | props={props_s.shape[0]}"
            )
            vis_count += 1

        # compute missed GTs
        if gt.numel() > 0:
            if props_s.numel() == 0:
                best_iou = torch.zeros((gt.shape[0],), dtype=torch.float32)
            else:
                iou = interval_iou_1d(props_s, gt)  # [P,G]
                best_iou = iou.max(dim=0).values    # [G]

            missed = torch.where(best_iou < args.miss_iou_thresh)[0]
            if missed.numel() > 0:
                # save zoom visuals around missed GTs
                x_cpu = x[0, 0].detach().cpu()
                for g_idx in missed.tolist():
                    g0, g1 = gt[g_idx].tolist()
                    bi = float(best_iou[g_idx].item())
                    missed_rows.append([idx, g_idx, g0, g1, bi, int(gt.shape[0]), int(props_s.shape[0])])

                    out_path = os.path.join(miss_dir, f"miss_test{idx:04d}_gt{g_idx:02d}_iou{bi:.2f}.png")
                    plot_overlay_1d_and_proj(
                        out_path,
                        x_cpu,
                        gt[g_idx:g_idx+1],     # only that GT highlighted
                        props_s,
                        scores,
                        miss_iou_thresh=args.miss_iou_thresh,
                        title=f"MISS: val idx={idx} gt#{g_idx} [{g0:.0f},{g1:.0f}] best_iou={bi:.3f}"
                    )

    # write CSV
    with open(miss_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "gt_idx", "gt_start", "gt_end", "best_iou", "num_gt_in_sample", "num_props"])
        w.writerows(missed_rows)

    print("Done.")
    print("Saved:")
    print(" - all vis:", vis_dir)
    print(" - missed vis:", miss_dir)
    print(" - missed CSV:", miss_csv)


if __name__ == "__main__":
    main()
