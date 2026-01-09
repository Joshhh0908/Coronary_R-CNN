import os
import csv
import random
import argparse
from datetime import datetime
import sys
from tqdm import tqdm


import numpy as np
import torch
from torch.utils.data import DataLoader

# ---- import your code ----
from FE import FeatureExtractorFE
from RPN import RPN1D, rpn_loss_1d, interval_iou_1d

from cached_ds import CachedWindowDataset, collate_fn


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def write_run_config_txt(path: str, args: argparse.Namespace, extra: dict):
    lines = []
    lines.append(f"run_created_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("[args]")
    for k, v in sorted(vars(args).items()):
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("[extra]")
    for k, v in sorted(extra.items()):
        lines.append(f"{k}: {v}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def append_metrics_csv(csv_path: str, header: list, row: dict):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def debug_miss_stages(obj_logits_1, deltas_1, anchors, gt_feat, Lf,
                      score_thresh, pre_nms_topk, post_nms_topk, nms_iou, stride):
    # decode ALL boxes
    scores_all = torch.sigmoid(obj_logits_1)              # [N]
    boxes_all  = decode_deltas_to_boxes(anchors, deltas_1[None, ...], Lf=Lf)[0]  # [N,2]

    # IoU of all decoded boxes vs GT (in feature coords)
    iou_all = interval_iou_1d(boxes_all, gt_feat)         # [N,G]
    best_all = iou_all.max(dim=0).values if gt_feat.numel() else torch.zeros((0,), device=boxes_all.device)
    best_all_max = float(best_all.max().item()) if best_all.numel() else 0.0

    # after score thresh
    keep = torch.where(scores_all >= score_thresh)[0]
    best_after_thresh_max = 0.0
    if keep.numel():
        iou_thr = interval_iou_1d(boxes_all[keep], gt_feat)
        best_after_thresh_max = float(iou_thr.max(dim=0).values.max().item())

    # after pre-topk on score
    best_after_pre_max = 0.0
    if keep.numel():
        s = scores_all[keep]
        b = boxes_all[keep]
        if s.numel() > pre_nms_topk:
            idx = torch.topk(s, pre_nms_topk).indices
            s = s[idx]; b = b[idx]
        iou_pre = interval_iou_1d(b, gt_feat)
        best_after_pre_max = float(iou_pre.max(dim=0).values.max().item())

        # after NMS + post topk
        keep_nms = nms_1d(b, s, iou_thresh=nms_iou, topk=post_nms_topk)
        if keep_nms.numel():
            iou_nms = interval_iou_1d(b[keep_nms], gt_feat)
            best_after_nms_max = float(iou_nms.max(dim=0).values.max().item())
        else:
            best_after_nms_max = 0.0
    else:
        best_after_nms_max = 0.0

    return {
        "best_all_max": best_all_max,
        "best_after_thresh_max": best_after_thresh_max,
        "best_after_pre_max": best_after_pre_max,
        "best_after_nms_max": best_after_nms_max,
        "num_keep_after_thresh": int(keep.numel()),
        "max_score_all": float(scores_all.max().item()),
        "mean_score_all": float(scores_all.mean().item()),
    }


@torch.no_grad()
def evaluate_rpn(
    fe, rpn, loader, device,
    stride=16.0,
    pos_iou_thresh=0.4,
    neg_iou_thresh=0.1,
    score_thresh=0.1,
    pre_nms_topk=600,
    post_nms_topk=100,
    nms_iou=0.5,
    topk_for_recall=50,
    use_pbar: bool = False,
    pbar_desc: str = "Val",
    max_batches=None
):
    fe.eval()
    rpn.eval()

    loss_sum = 0.0
    n_batches = 0

    total_gt = 0
    covered_03 = 0
    covered_05 = 0
    covered_07 = 0

    best_iou_sum = 0.0
    props_sum = 0
    props_nonempty = 0

    # NEW: distribution + crowding + score/length diagnostics (GT-level)
    best_ious_all = []          # list of best IoU per GT
    best_scores_all = []        # score of best-matching proposal per GT
    gt_lens_all = []            # GT length (slices) per GT
    best_prop_lens_all = []     # length of best proposal (slices) per GT
    props_ge_03_sum = 0         # total #props with IoU>=0.3 across all GTs
    props_ge_05_sum = 0         # total #props with IoU>=0.5 across all GTs

    num_samples_total = 0
    num_samples_with_gt = 0

    props_sum_all = 0
    samples_with_any_props = 0 

    eval_iter = loader
    if use_pbar:
        eval_iter = tqdm(loader, desc=pbar_desc, leave=False, dynamic_ncols=True)

    for step, (x, targets) in enumerate(eval_iter, start=1):
        num_samples_total += len(targets)

        x = x.to(device, non_blocking=True)
        feat = fe(x)
        obj_logits, deltas, anchors = rpn(feat)

        loss, _stats = rpn_loss_1d(
            obj_logits, deltas, anchors,
            targets,
            stride=stride,
            pos_iou_thresh=pos_iou_thresh,
            neg_iou_thresh=neg_iou_thresh,
            sample_size=256,
            pos_fraction=0.5,
        )

        loss_sum += float(loss.item())
        n_batches += 1

        if max_batches is not None and n_batches >= max_batches:
            break

        B, _, Lf = feat.shape
        proposals_feat, scores_list = rpn.propose(
            obj_logits, deltas, anchors, Lf=Lf,
            score_thresh=score_thresh,
            pre_nms_topk=pre_nms_topk,
            post_nms_topk=post_nms_topk,
            iou_thresh=nms_iou,
        )

        for b in range(B):

            # count proposals for ALL windows (including no-GT)
            props_f_all = proposals_feat[b]
            nprops = 0 if (props_f_all is None or props_f_all.numel() == 0) else int(props_f_all.shape[0])
            props_sum_all += nprops
            if nprops > 0:
                samples_with_any_props += 1

            #GT only logic
            gt = targets[b]["boxes"].to(device)  # [G,2] slice coords
            if gt.numel() == 0:
                continue
                # ---- DEBUG INSERT START ----
            # convert GT to FEATURE coords for debugging
            gt_feat = gt / float(stride)

            dbg = debug_miss_stages(
                obj_logits_1=obj_logits[b],    # [N]
                deltas_1=deltas[b],            # [N,2]
                anchors=anchors,               # [N,2]
                gt_feat=gt_feat,               # [G,2]
                Lf=Lf,
                score_thresh=score_thresh,
                pre_nms_topk=pre_nms_topk,
                post_nms_topk=post_nms_topk,
                nms_iou=nms_iou,
                stride=stride,
            )

            # optional: only print when it becomes a miss at your recall threshold (e.g. 0.5)
            # BUT print the stage-wise best so you know who killed it.
            if dbg["best_after_nms_max"] < 0.5:
                print(
                    f"[MISS@0.5] step={step} b={b} "
                    f"all={dbg['best_all_max']:.3f} "
                    f"thr={dbg['best_after_thresh_max']:.3f} "
                    f"pre={dbg['best_after_pre_max']:.3f} "
                    f"nms={dbg['best_after_nms_max']:.3f} "
                    f"keep={dbg['num_keep_after_thresh']} "
                    f"maxS={dbg['max_score_all']:.3f} meanS={dbg['mean_score_all']:.3f}"
                )
            # ---- DEBUG INSERT END ----
            num_samples_with_gt += 1
            G = gt.shape[0]
            total_gt += G

            # GT lengths
            gt_len = (gt[:, 1] - gt[:, 0]).clamp(min=0)
            gt_lens_all.extend(gt_len.detach().cpu().tolist())

            props_f = proposals_feat[b]
            if props_f is None or props_f.numel() == 0:
                # no proposals => all GTs missed
                best_ious_all.extend([0.0] * G)
                best_scores_all.extend([0.0] * G)
                best_prop_lens_all.extend([0.0] * G)
                continue

            # proposals in SLICE coords
            props_s = props_f * float(stride)  # [P,2]
            s = scores_list[b]                      # [P]
            if s is None:
                s = torch.empty((0,), device=props_s.device)
            props_sum += props_s.shape[0]
            props_nonempty += 1

            # restrict to topk_for_recall for IoU-based metrics
            if props_s.shape[0] > topk_for_recall:
                props_s = props_s[:topk_for_recall]
                s = s[:topk_for_recall]

            iou = interval_iou_1d(props_s, gt)  # [P,G]
            best_per_gt, best_idx = iou.max(dim=0)  # [G], [G]

            # basic aggregates (existing)
            best_iou_sum += float(best_per_gt.sum().item())
            covered_03 += int((best_per_gt >= 0.3).sum().item())
            covered_05 += int((best_per_gt >= 0.5).sum().item())
            covered_07 += int((best_per_gt >= 0.7).sum().item())

            # NEW: distribution tracking
            best_ious_all.extend(best_per_gt.detach().cpu().tolist())

            # NEW: score of best-matching proposal per GT
            if s is None or s.numel() == 0:
                best_scores = torch.zeros_like(best_per_gt)
            else:
                best_scores = s[best_idx]
            best_scores_all.extend(best_scores.detach().cpu().tolist())

            # NEW: length of best proposal per GT
            best_props = props_s[best_idx]  # [G,2]
            best_prop_len = (best_props[:, 1] - best_props[:, 0]).clamp(min=0)
            best_prop_lens_all.extend(best_prop_len.detach().cpu().tolist())

            # NEW: crowding stats (# proposals overlapping each GT above thresholds)
            # iou is [P,G]
            props_ge_03_sum += int((iou >= 0.3).sum().item())
            props_ge_05_sum += int((iou >= 0.5).sum().item())

    # Percentiles for best IoU / best score
    if len(best_ious_all) > 0:
        p10, p50, p90 = np.percentile(best_ious_all, [10, 50, 90]).tolist()
    else:
        p10 = p50 = p90 = 0.0

    if len(best_scores_all) > 0:
        s_p10, s_p50, s_p90 = np.percentile(best_scores_all, [10, 50, 90]).tolist()
        s_mean = float(np.mean(best_scores_all))
    else:
        s_p10 = s_p50 = s_p90 = 0.0
        s_mean = 0.0

    gt_len_mean = float(np.mean(gt_lens_all)) if len(gt_lens_all) else 0.0
    best_prop_len_mean = float(np.mean(best_prop_lens_all)) if len(best_prop_lens_all) else 0.0

    out = {
        # existing
        "val_loss": loss_sum / max(1, n_batches),
        "gt_count": total_gt,
        "recall@0.3": (covered_03 / total_gt) if total_gt > 0 else 0.0,
        "recall@0.5": (covered_05 / total_gt) if total_gt > 0 else 0.0,
        "mean_best_iou": (best_iou_sum / total_gt) if total_gt > 0 else 0.0,
        "avg_props_per_sample_with_gt": (props_sum / max(1, props_nonempty)),

        # NEW
        "recall@0.7": (covered_07 / total_gt) if total_gt > 0 else 0.0,
        "miss@0.3": 1.0 - ((covered_03 / total_gt) if total_gt > 0 else 0.0),
        "miss@0.5": 1.0 - ((covered_05 / total_gt) if total_gt > 0 else 0.0),

        "best_iou_p10": float(p10),
        "best_iou_p50": float(p50),
        "best_iou_p90": float(p90),

        "best_score_mean": float(s_mean),
        "best_score_p50": float(s_p50),
        "best_score_p10": float(s_p10),
        "best_score_p90": float(s_p90),

        "gt_len_mean": gt_len_mean,
        "best_prop_len_mean": best_prop_len_mean,

        # crowding: average number of overlapping proposals per GT
        "props_per_gt_iou_ge_03": (props_ge_03_sum / total_gt) if total_gt > 0 else 0.0,
        "props_per_gt_iou_ge_05": (props_ge_05_sum / total_gt) if total_gt > 0 else 0.0,

        "samples_total": int(num_samples_total),
        "samples_with_gt": int(num_samples_with_gt),
        "frac_samples_with_gt": (num_samples_with_gt / max(1, num_samples_total)),

        "avg_props_per_sample_all": props_sum_all / max(1, num_samples_total),
        "frac_samples_with_any_props": samples_with_any_props / max(1, num_samples_total),
    }

    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", type=str, required=True,
                    help="folder that contains train/val/test pt dirs")
    ap.add_argument(
    "--out_root",
    type=str,
    default="RPN_results",
    help="where to save runs (checkpoints/metrics/config). Can be absolute or relative.",
)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--no_pbar", action="store_true", help="disable tqdm progress bars")
    ap.add_argument("--pbar_update", type=int, default=10, help="update tqdm postfix every N iters")

    # training label thresholds
    ap.add_argument("--pos_iou_thresh", type=float, default=0.4)
    ap.add_argument("--neg_iou_thresh", type=float, default=0.1)

    # proposal params (affects recall eval)
    ap.add_argument("--score_thresh", type=float, default=0.1)
    ap.add_argument("--pre_nms_topk", type=int, default=600)
    ap.add_argument("--post_nms_topk", type=int, default=100)
    ap.add_argument("--nms_iou", type=float, default=0.5)
    ap.add_argument("--topk_for_recall", type=int, default=50)

    # anchors
    ap.add_argument("--anchor_lengths", type=str, default="1,2,3,5,7",
                    help="comma-separated anchor lengths in FEATURE units")

    # output naming
    ap.add_argument("--run_name", type=str, default="",
                    help="optional run name prefix")

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- output folder in same folder as cache_root ----
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = (args.run_name.strip() + "_") if args.run_name.strip() else ""
    out_root = os.path.abspath(args.out_root)  # e.g. ./RPN_results or /some/path/RPN_results
    out_dir = ensure_dir(os.path.join(out_root, f"{prefix}fe_rpn_{run_stamp}"))
    ckpt_dir = ensure_dir(os.path.join(out_dir, "checkpoints"))

    metrics_csv = os.path.join(out_dir, "metrics.csv")
    config_txt = os.path.join(out_dir, "run_config.txt")


    anchor_lengths = tuple(int(x) for x in args.anchor_lengths.split(",") if x.strip() != "")
    stride = 16.0  # must match FE downsample along D

    write_run_config_txt(
        config_txt, args,
        extra={
            "device": str(device),
            "out_dir": out_dir,
            "stride": stride,
            "anchor_lengths": anchor_lengths,
        }
    )

    print("Outputs:", out_dir)

    # ---- datasets ----
    ds_train = CachedWindowDataset(args.data, "train")
    ds_val = CachedWindowDataset(args.data, "val")
    print(f"Train samples: {len(ds_train)} | Val samples: {len(ds_val)}")

    train_loader = DataLoader(
        ds_train, batch_size=args.bs, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.bs, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn
    )

    # ---- models ----
    fe = FeatureExtractorFE(in_ch=1, norm="gn").to(device)
    rpn = RPN1D(in_c=512, anchor_lengths=anchor_lengths).to(device)

    # ---- optimizer ----
    params = list(fe.parameters()) + list(rpn.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    # AMP
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_recall05 = -1.0

    header = [
        "epoch",
        "train_loss", "train_obj", "train_reg",
        "train_pos_anchors", "train_sampled_anchors",
        "val_loss",
        "val_recall@0.3", "val_recall@0.5", "val_recall@0.7",
        "val_miss@0.3", "val_miss@0.5",
        "val_mean_best_iou",
        "val_best_iou_p10", "val_best_iou_p50", "val_best_iou_p90",
        "val_best_score_mean", "val_best_score_p10", "val_best_score_p50", "val_best_score_p90",
        "val_gt_len_mean", "val_best_prop_len_mean",
        "val_props_per_gt_iou_ge_03", "val_props_per_gt_iou_ge_05",
        "val_avg_props_per_sample_with_gt",
        "val_gt_count",
        "val_samples_total", "val_samples_with_gt", "val_frac_samples_with_gt",
        "lr",
        "ckpt_path",
        "val_avg_props_per_sample_all",
        "val_frac_samples_with_any_props",

    ]

    # ---- training ----
    for epoch in range(1, args.epochs + 1):
        fe.train()
        rpn.train()
        print(f"\n=== Epoch {epoch:03d}/{args.epochs} started ===", flush=True)

        running_loss = 0.0
        running_obj = 0.0
        running_reg = 0.0
        running_pos = 0
        running_samp = 0
        n_batches = 0
        train_iter = train_loader
        if not args.no_pbar:
            train_iter = tqdm(train_loader, desc=f"Train E{epoch:03d}", leave=False, dynamic_ncols=True)

        for step, (x, targets) in enumerate(train_iter, start=1):
            x = x.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                feat = fe(x)
                obj_logits, deltas, anchors = rpn(feat)

                loss, stats = rpn_loss_1d(
                    obj_logits, deltas, anchors,
                    targets,
                    stride=stride,
                    pos_iou_thresh=args.pos_iou_thresh,
                    neg_iou_thresh=args.neg_iou_thresh,
                    sample_size=256,
                    pos_fraction=0.5,
                    reg_weight=1.0,
                )

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running_loss += float(stats["loss_total"])
            running_obj += float(stats["loss_obj"])
            running_reg += float(stats["loss_reg"])
            running_pos += int(stats["pos_anchors"])
            running_samp += int(stats["sampled_anchors"])
            n_batches += 1
            if (not args.no_pbar) and (step % args.pbar_update == 0):
                train_iter.set_postfix(
                    loss=f"{stats['loss_total']:.4f}",
                    obj=f"{stats['loss_obj']:.4f}",
                    reg=f"{stats['loss_reg']:.4f}",
                    pos=int(stats["pos_anchors"]),
                )

        train_loss = running_loss / max(1, n_batches)
        train_obj = running_obj / max(1, n_batches)
        train_reg = running_reg / max(1, n_batches)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} "
            f"obj={train_obj:.4f} "
            f"reg={train_reg:.4f} | "
            f"pos={running_pos} samp={running_samp}"
        )

        # ---- eval ----
        metrics = evaluate_rpn(
            fe, rpn, val_loader, device,
            stride=stride,
            pos_iou_thresh=args.pos_iou_thresh,
            neg_iou_thresh=args.neg_iou_thresh,
            score_thresh=args.score_thresh,
            pre_nms_topk=args.pre_nms_topk,
            post_nms_topk=args.post_nms_topk,
            nms_iou=args.nms_iou,
            topk_for_recall=args.topk_for_recall,
            use_pbar=(not args.no_pbar),
            pbar_desc=f"Val   E{epoch:03d}",
        )

        print(
            f"           val_loss={metrics['val_loss']:.4f} | "
            f"recall@0.3={metrics['recall@0.3']:.3f} "
            f"recall@0.5={metrics['recall@0.5']:.3f} | "
            f"mean_best_iou={metrics['mean_best_iou']:.3f} | "
            f"avg_props/gt-sample={metrics['avg_props_per_sample_with_gt']:.1f} "
            f"(gt={metrics['gt_count']})"
        )

        # ---- save checkpoint ----
        ckpt = {
            "epoch": epoch,
            "fe": fe.state_dict(),
            "rpn": rpn.state_dict(),
            "optimizer": optim.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "args": vars(args),
            "metrics": {
                "train_loss": train_loss,
                "train_obj": train_obj,
                "train_reg": train_reg,
                **metrics,
            },
            "stride": stride,
            "anchor_lengths": anchor_lengths,
        }

        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
        torch.save(ckpt, ckpt_path)

        # save best checkpoint by recall@0.5
        if metrics["recall@0.5"] > best_recall05:
            best_recall05 = metrics["recall@0.5"]
            torch.save(ckpt, os.path.join(ckpt_dir, "best_by_recall05.pt"))

        # ---- append metrics row ----
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_obj": train_obj,
            "train_reg": train_reg,
            "train_pos_anchors": int(running_pos),
            "train_sampled_anchors": int(running_samp),

            "val_loss": float(metrics["val_loss"]),
            "val_recall@0.3": float(metrics["recall@0.3"]),
            "val_recall@0.5": float(metrics["recall@0.5"]),
            "val_recall@0.7": float(metrics["recall@0.7"]),
            "val_miss@0.3": float(metrics["miss@0.3"]),
            "val_miss@0.5": float(metrics["miss@0.5"]),
            "val_mean_best_iou": float(metrics["mean_best_iou"]),
            "val_best_iou_p10": float(metrics["best_iou_p10"]),
            "val_best_iou_p50": float(metrics["best_iou_p50"]),
            "val_best_iou_p90": float(metrics["best_iou_p90"]),

            "val_best_score_mean": float(metrics["best_score_mean"]),
            "val_best_score_p10": float(metrics["best_score_p10"]),
            "val_best_score_p50": float(metrics["best_score_p50"]),
            "val_best_score_p90": float(metrics["best_score_p90"]),

            "val_gt_len_mean": float(metrics["gt_len_mean"]),
            "val_best_prop_len_mean": float(metrics["best_prop_len_mean"]),

            "val_props_per_gt_iou_ge_03": float(metrics["props_per_gt_iou_ge_03"]),
            "val_props_per_gt_iou_ge_05": float(metrics["props_per_gt_iou_ge_05"]),

            "val_avg_props_per_sample_with_gt": float(metrics["avg_props_per_sample_with_gt"]),
            "val_gt_count": int(metrics["gt_count"]),

            "val_samples_total": int(metrics["samples_total"]),
            "val_samples_with_gt": int(metrics["samples_with_gt"]),
            "val_frac_samples_with_gt": float(metrics["frac_samples_with_gt"]),

            "val_avg_props_per_sample_all": float(metrics["avg_props_per_sample_all"]),
            "val_frac_samples_with_any_props": float(metrics["frac_samples_with_any_props"]),


            "lr": float(optim.param_groups[0]["lr"]),
            "ckpt_path": ckpt_path,
        }

        append_metrics_csv(metrics_csv, header, row)


if __name__ == "__main__":
    main()
