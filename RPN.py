import torch
import torch.nn as nn
import torch.nn.functional as F


class RPN1D(nn.Module):
    """
    Input:  feat [B, 512, Lf]   (Lf = D/16)
    Output:
      obj_logits: [B, N]        (N = Lf*A) raw logits
      deltas:     [B, N, 2]     (t_c, t_w)
      anchors:    [N, 2]        anchor intervals in feature coords
    """
    def __init__(self, in_c=512, anchor_lengths=(1, 2, 3, 4, 5, 7, 9)):
        super().__init__()
        self.anchor_lengths = list(anchor_lengths)
        self.A = len(self.anchor_lengths)

        # shared 1D conv (paper: kernel size 3 sliding window)
        self.conv = nn.Conv1d(in_c, in_c, kernel_size=3, padding=1)

        # sibling heads (paper: 2 FC layers)
        # we implement as 1x1 convs (equivalent per position)
        self.obj_head = nn.Conv1d(in_c, self.A, kernel_size=1)        # lesion vs bg logits
        self.reg_head = nn.Conv1d(in_c, self.A * 2, kernel_size=1)    # (t_c,t_w) per anchor

    def forward(self, feat):
        """
        feat: [B,512,Lf]
        """
        B, C, Lf = feat.shape
        h = F.relu(self.conv(feat), inplace=True)

        obj = self.obj_head(h)          # [B,A,Lf]
        reg = self.reg_head(h)          # [B,2A,Lf]

        # reshape to flatten anchors across all positions
        obj = obj.permute(0, 2, 1).contiguous()               # [B,Lf,A]
        reg = reg.permute(0, 2, 1).contiguous()               # [B,Lf,2A]
        reg = reg.view(B, Lf, self.A, 2)                      # [B,Lf,A,2]

        obj = obj.view(B, Lf * self.A)                        # [B,N]
        reg = reg.view(B, Lf * self.A, 2)                     # [B,N,2]

        anchors = make_anchors_1d(Lf, self.anchor_lengths, device=feat.device)  # [N,2]
        return obj, reg, anchors
    

    @torch.no_grad()
    def propose(self, obj_logits, deltas, anchors, Lf,
                score_thresh=0.1, pre_nms_topk=600, post_nms_topk=100, iou_thresh=0.5, allow_fallback=False):
        B, N = obj_logits.shape
        scores = torch.sigmoid(obj_logits)  # [B,N]
        boxes = decode_deltas_to_boxes(anchors, deltas, Lf=Lf)  # [B,N,2]

        # if torch.rand(()) < 0.02:  # ~2% of calls
        #     s_all = scores.detach().flatten()
        #     qs = torch.quantile(s_all, torch.tensor([0.5, 0.9, 0.99], device=s_all.device))
        #     print(
        #         f"[RPN] scores: min={float(s_all.min()):.6f} "
        #         f"p50={float(qs[0]):.6f} p90={float(qs[1]):.6f} p99={float(qs[2]):.6f} "
        #         f"max={float(s_all.max()):.6f} "
        #         f"frac<0.01={float((s_all<0.01).float().mean()):.4f} "
        #         f"frac<0.05={float((s_all<0.05).float().mean()):.4f}"
        #     )
        #     l = obj_logits.detach().flatten()
        #     print(f"[RPN] logits: min={float(l.min()):.3f} mean={float(l.mean()):.3f} max={float(l.max()):.3f}")

        proposals = []
        scores_out = []
        for b in range(B):
            s = scores[b]
            bxs = boxes[b]

            keep = torch.where(s >= score_thresh)[0]

            # fallback: if nothing passes threshold, still take top-k so downstream has something
            if keep.numel() == 0:
                if not allow_fallback:
                    proposals.append(torch.empty((0,2), device=bxs.device))
                    scores_out.append(torch.empty((0,), device=s.device))
                    continue
                k = min(pre_nms_topk, s.numel())
                keep = torch.topk(s, k).indices


            s = s[keep]
            bxs = bxs[keep]

            if s.numel() > pre_nms_topk:
                topk_idx = torch.topk(s, pre_nms_topk).indices
                s = s[topk_idx]
                bxs = bxs[topk_idx]

            keep_nms = nms_1d(bxs, s, iou_thresh=iou_thresh, topk=post_nms_topk)
            proposals.append(bxs[keep_nms])
            scores_out.append(s[keep_nms])

        return proposals, scores_out
    

def make_anchors_1d(Lf: int, anchor_lengths, device):
    """
    Build anchors on the FEATURE axis (length Lf = D/16).
    Returns anchors: [Lf*A, 2] in feature coordinates (start, end)
    """
    anchor_lengths = torch.tensor(anchor_lengths, dtype=torch.float32, device=device)  # [A]
    A = anchor_lengths.numel()

    # centers at 0.5, 1.5, ..., Lf-0.5  (feature bins)
    centers = torch.arange(Lf, device=device, dtype=torch.float32) + 0.5  # [Lf]

    # Broadcast to [Lf, A]
    c = centers[:, None].expand(Lf, A)
    w = anchor_lengths[None, :].expand(Lf, A)

    start = c - 0.5 * w
    end   = c + 0.5 * w
    anchors = torch.stack([start, end], dim=-1).reshape(Lf * A, 2)  # [Lf*A, 2]
    return anchors


def decode_deltas_to_boxes(anchors, deltas, Lf: int):
    a_start = anchors[:, 0]
    a_end   = anchors[:, 1]
    a_c = 0.5 * (a_start + a_end)
    a_w = (a_end - a_start).clamp(min=1e-6)

    t_c = deltas[..., 0]
    t_w = deltas[..., 1].clamp(min=-10.0, max=10.0)

    c = a_c[None, :] + t_c * a_w[None, :]
    w = a_w[None, :] * torch.exp(t_w)

    start = c - 0.5 * w
    end   = c + 0.5 * w

    # scalar clamp to [0, Lf]
    start = start.clamp(min=0.0, max=float(Lf))
    end   = end.clamp(min=0.0, max=float(Lf))

    # enforce end >= start + eps (elementwise)
    eps = 1e-3
    end = torch.maximum(end, start + eps)

    # (optional) if that pushed end above Lf, re-clamp and re-enforce
    end = end.clamp(max=float(Lf))
    start = torch.minimum(start, end - eps)
    start = start.clamp(min=0.0)

    return torch.stack([start, end], dim=-1)



def interval_iou_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: [N,2], b: [M,2]
    returns: [N,M]
    """
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device)

    inter = (torch.min(a[:, None, 1], b[None, :, 1]) -
             torch.max(a[:, None, 0], b[None, :, 0])).clamp(min=0)

    a_len = (a[:, 1] - a[:, 0]).clamp(min=0)[:, None]  # [N,1]
    b_len = (b[:, 1] - b[:, 0]).clamp(min=0)[None, :]  # [1,M]

    union = (a_len + b_len - inter).clamp(min=1e-6)
    return inter / union

def nms_1d(boxes, scores, iou_thresh=0.5, topk=200):
    """
    boxes:  [N,2]
    scores: [N]
    returns indices to keep
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    idx = scores.argsort(descending=True)
    keep = []
    while idx.numel() > 0 and len(keep) < topk:
        i = idx[0]
        keep.append(i.item())
        if idx.numel() == 1:
            break
        ious = interval_iou_1d(boxes[i:i+1], boxes[idx[1:]]).squeeze(0)
        idx = idx[1:][ious <= iou_thresh]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
    

def encode_boxes_to_deltas(anchors, gt_boxes):
    """
    anchors:  [N,2] (start,end) in feature coords
    gt_boxes: [N,2] matched GT for each anchor in feature coords

    returns deltas [N,2] = (t_c, t_w)
    """
    a0, a1 = anchors[:, 0], anchors[:, 1]
    g0, g1 = gt_boxes[:, 0], gt_boxes[:, 1]

    a_c = 0.5 * (a0 + a1)
    a_w = (a1 - a0).clamp(min=1e-6)

    g_c = 0.5 * (g0 + g1)
    g_w = (g1 - g0).clamp(min=1e-6)

    t_c = (g_c - a_c) / a_w
    t_w = torch.log(g_w / a_w)
    return torch.stack([t_c, t_w], dim=-1)


def assign_rpn_targets_1d(
    anchors,              # [N,2] feature coords
    gt_boxes,             # [M,2] feature coords
    pos_iou_thresh=0.5,
    neg_iou_thresh=0.1,
):
    """
    Returns:
      labels: [N] with {1 pos, 0 neg, -1 ignore}
      matched_gt: [N,2] matched GT box per anchor (undefined for neg/ignore)
    """
    N = anchors.shape[0]
    device = anchors.device

    labels = torch.full((N,), -1, dtype=torch.long, device=device)
    matched_gt = torch.zeros((N, 2), dtype=torch.float32, device=device)

    # No GT -> all negatives
    if gt_boxes.numel() == 0:
        labels[:] = 0
        return labels, matched_gt

    # IoU [N,M]
    iou = interval_iou_1d(anchors, gt_boxes)

    max_iou, argmax = iou.max(dim=1)  # for each anchor, best GT
    matched_gt = gt_boxes[argmax]

    # assign neg / pos by thresholds
    labels[max_iou < neg_iou_thresh] = 0
    labels[max_iou >= pos_iou_thresh] = 1

    # Force each GT to have at least one positive anchor (best anchor per GT)
    best_anchor = iou.argmax(dim=0)          # [M]
    best_iou_per_gt = iou.max(dim=0).values  # [M]
    good = best_iou_per_gt >= 0.3
    best_anchor = best_anchor[good]
    labels[best_anchor] = 1
    matched_gt[best_anchor] = gt_boxes[good]
    return labels, matched_gt


def sample_anchors(labels, batch_size=256, pos_fraction=0.5):
    """
    Subsample anchors to keep loss stable (Faster R-CNN style).
    Returns indices of sampled anchors.
    """
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

    keep = torch.cat([pos_keep, neg_keep], dim=0)
    return keep


def rpn_loss_1d(
    obj_logits,     # [B,N]
    deltas,         # [B,N,2]
    anchors,        # [N,2] feature coords
    targets,        # list of dicts, each has "boxes" in SLICE coords
    stride=16.0,    # feature->slice scale
    pos_iou_thresh=0.5,
    neg_iou_thresh=0.1,
    sample_size=256,
    pos_fraction=0.5,
    reg_weight=1.0,
):
    """
    Returns total_loss, dict of scalars
    """
    B, N = obj_logits.shape
    device = obj_logits.device

    total_obj = torch.zeros((), device=device)
    total_reg = torch.zeros((), device=device)
    total_pos = 0
    total_samp = 0
    num_valid = 0
    
    debug_max = 10
    debug_count = 0

    for b in range(B):
        gt_boxes_slice = targets[b]["boxes"].to(device)  # [M,2] slice coords
        gt_boxes_feat = gt_boxes_slice / float(stride)   # feature coords

        labels, matched_gt = assign_rpn_targets_1d(
            anchors, gt_boxes_feat,
            pos_iou_thresh=pos_iou_thresh,
            neg_iou_thresh=neg_iou_thresh,
        )

        # # ---- DEBUG: GT coverage + scores ----
        # if debug_count < debug_max:
        #     with torch.no_grad():
        #         # scores for all anchors in this window
        #         s_all = torch.sigmoid(obj_logits[b])  # [N]

        #         # IoU between every anchor and every GT: [N, M]
        #         iou_am = interval_iou_1d(anchors, gt_boxes_feat)  # anchors vs GT

        #         if gt_boxes_feat.numel() > 0:
        #             # for each GT: best IoU any anchor achieves
        #             best_iou_per_gt, best_anchor_idx = iou_am.max(dim=0)  # [M], [M]

        #             # score of that best-IoU anchor
        #             score_at_best_iou = s_all[best_anchor_idx]  # [M]

        #             # also: best score among anchors that overlap the GT reasonably (IoU >= 0.3)
        #             overlap = (iou_am >= 0.3)  # [N,M]
        #             best_score_overlap = torch.zeros((iou_am.shape[1],), device=device)
        #             for gi in range(iou_am.shape[1]):
        #                 idx = torch.where(overlap[:, gi])[0]
        #                 if idx.numel() == 0:
        #                     best_score_overlap[gi] = -1.0  # means "no anchor overlaps at 0.3"
        #                 else:
        #                     best_score_overlap[gi] = s_all[idx].max()

        #             def pct(x, p):
        #                 return torch.quantile(x.float(), torch.tensor(p, device=x.device)).item()

        #             print(
        #                 f"[GT-COVERAGE] M={iou_am.shape[1]} "
        #                 f"bestIoU p10={pct(best_iou_per_gt,0.10):.3f} p50={pct(best_iou_per_gt,0.50):.3f} p90={pct(best_iou_per_gt,0.90):.3f} "
        #                 f"| score@bestIoU p10={pct(score_at_best_iou,0.10):.3g} p50={pct(score_at_best_iou,0.50):.3g} p90={pct(score_at_best_iou,0.90):.3g} "
        #                 f"| bestScore(overlap>=0.3) p10={pct(best_score_overlap.clamp(min=0),0.10):.3g} p50={pct(best_score_overlap.clamp(min=0),0.50):.3g} p90={pct(best_score_overlap.clamp(min=0),0.90):.3g} "
        #                 f"| frac(noOverlap0.3)={(best_score_overlap<0).float().mean().item():.3f}"
        #             )
        #         debug_count += 1
        # # ---- END DEBUG ----


        # sample anchors for stable training
        keep = sample_anchors(labels, batch_size=sample_size, pos_fraction=pos_fraction)
        if keep.numel() == 0:
            continue
        num_valid += 1

        total_samp += keep.numel()

        # objectness targets
        obj_t = (labels[keep] == 1).float()  # [K]
        obj_l = F.binary_cross_entropy_with_logits(obj_logits[b, keep], obj_t, reduction="mean")
        total_obj += obj_l

        # regression only for positive anchors
        pos_keep = keep[labels[keep] == 1]
        total_pos += pos_keep.numel()

        if pos_keep.numel() > 0:
            target_d = encode_boxes_to_deltas(anchors[pos_keep], matched_gt[pos_keep])  # [P,2]
            reg_l = F.smooth_l1_loss(deltas[b, pos_keep], target_d, reduction="mean")
            total_reg += reg_l

    # average over batch items that contributed

    denom = max(1, num_valid)
    total_obj /= denom
    total_reg /= denom
    loss = total_obj + reg_weight * total_reg
    stats = {
        "loss_total": float(loss.detach().cpu()),
        "loss_obj": float(total_obj.detach().cpu()),
        "loss_reg": float(total_reg.detach().cpu()),
        "pos_anchors": int(total_pos),
        "sampled_anchors": int(total_samp),
    }
    return loss, stats