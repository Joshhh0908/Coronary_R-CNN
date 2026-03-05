import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms as tv_nms


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
        # self.obj_head = nn.Conv1d(in_c, self.A, kernel_size=1)        # lesion vs bg logits
        # self.reg_head = nn.Conv1d(in_c, self.A * 2, kernel_size=1)    # (t_c,t_w) per anchor
        
        self.obj_head = nn.Linear(in_c, self.A)        # lesion vs bg logits 
        self.reg_head = nn.Linear(in_c, self.A*2)        # lesion vs bg logits 

    def forward(self, feat):
        """
        feat: [B,512,Lf]
        """
        B, C, Lf = feat.shape
        h = F.relu(self.conv(feat), inplace=True)

        x = h.permute(0, 2, 1).contiguous()
        obj = self.obj_head(x)     # [B,Lf,A]
        reg = self.reg_head(x)     # [B,Lf,2A]
        reg = reg.view(B, Lf, self.A, 2)
        obj = obj.reshape(B, Lf * self.A)
        reg = reg.reshape(B, Lf * self.A, 2)
        anchors = make_anchors_1d(Lf, self.anchor_lengths, device=feat.device)  # [N,2]
        return obj, reg, anchors
        

    @torch.no_grad()
    def propose(
        self, obj_logits, deltas, anchors, Lf,
        score_thresh=0.1, pre_nms_topk=600, post_nms_topk=100, iou_thresh=0.5,
        allow_fallback=False,
        has_gt=None,
        fallback_topk=100,
    ):
        B, N = obj_logits.shape
        scores = torch.sigmoid(obj_logits)                 # [B,N]
        boxes = decode_deltas_to_boxes(anchors, deltas, Lf=Lf)  # [B,N,2]

        proposals = []
        scores_out = []

        for b in range(B):
            s = scores[b]          # [N]
            bxs = boxes[b]         # [N,2]

            # -------------------------
            # (1) TOPK FIRST (caps work)
            # -------------------------
            k = min(int(pre_nms_topk), s.numel())
            topk_idx = torch.topk(s, k).indices
            s = s[topk_idx]
            bxs = bxs[topk_idx]

            # -------------------------
            # (2) THEN threshold (cheap)
            # -------------------------
            keep = (s >= score_thresh)
            s = s[keep]
            bxs = bxs[keep]

            # -------------------------
            # (3) Fallback logic (GT-only)
            # -------------------------
            if s.numel() == 0:
                # Old behavior: allow_fallback means "always fallback"
                # New recommended behavior: fallback only if this sample has GT
                do_fb = False
                if allow_fallback:
                    do_fb = True
                elif has_gt is not None and bool(has_gt[b]):
                    do_fb = True

                if not do_fb:
                    proposals.append(torch.empty((0, 2), device=bxs.device))
                    scores_out.append(torch.empty((0,), device=s.device))
                    continue

                # small fallback, not pre_nms_topk
                kfb = min(int(fallback_topk), scores[b].numel())
                fb_idx = torch.topk(scores[b], kfb).indices
                s = scores[b][fb_idx]
                bxs = boxes[b][fb_idx]

            # -------------------------
            # (4) NMS (fast version)
            # -------------------------
            keep_nms = nms_1d_torchvision(bxs, s, iou_thresh=iou_thresh, topk=post_nms_topk)
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


def nms_1d_torchvision(boxes_1d: torch.Tensor, scores: torch.Tensor, iou_thresh=0.5, topk=200):
    """
    boxes_1d: [N,2] (start,end)
    scores:   [N]
    returns: indices to keep
    """
    if boxes_1d.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes_1d.device)

    x1 = boxes_1d[:, 0]
    x2 = boxes_1d[:, 1]

    # create fake 2D boxes: [x1, y1, x2, y2]
    y1 = torch.zeros_like(x1)
    y2 = torch.ones_like(x1)
    boxes_2d = torch.stack([x1, y1, x2, y2], dim=1)

    keep = tv_nms(boxes_2d, scores, iou_thresh)

    if topk is not None and keep.numel() > topk:
        keep = keep[:topk]
    return keep

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
    good = best_iou_per_gt >= pos_iou_thresh
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

    for b in range(B):
        gt_boxes_slice = targets[b]["boxes"].to(device)  # [M,2] slice coords
        gt_boxes_feat = gt_boxes_slice / float(stride)   # feature coords

        labels, matched_gt = assign_rpn_targets_1d(
            anchors, gt_boxes_feat,
            pos_iou_thresh=pos_iou_thresh,
            neg_iou_thresh=neg_iou_thresh,
        )


        # sample anchors for stable training
        scores_b = torch.sigmoid(obj_logits[b].detach())
        keep = sample_anchors(labels, batch_size=sample_size, pos_fraction=pos_fraction)
        if not hasattr(rpn_loss_1d, "_dbg"):
            rpn_loss_1d._dbg = True
            with torch.no_grad():
                kpos = keep[labels[keep] == 1]
                kneg = keep[labels[keep] == 0]
                print("[sample] pos", int(kpos.numel()), "neg", int(kneg.numel()))
                if kpos.numel(): print("[sample] pos score mean", float(scores_b[kpos].mean()))
                if kneg.numel(): print("[sample] neg score mean", float(scores_b[kneg].mean()))

        if keep.numel() == 0:
            continue
        num_valid += 1

        total_samp += keep.numel()

        # objectness targets
        obj_t = (labels[keep] == 1).float()  # [K]
        # obj_l = focal_bce_with_logits(obj_logits[b, keep], obj_t, alpha=0.25, gamma=2.0)
        obj_l = F.binary_cross_entropy_with_logits(obj_logits[b, keep], obj_t)
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

def sample_anchors_hard(labels, scores, batch_size=256, pos_fraction=0.5, hard_neg_frac=0.5):
    device = labels.device
    pos_idx = torch.where(labels == 1)[0]
    neg_idx = torch.where(labels == 0)[0]

    num_pos = min(int(batch_size * pos_fraction), pos_idx.numel())
    num_neg = min(batch_size - num_pos, neg_idx.numel())

    # sample positives randomly (fine)
    if num_pos > 0:
        pos_keep = pos_idx[torch.randperm(pos_idx.numel(), device=device)[:num_pos]]
    else:
        pos_keep = torch.empty((0,), dtype=torch.long, device=device)

    if num_neg > 0:
        # hard negatives: pick top scores among negatives
        neg_scores = scores[neg_idx]
        k_hard = int(num_neg * hard_neg_frac)
        k_hard = min(k_hard, neg_idx.numel())
        hard_neg = neg_idx[torch.topk(neg_scores, k_hard).indices]

        # if you want some random negatives too:
        k_rand = num_neg - k_hard
        if k_rand > 0:
            rest = neg_idx[torch.randperm(neg_idx.numel(), device=device)[:k_rand]]
            neg_keep = torch.cat([hard_neg, rest], dim=0)
        else:
            neg_keep = hard_neg
    else:
        neg_keep = torch.empty((0,), dtype=torch.long, device=device)

    return torch.cat([pos_keep, neg_keep], dim=0)

#unused for now
def focal_bce_with_logits(logits, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()
