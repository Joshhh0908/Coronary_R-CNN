import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock3D(nn.Module):
    """
    Basic residual block for 3D volumes.
    """
    def __init__(self, c_in, c_out, stride=1, norm="gn"):
        super().__init__()

        Norm = nn.BatchNorm3d if norm == "bn" else nn.GroupNorm

        self.conv1 = nn.Conv3d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = Norm(c_out) if norm == "bn" else Norm(num_groups=16, num_channels=c_out)
        self.conv2 = nn.Conv3d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = Norm(c_out) if norm == "bn" else Norm(num_groups=16, num_channels=c_out)

        self.proj = None

        ##what is sel.fproj?
        if stride != 1 or c_in != c_out:
            self.proj = nn.Sequential(
                nn.Conv3d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                (Norm(c_out) if norm == "bn" else Norm(num_groups=16, num_channels=c_out)),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.proj is not None:
            identity = self.proj(identity)

        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class FeatureExtractorFE(nn.Module):
    """
    Input:  x [B, 1, D, H, W]
    Output: y [B, 512, D//16]

    Downsamples by /16 along D using 4 stride-2 operations:
      stem conv stride2  -> /2
      maxpool stride2    -> /4
      layer2 stride2     -> /8
      layer3 stride2     -> /16
    Then global-average-pool over H,W (cross-section), keep depth tokens.
    """
    def __init__(self, in_ch=1, norm="gn"):
        super().__init__()
        #whats stem???
        # Stem: /4
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),  # /2
            nn.GroupNorm(16, 64) if norm == "gn" else nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),                      # /4
        )

        # Residual stages (ResNet-like)
        self.layer1 = ResBlock3D(64,  64,  stride=1, norm=norm)   # keep
        self.layer2 = ResBlock3D(64,  128, stride=2, norm=norm)   # /8
        self.layer3 = ResBlock3D(128, 256, stride=2, norm=norm)   # /16
        self.layer4 = ResBlock3D(256, 512, stride=1, norm=norm)   # keep /16

    def forward(self, x):
        """
        x: [B,1,D,H,W]
        returns: [B,512,D//16]
        """
        x = self.stem(x)    # [B,64, D/4,  H/4,  W/4]
        x = self.layer1(x)  # [B,64, ...]
        x = self.layer2(x)  # [B,128, D/8,  H/8,  W/8]
        x = self.layer3(x)  # [B,256, D/16, H/16, W/16]
        x = self.layer4(x)  # [B,512, D/16, H/16, W/16]

        # Global average pool over cross-section (H,W), keep depth
        x = x.mean(dim=(-1, -2))   # [B,512,D/16]
        return x


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
    """
    anchors: [N,2] (start,end) in feature coords
    deltas:  [B,N,2] where (t_c, t_w)
    returns: boxes [B,N,2] in feature coords, clipped to [0, Lf]
    """
    # anchor center/width
    a_start = anchors[:, 0]
    a_end   = anchors[:, 1]
    a_c = 0.5 * (a_start + a_end)
    a_w = (a_end - a_start).clamp(min=1e-6)

    t_c = deltas[..., 0]
    t_w = deltas[..., 1]

    # decode (center + log-width)
    c = a_c[None, :] + t_c * a_w[None, :]
    w = a_w[None, :] * torch.exp(t_w).clamp(max=50)  # avoid inf

    start = c - 0.5 * w
    end   = c + 0.5 * w

    # clip
    start = start.clamp(min=0.0, max=float(Lf))
    end   = end.clamp(min=0.0, max=float(Lf))

    # enforce start<end
    end = torch.max(end, start + 1e-3)
    return torch.stack([start, end], dim=-1)  # [B,N,2]


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


class RPN1D(nn.Module):
    """
    Input:  feat [B, 512, Lf]   (Lf = D/16)
    Output:
      obj_logits: [B, N]        (N = Lf*A) raw logits
      deltas:     [B, N, 2]     (t_c, t_w)
      anchors:    [N, 2]        anchor intervals in feature coords
    """
    def __init__(self, in_c=512, anchor_lengths=(2, 4, 6, 9, 13, 18)):
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
                score_thresh=0.1, pre_nms_topk=600, post_nms_topk=100, iou_thresh=0.5):
        B, N = obj_logits.shape
        scores = torch.sigmoid(obj_logits)  # [B,N]
        boxes = decode_deltas_to_boxes(anchors, deltas, Lf=Lf)  # [B,N,2]

        proposals = []
        scores_out = []
        for b in range(B):
            s = scores[b]
            bxs = boxes[b]

            keep = torch.where(s >= score_thresh)[0]

            # fallback: if nothing passes threshold, still take top-k so downstream has something
            if keep.numel() == 0:
                k = min(pre_nms_topk, s.numel())
                topk_idx = torch.topk(s, k).indices
                keep = topk_idx


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
    best_anchor_for_each_gt = iou.argmax(dim=0)  # [M]
    labels[best_anchor_for_each_gt] = 1
    matched_gt[best_anchor_for_each_gt] = gt_boxes

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

    for b in range(B):
        gt_boxes_slice = targets[b]["boxes"].to(device)  # [M,2] slice coords
        gt_boxes_feat = gt_boxes_slice / float(stride)   # feature coords

        labels, matched_gt = assign_rpn_targets_1d(
            anchors, gt_boxes_feat,
            pos_iou_thresh=pos_iou_thresh,
            neg_iou_thresh=neg_iou_thresh,
        )

        # sample anchors for stable training
        keep = sample_anchors(labels, batch_size=sample_size, pos_fraction=pos_fraction)
        if keep.numel() == 0:
            continue

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
    denom = max(1, B)
    total_obj = total_obj / denom
    total_reg = total_reg / denom
    loss = total_obj + reg_weight * total_reg

    stats = {
        "loss_total": float(loss.detach().cpu()),
        "loss_obj": float(total_obj.detach().cpu()),
        "loss_reg": float(total_reg.detach().cpu()),
        "pos_anchors": int(total_pos),
        "sampled_anchors": int(total_samp),
    }
    return loss, stats

####################### ROI POOLING ############################

def roi_pool_1d(
    feat: torch.Tensor,          # [B, C, Lf]
    proposals: list,             # list length B, each: [Pi,2] in FEATURE coords (float)
    roi_len: int = 16,
    pool: str = "max",           # "max" or "avg"
):
    """
    Returns:
      roi_feat:  [R, C, roi_len]  (R = total proposals across batch)
      roi_batch: [R]             (which batch item each RoI came from)
      roi_boxes: [R, 2]          (proposal boxes in FEATURE coords, clipped)
    """
    assert feat.dim() == 3, "feat must be [B,C,Lf]"
    B, C, Lf = feat.shape
    device = feat.device

    roi_feats = []
    roi_batch = []
    roi_boxes = []

    for b in range(B):
        props = proposals[b]
        if props is None or props.numel() == 0:
            continue

        # Ensure shape [Pi,2]
        if props.dim() == 1:
            props = props.view(1, 2)

        for i in range(props.shape[0]):
            s = float(props[i, 0].item())
            e = float(props[i, 1].item())

            # clip to [0, Lf]
            s = max(0.0, min(s, float(Lf)))
            e = max(0.0, min(e, float(Lf)))
            if e <= s:
                continue

            # convert to integer slicing range (inclusive-ish)
            s_idx = int(torch.floor(torch.tensor(s)).item())
            e_idx = int(torch.ceil(torch.tensor(e)).item())
            s_idx = max(0, min(s_idx, Lf - 1))
            e_idx = max(s_idx + 1, min(e_idx, Lf))  # ensure at least 1

            region = feat[b:b+1, :, s_idx:e_idx]  # [1,C,Lr]

            if pool == "max":
                pooled = F.adaptive_max_pool1d(region, roi_len)  # [1,C,roi_len]
            elif pool == "avg":
                pooled = F.adaptive_avg_pool1d(region, roi_len)
            else:
                raise ValueError("pool must be 'max' or 'avg'")

            roi_feats.append(pooled.squeeze(0))  # [C,roi_len]
            roi_batch.append(b)
            roi_boxes.append([s, e])

    if len(roi_feats) == 0:
        roi_feat = torch.zeros((0, C, roi_len), device=device, dtype=feat.dtype)
        roi_batch = torch.zeros((0,), device=device, dtype=torch.long)
        roi_boxes = torch.zeros((0, 2), device=device, dtype=torch.float32)
        return roi_feat, roi_batch, roi_boxes

    roi_feat = torch.stack(roi_feats, dim=0)  # [R,C,roi_len]
    roi_batch = torch.tensor(roi_batch, device=device, dtype=torch.long)  # [R]
    roi_boxes = torch.tensor(roi_boxes, device=device, dtype=torch.float32)  # [R,2]
    return roi_feat, roi_batch, roi_boxes

#################### ROI HEADS ##########################


def iou_1d_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # can just call interval_iou_1d for consistency
    return interval_iou_1d(a, b)


def sample_rois(labels: torch.Tensor, batch_size: int = 128, pos_fraction: float = 0.25) -> torch.Tensor:
    """
    labels: [P] with {1 pos, 0 neg, -1 ignore}
    returns indices to keep
    """
    device = labels.device
    pos = torch.where(labels == 1)[0]
    neg = torch.where(labels == 0)[0]

    num_pos = min(int(batch_size * pos_fraction), pos.numel())
    num_neg = min(batch_size - num_pos, neg.numel())

    keep = []
    if num_pos > 0:
        keep.append(pos[torch.randperm(pos.numel(), device=device)[:num_pos]])
    if num_neg > 0:
        keep.append(neg[torch.randperm(neg.numel(), device=device)[:num_neg]])
    if len(keep) == 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    return torch.cat(keep, dim=0)


def assign_rois_to_gt(
    proposals_slice: torch.Tensor,  # [P,2] slice coords
    gt_boxes_slice: torch.Tensor,   # [G,2] slice coords
    gt_plaque: torch.Tensor,        # [G] ints (0..3 in your setup)
    gt_sten: torch.Tensor,          # [G] ints (0..5)
    pos_iou_thresh: float = 0.5,
    neg_iou_thresh: float = 0.1,
):
    """
    Returns:
      labels: [P] {1 pos, 0 neg, -1 ignore}
      plaque_t: [P] int class target (0 background, else 1..3)
      sten_t:   [P] int class target (0 background, else 1..5)
      matched_gt: [P,2] best-matching GT box for each proposal (zeros if no GT)
    """
    device = proposals_slice.device
    P = proposals_slice.shape[0]

    labels = torch.full((P,), -1, dtype=torch.long, device=device)
    plaque_t = torch.zeros((P,), dtype=torch.long, device=device)
    sten_t = torch.zeros((P,), dtype=torch.long, device=device)
    matched_gt = torch.zeros((P, 2), dtype=torch.float32, device=device)

    if P == 0:
        return labels, plaque_t, sten_t, matched_gt

    # If no GT -> all negatives/background
    if gt_boxes_slice.numel() == 0:
        labels[:] = 0
        return labels, plaque_t, sten_t, matched_gt

    iou = iou_1d_matrix(proposals_slice, gt_boxes_slice)  # [P,G]
    max_iou, argmax = iou.max(dim=1)  # best GT for each proposal
    matched_gt = gt_boxes_slice[argmax].float()

    # neg/pos
    labels[max_iou < neg_iou_thresh] = 0
    labels[max_iou >= pos_iou_thresh] = 1

    # assign class targets for positives
    pos_idx = torch.where(labels == 1)[0]
    if pos_idx.numel() > 0:
        m = argmax[pos_idx]
        plaque_t[pos_idx] = gt_plaque[m].long()
        sten_t[pos_idx] = gt_sten[m].long()

    return labels, plaque_t, sten_t, matched_gt


class RoIHeads(nn.Module):
    """
    Inputs:
      roi_feat: [R, C=512, roi_len]

    Outputs:
      calc_logit:     [R]   (sigmoid -> P(calcified present))
      noncalc_logit:  [R]   (sigmoid -> P(non-calcified present))
      sten_logits:    [R, 6]  (softmax over 0..5; 0 can be background/normal)
      roi_deltas:     [R, 2]  (t_c, t_w) refinement deltas for proposal boxes
    """
    def __init__(self, in_c=512, roi_len=16, num_stenosis_classes=5):
        super().__init__()
        self.roi_len = roi_len

        self.fc = nn.Sequential(
            nn.Linear(in_c, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )

        self.calc_head = nn.Linear(256, 1)
        self.noncalc_head = nn.Linear(256, 1)

        # 5-way stenosis: internal labels 0..4 correspond to original grades 1..5
        self.sten_head = nn.Linear(256, num_stenosis_classes)

        self.roi_reg_head = nn.Linear(256, 2)

    def forward(self, roi_feat):
        x = roi_feat.mean(dim=-1)   # [R,C]
        x = self.fc(x)              # [R,256]

        calc_logit = self.calc_head(x).squeeze(-1)
        noncalc_logit = self.noncalc_head(x).squeeze(-1)
        sten_logits = self.sten_head(x)          # [R,5]
        roi_deltas = self.roi_reg_head(x)        # [R,2]
        return calc_logit, noncalc_logit, sten_logits, roi_deltas


def decode_deltas_to_boxes_1d(boxes, deltas, clip_min=0.0, clip_max=None):
    """
    boxes:  [R,2]  (start,end) in slice coords (or feature coords — just be consistent)
    deltas: [R,2]  (t_c, t_w)
    returns refined_boxes: [R,2]
    """
    b0, b1 = boxes[:, 0], boxes[:, 1]
    c = 0.5 * (b0 + b1)
    w = (b1 - b0).clamp(min=1e-6)

    t_c = deltas[:, 0]
    t_w = deltas[:, 1]

    c2 = c + t_c * w
    w2 = w * torch.exp(t_w).clamp(max=50)

    start = c2 - 0.5 * w2
    end   = c2 + 0.5 * w2

    if clip_max is not None:
        start = start.clamp(min=clip_min, max=clip_max)
        end   = end.clamp(min=clip_min, max=clip_max)
    else:
        start = start.clamp(min=clip_min)
        end   = end.clamp(min=clip_min)

    end = torch.max(end, start + 1e-3)
    return torch.stack([start, end], dim=-1)
