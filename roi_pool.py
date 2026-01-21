import torch


def roi_pool_1d(
    feat: torch.Tensor,
    rois: list,
    roi_len: int = 16,
    aligned: bool = True,
):
    """
    1D RoIAlign-style pooling on a feature sequence.

    Args:
        feat: [B, C, L] feature sequence
        rois: list length B, each element is Tensor [P,2] of (start,end) in FEATURE coords
        roi_len: fixed output length per RoI
        aligned: if True, uses bin-center sampling (RoIAlign-like), else uses edge sampling

    Returns:
        pooled:   [R, C, roi_len] pooled features for all RoIs in the batch (R = sum P_b)
        batch_ix: [R] long tensor mapping each pooled RoI to its batch index
        rois_cat: [R,2] concatenated rois (feature coords)
    """
    assert feat.dim() == 3, f"feat must be [B,C,L], got {feat.shape}"
    B, C, L = feat.shape
    device = feat.device
    dtype = feat.dtype

    pooled_list = []
    batch_ix_list = []
    rois_list = []

    for b in range(B):
        rb = rois[b]
        if rb is None or rb.numel() == 0:
            continue

        rb = rb.to(device=device, dtype=torch.float32)
        P = rb.shape[0]

        start = rb[:, 0].clamp(0.0, float(L))
        end = rb[:, 1].clamp(0.0, float(L))
        end = torch.maximum(end, start + 1e-3)

        if aligned:
            t = (torch.arange(roi_len, device=device, dtype=torch.float32) + 0.5) / float(roi_len)
        else:
            if roi_len == 1:
                t = torch.zeros((1,), device=device, dtype=torch.float32)
            else:
                t = torch.arange(roi_len, device=device, dtype=torch.float32) / float(roi_len - 1)

        pos = start[:, None] + (end - start)[:, None] * t[None, :]  # [P, roi_len]
        pos = pos.clamp(0.0, float(L - 1))

        x0 = torch.floor(pos).to(torch.long)            # [P, roi_len]
        x1 = (x0 + 1).clamp(max=L - 1)                  # [P, roi_len]
        w = (pos - x0.to(pos.dtype)).to(dtype)          # [P, roi_len]
        fb = feat[b]  # [C, L]

        # Expand to [C, P, L] so gather can work with index [C, P, roi_len]
        fb_exp = fb.unsqueeze(1).expand(-1, P, -1)           # [C, P, L]

        x0e = x0.unsqueeze(0).expand(C, -1, -1)              # [C, P, roi_len]
        x1e = x1.unsqueeze(0).expand(C, -1, -1)              # [C, P, roi_len]

        v0 = torch.gather(fb_exp, dim=2, index=x0e)          # [C, P, roi_len]
        v1 = torch.gather(fb_exp, dim=2, index=x1e)          # [C, P, roi_len]

        pooled = (1.0 - w).unsqueeze(0) * v0 + w.unsqueeze(0) * v1  # [C,P,roi_len]
        pooled = pooled.permute(1, 0, 2).contiguous()               # [P,C,roi_len]

        pooled_list.append(pooled)
        batch_ix_list.append(torch.full((P,), b, device=device, dtype=torch.long))
        rois_list.append(torch.stack([start, end], dim=-1))

    if len(pooled_list) == 0:
        pooled = torch.empty((0, C, roi_len), device=device, dtype=dtype)
        batch_ix = torch.empty((0,), device=device, dtype=torch.long)
        rois_cat = torch.empty((0, 2), device=device, dtype=torch.float32)
        return pooled, batch_ix, rois_cat

    pooled = torch.cat(pooled_list, dim=0)     # [R,C,roi_len]
    batch_ix = torch.cat(batch_ix_list, dim=0) # [R]
    rois_cat = torch.cat(rois_list, dim=0)     # [R,2]
    return pooled, batch_ix, rois_cat
