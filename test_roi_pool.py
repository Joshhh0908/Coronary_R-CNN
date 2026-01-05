# roi_smoketest.py
import torch
from torch.utils.data import DataLoader

from dataprep import VesselWindowDataset, collate_fn
from detection_module import FeatureExtractorFE, RPN1D, roi_pool_1d


def iou_1d(gt, pr):
    # gt: [G,2], pr: [P,2]
    if gt.numel() == 0 or pr.numel() == 0:
        return torch.zeros((gt.shape[0], pr.shape[0]), device=gt.device)
    inter = (torch.min(gt[:, None, 1], pr[None, :, 1]) - torch.max(gt[:, None, 0], pr[None, :, 0])).clamp(min=0)
    union = (torch.max(gt[:, None, 1], pr[None, :, 1]) - torch.min(gt[:, None, 0], pr[None, :, 0])).clamp(min=1e-6)
    return inter / union


def main():
    csv_path = "/home/joshua/Coronary_R-CNN/train_cpr_all26_allbranch_02mm.csv"
    window_len = 768
    batch_size = 2
    roi_len = 16
    stride = 16.0  # feature -> slice

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ds = VesselWindowDataset(
        csv_path=csv_path,
        window_len=window_len,
        train=False,
        do_augment=False,
        do_random_rotate=False,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device=="cuda"), collate_fn=collate_fn)

    x, targets = next(iter(dl))
    x = x.to(device)

    fe = FeatureExtractorFE(in_ch=1, norm="gn").to(device).eval()
    rpn = RPN1D(in_c=512, anchor_lengths=(2,4,6,9,13,18)).to(device).eval()

    with torch.no_grad():
        feat = fe(x)  # [B,512,Lf]
        obj_logits, deltas, anchors = rpn(feat)
        Lf = feat.shape[-1]

        proposals_feat = rpn.propose(
            obj_logits, deltas, anchors, Lf=Lf,
            score_thresh=0.3, pre_nms_topk=600, post_nms_topk=50, iou_thresh=0.5
        )

        roi_feat, roi_batch, roi_boxes_feat = roi_pool_1d(feat, proposals_feat, roi_len=roi_len, pool="max")

    print("feat:", feat.shape)
    print("proposals per item:", [p.shape[0] for p in proposals_feat])
    print("roi_feat:", roi_feat.shape)      # [R,512,roi_len]
    print("roi_batch:", roi_batch.shape)    # [R]
    print("roi_boxes_feat:", roi_boxes_feat.shape)

    # show mapping back to slice coords for first RoI
    if roi_boxes_feat.shape[0] > 0:
        first = roi_boxes_feat[0]
        print("First RoI (feature coords):", first)
        print("First RoI (slice coords):  ", (first * stride).cpu())

    # quick IoU sanity: compare first sample's GT with first sample's proposals (in slice coords)
    gt0 = targets[0]["boxes"].to(device)
    pr0 = proposals_feat[0] * stride
    if gt0.numel() > 0 and pr0.numel() > 0:
        ious = iou_1d(gt0, pr0)
        best = ious.max(dim=1).values
        print("Sample0: GT count:", gt0.shape[0], "Proposal count:", pr0.shape[0])
        print("Sample0: best IoU per GT (first 5):", best[:5].detach().cpu())


if __name__ == "__main__":
    main()
