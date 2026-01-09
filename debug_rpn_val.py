import torch
from torch.utils.data import DataLoader

from FE import FeatureExtractorFE
from RPN import RPN1D
from cached_ds import CachedWindowDataset, collate_fn
from FE_RPN import evaluate_rpn   # import your function with debug prints enabled

def main():
    data = "/home/joshua/Coronary_R-CNN/data"
    ckpt_path = "/home/joshua/Coronary_R-CNN/RPN_results/fe_rpn_20260108_175945/checkpoints/epoch_004.pt"
    split = "val"  # or "test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = CachedWindowDataset(data, split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    ckpt = torch.load(ckpt_path, map_location=device)
    fe = FeatureExtractorFE(in_ch=1, norm="gn").to(device)
    rpn = RPN1D(in_c=512, anchor_lengths=tuple(ckpt.get("anchor_lengths", (1,2,3,5,7)))).to(device)
    fe.load_state_dict(ckpt["fe"])
    rpn.load_state_dict(ckpt["rpn"])
    metrics = evaluate_rpn(
        fe, rpn, loader, device,
        stride=16.0,
        score_thresh=0.1,
        pre_nms_topk=600,
        post_nms_topk=100,
        nms_iou=0.5,
        topk_for_recall=50,
        max_batches=None,   # IMPORTANT
    )

    print("\n=== METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
