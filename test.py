# import pandas as pd
# import SimpleITK as sitk

# csv_path = "/home/joshua/Coronary_R-CNN/test_cpr_all26_allbranch_02mm.csv"
# df = pd.read_csv(csv_path, header=None, index_col=0)
# rows = df.values.tolist()

# print(df.count())



# count = 0

# for r in rows:
#     cpr_path = r[2]
#     itk = sitk.ReadImage(cpr_path)
#     vol = sitk.GetArrayFromImage(itk)  # [D,H,W]
#     if vol.shape[0] > 768:
#         count += 1

# print(count)

# fe_smoketest.py
import torch
from torch.utils.data import DataLoader

from dataprep import VesselWindowDataset, collate_fn
from detection_module import FeatureExtractorFE, RPN1D


def main():
    # ---- config ----
    csv_path = "/home/joshua/Coronary_R-CNN/train_cpr_all26_allbranch_02mm.csv"
    window_len = 768
    batch_size = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ---- dataset/loader ----
    ds = VesselWindowDataset(
        csv_path=csv_path,
        window_len=window_len,
        train=False,        # deterministic
        do_augment=False,   # disable aug for a clean test
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,      # set 0 first to avoid dataloader worker issues
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
    )

    # ---- model ----
    fe = FeatureExtractorFE(in_ch=1, norm="gn").to(device)
    fe.eval()

    # ---- one batch ----
    x, targets = next(iter(dl))  # x: [B,1,D,H,W]
    print("Input x:", x.shape, x.dtype, "min/max:", float(x.min()), float(x.max()))
    print("Example target keys:", targets[0].keys())
    print("Example boxes shape:", targets[0]["boxes"].shape)

    x = x.to(device, non_blocking=True)

    with torch.no_grad():
        y = fe(x)

    print("FE output y:", y.shape, y.dtype)
    expected_Lf = window_len // 16
    assert y.shape[0] == batch_size
    assert y.shape[1] == 512
    assert y.shape[2] == expected_Lf, f"Expected D/16={expected_Lf}, got {y.shape[2]}"
    print("✅ FE forward pass works and output shape matches [B,512,D/16].")




    fe = FeatureExtractorFE(in_ch=1, norm="gn").cuda().eval()
    rpn = RPN1D(in_c=512, anchor_lengths=(2,4,6,9,13,18)).cuda().eval()

    with torch.no_grad():
        feat = fe(x.cuda())                 # [B,512,48]
        obj_logits, deltas, anchors = rpn(feat)
        print("feat:", feat.shape)
        print("obj_logits:", obj_logits.shape)  # [B, N]
        print("deltas:", deltas.shape)          # [B, N, 2]
        print("anchors:", anchors.shape)        # [N, 2]

        Lf = feat.shape[-1]
        props = rpn.propose(obj_logits, deltas, anchors, Lf=Lf)
        print("num proposals per batch item:", [p.shape[0] for p in props])
        print("first proposal (feature coords):", props[0][0] if props[0].shape[0] else None)

if __name__ == "__main__":
    main()
