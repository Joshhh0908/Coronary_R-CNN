import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataprep import VesselWindowDataset, collate_fn



train_csv = "/home/joshua/Coronary_R-CNN/train_cpr_all26_allbranch_02mm.csv"
val_csv   = "/home/joshua/Coronary_R-CNN/val_cpr_all26_allbranch_02mm.csv"
test_csv = "/home/joshua/Coronary_R-CNN/test_cpr_all26_allbranch_02mm.csv"

train_ds = VesselWindowDataset(
    csv_path=train_csv,
    window_len=480,
    train=True,
    do_augment=True,
)

val_ds = VesselWindowDataset(
    csv_path=val_csv,
    window_len=480,
    train=False,
    do_augment=False,
)

test_ds = VesselWindowDataset(
    csv_path=test_csv,
    window_len=480,
    train=False,
    do_augment=False,   
)


train_loader = DataLoader(
    train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_ds,
    batch_size=2,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn,
)

test_loader = DataLoader(
    test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn,
)


ds = VesselWindowDataset(train_csv, window_len=480, train=False, do_augment=False)
x, t = ds[0]  # x: [1,L,H,W]
vol = x.squeeze(0).numpy()  # [L,H,W]

mid = vol.shape[0] // 2
plt.imshow(vol[mid], cmap="gray")
plt.title(f"{t['id']}  z0={int(t['z0'])}  boxes={t['boxes'].shape[0]}")
plt.show()

print("Boxes (start,end):", t["boxes"])
print("Plaque:", t["plaque"])
print("Stenosis:", t["stenosis"])
