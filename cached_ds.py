import os
import glob
import torch
from torch.utils.data import Dataset

class CachedWindowDataset(Dataset):
    """
    Reads .pt files created by your caching script.
    Each .pt contains: x, boxes, plaque, stenosis, meta
    """
    def __init__(self, root_dir: str, split: str):
        self.dir = os.path.join(root_dir, split)
        self.files = sorted(glob.glob(os.path.join(self.dir, "*.pt")))
        if len(self.files) == 0:
            raise RuntimeError(f"No .pt files found in {self.dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(self.files[idx], map_location="cpu")

        x = d["x"]  # [1,L,H,W] float16 or float32
        if x.dtype == torch.float16:
            x = x.float()  # FE expects float32

        boxes = d["boxes"].float()
        if boxes.numel() == 0:
            boxes = boxes.reshape(0, 2)

        plaque = d["plaque"].long()
        sten = d["stenosis"].long()
        sten = d["stenosis"].long()
        if sten.numel() > 0:
            assert sten.min() >= 0 and sten.max() < 2


        target = {"boxes": boxes, "plaque": plaque, "stenosis": sten}
        # optional: keep meta around for debugging
        if "meta" in d:
            target.update(d["meta"])

        return x, target


def collate_fn(batch):
    """
    xs:      [B, 1, L, H, W]
    targets: list[dict] length B
    """
    xs, targets = zip(*batch)
    xs = torch.stack(xs, dim=0)
    return xs, list(targets)
