import random
from ast import literal_eval

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
import torchio as tio
from scipy import ndimage

def choose_window_start_focus_lesion(
    z_len: int,
    window_len: int,
    lesions_full,                 # list of (s,e,...) in SLICE coords
    train: bool = True,
    focus_prob: float = 0.8,      # probability to focus around a lesion if lesions exist
    jitter_frac: float = 0.25,    # random shift up to 25% of window length
) -> int:
    """
    Returns z0 for cropping [z0 : z0 + window_len] along slice axis.

    - If z_len <= window_len => z0=0 (padding handles shorter)
    - If no lesions OR not training => fallback to deterministic/random
    - If lesions exist and training:
        with probability focus_prob, pick a window likely containing a lesion
        otherwise random (keeps background variety)
    """
    if z_len <= window_len:
        return 0

    if not train:
        return 0

    # Ensure lesions_full is usable
    if not lesions_full:
        return random.randint(0, z_len - window_len)

    # Sometimes still do random crop (so model learns background-only too)
    if random.random() > focus_prob:
        return random.randint(0, z_len - window_len)

    # lesions_full can be [(s,e,pla,ste), ...] or [(s,e), ...]
    # we only need (s,e)
    s, e = random.choice([(x[0], x[1]) for x in lesions_full])

    # If a lesion is longer than the window, just start near lesion start
    if (e - s) >= window_len:
        z0 = int(s)
    else:
        center = 0.5 * (s + e)
        z0 = int(round(center - window_len / 2))

    # Jitter the crop start for augmentation
    jitter = int(round((random.random() * 2 - 1) * jitter_frac * window_len))
    z0 += jitter

    # Clamp to valid range
    z0 = max(0, min(z0, z_len - window_len))
    return z0


def augment_cpr_volume(data_array: np.ndarray) -> np.ndarray:
    """
    data_array: [D, H, W]
    returns:    [D, H, W] float32
    """
    data_tensor = torch.from_numpy(data_array).float().unsqueeze(0)  # [1, D, H, W]
    image = tio.ScalarImage(tensor=data_tensor)

    transform = tio.Compose([
        tio.RandomFlip(axes=(2,), p=0.5),                 # flip W axis
        tio.RandomGamma(log_gamma=(-0.2, 0.2), p=0.3),
        tio.RandomNoise(mean=0.0, std=(0, 3), p=0.3),
    ])

    augmented = transform(image)
    return augmented.tensor.squeeze(0).numpy().astype(np.float32)


def clip_hu(data_array: np.ndarray, hu_min: float = -300, hu_max: float = 900) -> np.ndarray:
    data_array = data_array.astype(np.float32)
    data_array[data_array < hu_min] = hu_min
    data_array[data_array > hu_max] = hu_max
    return data_array


def zscore_norm(volume: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m = float(np.mean(volume))
    s = float(np.std(volume))
    return (volume - m) / (s + eps)


def is_normal_list(x) -> bool:
    if not isinstance(x, (list, tuple)):
        return False
    return any(isinstance(v, str) and v.lower() == "normal" for v in x)


def parse_triplet_intervals(
    triplet_list,
    z_len: int,
    spacing_mm: float = 0.2,
):
    """
    triplet_list format:
      ['PATIENT', 'VESSEL', start_mm, end_mm, value, start_mm, end_mm, value, ...]

    returns list of tuples: (start_idx, end_idx, value)
    """
    intervals = []
    if not isinstance(triplet_list, (list, tuple)) or len(triplet_list) < 3:
        return intervals
    if is_normal_list(triplet_list):
        return intervals

    # Need at least: [id, vessel, s, e, v]
    if len(triplet_list) < 5:
        return intervals

    n = (len(triplet_list) - 2) // 3
    for k in range(n):
        s_mm = triplet_list[2 + 3 * k + 0]
        e_mm = triplet_list[2 + 3 * k + 1]
        val  = triplet_list[2 + 3 * k + 2]
        try:
            s = int(round(float(s_mm) / spacing_mm))
            e = int(round(float(e_mm) / spacing_mm))
        except Exception:
            continue

        s = max(0, s)
        e = min(z_len, e)
        if e > s:
            intervals.append((s, e, val))
    return intervals


def choose_window_start(z_len: int, window_len: int, train: bool = True) -> int:
    if z_len <= window_len:
        return 0
    return random.randint(0, z_len - window_len) if train else 0


def crop_or_pad_window(volume: np.ndarray, z0: int, window_len: int, pad_value: float = 0.0):
    """
    volume: [D,H,W]
    returns window: [L,H,W], valid_len
    """
    D, H, W = volume.shape
    L = window_len
    if D >= L:
        return volume[z0:z0 + L], L
    pad = np.full((L - D, H, W), pad_value, dtype=volume.dtype)
    return np.concatenate([volume, pad], axis=0), D


def remap_intervals_to_window(intervals, z0: int, window_len: int):
    """
    intervals: list of (s,e,val) in full indices
    returns:   list of (s_rel,e_rel,val) in window indices
    """
    ws, we = z0, z0 + window_len
    out = []
    for (s, e, val) in intervals:
        if e <= ws or s >= we:
            continue
        s_rel = max(0, s - ws)
        e_rel = min(window_len, e - ws)
        if e_rel > s_rel:
            out.append((s_rel, e_rel, val))
    return out


class VesselWindowDataset(Dataset):
    """
    Fixed-length CPR window dataset for vessel-wise detection.

    CSV (after index_col=0) expected columns:
      [patient_id, cpr_path, mask_path, stenosis_list_str, plaque_list_str]

    Returns:
      x:      FloatTensor [1, L, H, W]
      target: dict with:
        - boxes:    FloatTensor [N,2] (start,end) indices in window coords
        - plaque:   LongTensor  [N]   plaque label per interval
        - stenosis: FloatTensor [N]   stenosis value per interval (cast from your labels)
        - z0, orig_len, valid_len
        - id, vessel, cpr_path, mask_path
    """

    def __init__(
        self,
        csv_path: str,
        window_len: int = 480,
        spacing_mm: float = 0.2,
        train: bool = True,
        hu_min: float = -300,
        hu_max: float = 900,
        do_augment: bool = True,
        do_random_rotate: bool = True,
        pad_value: float = 0.0,
        normalize: str = "zscore",  # "zscore" or "none"
    ):
        super().__init__()

        df = pd.read_csv(csv_path, header=None, index_col=0)
        rows = df.values.tolist()

        self.patient_id = []
        self.cpr_path = []
        self.mask_path = []
        self.stenosis_list = []
        self.plaque_list = []
        self.vessel = []

        for r in rows:
            # r layout from your sample:
            # r[0]=patient_id, r[1]=cpr, r[2]=mask, r[3]=stenosis_str, r[4]=plaque_str
            pid = r[0]
            cpr = r[1]
            msk = r[2]

            sten = literal_eval(r[3])  # list
            plaq = literal_eval(r[4])  # list

            ves = None
            if isinstance(sten, (list, tuple)) and len(sten) > 1:
                ves = sten[1]
            elif isinstance(plaq, (list, tuple)) and len(plaq) > 1:
                ves = plaq[1]
            if ves is None:
                ves = "ves"

            self.patient_id.append(pid)
            self.cpr_path.append(cpr)
            self.mask_path.append(msk)
            self.stenosis_list.append(sten)
            self.plaque_list.append(plaq)
            self.vessel.append(ves)

        self.window_len = int(window_len)
        self.focus_prob = 0.9
        self.jitter_frac = 0.25
        self.spacing_mm = float(spacing_mm)
        self.train = bool(train)
        self.hu_min = float(hu_min)
        self.hu_max = float(hu_max)
        self.do_augment = bool(do_augment) and self.train
        self.do_random_rotate = bool(do_random_rotate) and self.train
        self.pad_value = float(pad_value)
        self.normalize = normalize

    def __len__(self):
        return len(self.patient_id)

    def __getitem__(self, index: int):
        cpr_path = self.cpr_path[index]
        itk = sitk.ReadImage(cpr_path)
        vol = sitk.GetArrayFromImage(itk)  # [D,H,W]
        vol = clip_hu(vol, self.hu_min, self.hu_max)

        if self.do_augment:
            vol = augment_cpr_volume(vol)

        if self.do_random_rotate and random.random() < 0.3:
            angle = random.randint(-45, 45)
            vol = ndimage.rotate(vol, angle, axes=(1, 2), reshape=False, order=1, mode="nearest")

        D, H, W = vol.shape
        if self.normalize == "zscore":
            vol = zscore_norm(vol)

        sten_list = self.stenosis_list[index]
        plaq_list = self.plaque_list[index]

        # parse full-vessel intervals
        sten_intervals = parse_triplet_intervals(sten_list, z_len=D, spacing_mm=self.spacing_mm)
        plaq_intervals = parse_triplet_intervals(plaq_list, z_len=D, spacing_mm=self.spacing_mm)

        # sanity: they should match (same start/end) for lesion cases
        # We'll merge by index; if mismatch, fall back to overlap matching.
        lesions_full = []
        if len(sten_intervals) == len(plaq_intervals) and len(sten_intervals) > 0:
            for (s1, e1, ste_val), (s2, e2, pla_val) in zip(sten_intervals, plaq_intervals):
                s = min(s1, s2)
                e = max(e1, e2)
                lesions_full.append((s, e, int(pla_val), float(ste_val)))
        else:
            # overlap-based merge (more robust)
            for (s, e, ste_val) in sten_intervals:
                best = None
                best_iou = 0.0
                for (ps, pe, pla_val) in plaq_intervals:
                    inter = max(0, min(e, pe) - max(s, ps))
                    union = max(e, pe) - min(s, ps)
                    iou = inter / union if union > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou
                        best = (ps, pe, pla_val)
                if best is None:
                    continue
                ps, pe, pla_val = best
                s2 = min(s, ps)
                e2 = max(e, pe)
                lesions_full.append((s2, e2, int(pla_val), float(ste_val)))

        # choose/crop/pad window
        z0 = choose_window_start_focus_lesion(
            z_len=D,
            window_len=self.window_len,
            lesions_full=lesions_full,   # slice coords
            train=self.train,
            focus_prob=self.focus_prob,
            jitter_frac=self.jitter_frac,
        )
        window, valid_len = crop_or_pad_window(vol, z0, self.window_len, pad_value=self.pad_value)

        # remap lesions to window coords
        lesions_win = remap_intervals_to_window([(s, e, (pla, ste)) for (s, e, pla, ste) in lesions_full], z0, self.window_len)

        boxes = []
        plaque = []
        stenosis = []
        for (s_rel, e_rel, valpair) in lesions_win:
            pla, ste = valpair
            boxes.append([float(s_rel), float(e_rel)])
            plaque.append(int(pla))
            stenosis.append(float(ste))

        x = torch.from_numpy(window).float().unsqueeze(0)  # [1,L,H,W]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),              # [N,2]
            "plaque": torch.tensor(plaque, dtype=torch.long),               # [N]
            "stenosis": torch.tensor(stenosis, dtype=torch.float32),         # [N]
            "z0": torch.tensor([z0], dtype=torch.long),
            "orig_len": torch.tensor([D], dtype=torch.long),
            "valid_len": torch.tensor([valid_len], dtype=torch.long),
            "patient_id": self.patient_id[index],
            "vessel": self.vessel[index],
            "id": f"{self.patient_id[index]}_{self.vessel[index]}",
            "cpr_path": cpr_path,
            "mask_path": self.mask_path[index],
        }

        return x, target


def collate_fn(batch):
    """
    xs:      [B, 1, L, H, W]
    targets: list[dict] length B
    """
    xs, targets = zip(*batch)
    xs = torch.stack(xs, dim=0)
    return xs, list(targets)
