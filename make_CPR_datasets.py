import os
import argparse
import random
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
from scipy import ndimage

# ---- import your helper funcs from dataprep.py ----
from dataprep import (
    clip_hu, zscore_norm,
    parse_triplet_intervals,
    choose_window_start_focus_lesion,
    choose_window_start,
    crop_or_pad_window,
    remap_intervals_to_window,
    augment_cpr_volume,
    stenosis_to_2class,  

)
def file_exists_mhd(path: str) -> bool:
    # For .mhd, the corresponding .raw/.zraw/etc is usually referenced inside the header.
    # But first we at least check the header exists.
    return isinstance(path, str) and os.path.isfile(path)

def has_modplus_lesion(sten_intervals) -> bool:
    return any(float(ste_val) > 2 for (_, _, ste_val) in sten_intervals)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rows(csv_path: str):
    df = pd.read_csv(csv_path, header=None, index_col=0)
    return df.values.tolist()


def merge_sten_plaque(sten_intervals, plaq_intervals):
    """
    Returns lesions_full list of tuples (s,e,pla,ste) in FULL slice coords.
    """
    lesions_full = []
    if len(sten_intervals) == len(plaq_intervals) and len(sten_intervals) > 0:
        for (s1, e1, ste_val), (s2, e2, pla_val) in zip(sten_intervals, plaq_intervals):
            s = min(s1, s2)
            e = max(e1, e2)
            lesions_full.append((s, e, int(pla_val), int(ste_val)))
        return lesions_full

    # overlap-based merge (fallback)
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
        lesions_full.append((min(s, ps), max(e, pe), int(pla_val), int(ste_val)))

    return lesions_full


def cache_split(
    split_name: str,
    csv_path: str,
    out_dir: str,
    window_len: int,
    spacing_mm: float,
    hu_min: float,
    hu_max: float,
    normalize: str,
    train_like: bool,          # if True: lesion-focused crop + augmentation options
    do_augment: bool,
    do_rotate: bool,
    K: int,
    K_modplus: int,
    seed: int,
    save_float16: bool,
    only_indices=None
):
    os.makedirs(out_dir, exist_ok=True)
    rows = load_rows(csv_path)

    print(f"\n[{split_name}] rows={len(rows)} out_dir={out_dir} K={K}")
    print(f"[{split_name}] train_like={train_like} do_augment={do_augment} do_rotate={do_rotate}")
    only = set(only_indices) if only_indices is not None else None

    for idx, r in enumerate(rows):
        if only is not None and idx not in only:
            continue
        pid = r[0]
        cpr_path = r[1]
        mask_path = r[2]
        plaque_list = literal_eval(r[3])
        sten_list = literal_eval(r[4])

        def fix_case_path(p: str) -> str:
            if not isinstance(p, str):
                return p
            # change "/SI_case_46/" -> "/SI_Case_46/" (adjust pattern to your real folder name)
            return p.replace("/SI_case_", "/SI_Case_")

        cpr_path = fix_case_path(cpr_path)
        mask_path = fix_case_path(mask_path)

        vessel = None
        if isinstance(plaque_list, (list, tuple)) and len(plaque_list) > 1:
            vessel = plaque_list[1]
        elif isinstance(sten_list, (list, tuple)) and len(sten_list) > 1:
            vessel = sten_list[1]
        vessel = "UNK" if vessel is None else str(vessel)

        # --- validate paths (skip bad rows instead of crashing) ---
        if not file_exists_mhd(cpr_path):
            print(f"[{split_name}] MISSING CPR: idx={idx} pid={pid} vessel={vessel} path={cpr_path}")
            continue
        if not file_exists_mhd(mask_path):
            print(f"[{split_name}] MISSING MASK: idx={idx} pid={pid} vessel={vessel} path={mask_path}")
            continue

        # Load volume once
        try:
            itk = sitk.ReadImage(cpr_path)
            vol = sitk.GetArrayFromImage(itk)  # [D,H,W]
        except Exception as e:
            print(f"[{split_name}] READ FAIL CPR: idx={idx} pid={pid} vessel={vessel} path={cpr_path} err={e}")
            continue

        vol = clip_hu(vol, hu_min, hu_max)


        D, H, W = vol.shape

        sten_intervals = parse_triplet_intervals(sten_list, z_len=D, spacing_mm=spacing_mm)
        plaq_intervals = parse_triplet_intervals(plaque_list, z_len=D, spacing_mm=spacing_mm)
        lesions_full = merge_sten_plaque(sten_intervals, plaq_intervals)

        is_modplus = has_modplus_lesion(sten_intervals)

        # choose how many cached variants to create
        if train_like:
            K_eff = K_modplus if is_modplus else K
        else:
            K_eff = K
        # Normalize (do this before augmentation so aug sees normalized values, matching your dataset)
        if normalize == "zscore":
            vol = zscore_norm(vol)

        for k in range(K_eff):
            # Make each cached variant reproducible if seed provided
            set_seed(seed + idx * 1000 + k)

            v = vol

            if do_augment and train_like and is_modplus:
                v = augment_cpr_volume(v)

            if do_rotate and train_like and is_modplus and random.random() < 0.3:
                angle = random.randint(-45, 45)
                v = ndimage.rotate(v, angle, axes=(1, 2), reshape=False, order=1, mode="nearest")

            # Choose crop start
            if train_like:
                z0 = choose_window_start_focus_lesion(
                    z_len=D,
                    window_len=window_len,
                    lesions_full=lesions_full,
                    train=True,
                    focus_prob=0.9,
                    jitter_frac=0.25,
                )
            else:
                z0 = choose_window_start(z_len=D, window_len=window_len, train=False)  # -> 0

            window, valid_len = crop_or_pad_window(v, z0, window_len, pad_value=0.0)

            # Remap lesions to window coords
            lesions_win = remap_intervals_to_window(
                [(s, e, (pla, ste)) for (s, e, pla, ste) in lesions_full],
                z0, window_len
            )

            boxes, plaque, stenosis = [], [], []
            for (s_rel, e_rel, valpair) in lesions_win:
                pla, ste = valpair
                boxes.append([float(s_rel), float(e_rel)])
                plaque.append(int(pla))
                stenosis.append(stenosis_to_2class(ste))

            x = torch.from_numpy(window).unsqueeze(0).float()  # [1,L,H,W]
            if save_float16:
                x = x.half()

            save_dict = {
                "x": x.cpu(),
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "plaque": torch.tensor(plaque, dtype=torch.long),
                "stenosis": torch.tensor(stenosis, dtype=torch.long),
                "meta": {
                    "split": split_name,
                    "idx": idx,
                    "aug_k": k,
                    "patient_id": str(pid),
                    "vessel": vessel,
                    "z0": int(z0),
                    "orig_len": int(D),
                    "valid_len": int(valid_len),
                    "cpr_path": str(cpr_path),
                    "mask_path": str(mask_path),
                }
            }

            fname = f"{idx:07d}_{pid}_{vessel}_z{z0}_k{k}.pt"
            torch.save(save_dict, os.path.join(out_dir, fname))

        if (idx + 1) % 50 == 0:
            print(f"[{split_name}] cached {idx+1}/{len(rows)}")

    print(f"[{split_name}] done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_root", required=True)

    ap.add_argument("--window_len", type=int, default=768)
    ap.add_argument("--spacing_mm", type=float, default=0.2)
    ap.add_argument("--hu_min", type=float, default=-300)
    ap.add_argument("--hu_max", type=float, default=900)
    ap.add_argument("--normalize", type=str, default="zscore", choices=["zscore", "none"])

    ap.add_argument("--K_train_base", type=int, default=2, help="K for normal + mild (1-2)")
    ap.add_argument("--K_train_modplus", type=int, default=8, help="K for moderate+ (3-5)")
    ap.add_argument("--K_test", type=int, default=1)

    ap.add_argument("--augment_train", action="store_true")
    ap.add_argument("--rotate_train", action="store_true")


    ap.add_argument("--augment_test", action="store_true")
    ap.add_argument("--rotate_test", action="store_true")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_float16", action="store_true")
    ap.add_argument("--only_idx_file", type=str, default="")

    args = ap.parse_args()

    out_train = os.path.join(args.out_root, "train")
    out_test = os.path.join(args.out_root, "test")

    only_idx = None
    if args.only_idx_file:
        with open(args.only_idx_file) as f:
            only_idx = [int(x.strip()) for x in f if x.strip()]


    cache_split(
        "train", args.train_csv, out_train,
        window_len=args.window_len,
        spacing_mm=args.spacing_mm,
        hu_min=args.hu_min, hu_max=args.hu_max,
        normalize=args.normalize,
        train_like=True,
        do_augment=args.augment_train,
        do_rotate=args.rotate_train,
        K=args.K_train_base,
        K_modplus=args.K_train_modplus,
        seed=args.seed,
        save_float16=args.save_float16,
        only_indices=only_idx
    )

    cache_split(
        "test", args.test_csv, out_test,
        window_len=args.window_len,
        spacing_mm=args.spacing_mm,
        hu_min=args.hu_min, hu_max=args.hu_max,
        normalize=args.normalize,
        train_like=False,
        do_augment=args.augment_test,
        do_rotate=args.rotate_test,
        K=args.K_test,
        K_modplus=args.K_test,
        seed=args.seed + 54321,
        save_float16=args.save_float16,
    )

    print("\nAll caching complete.")
    print("Train:", out_train)
    print("Test: ", out_test)


if __name__ == "__main__":
    main()
